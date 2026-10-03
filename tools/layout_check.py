#!/usr/bin/env python3
"""Layout checks for one compiled Beamer chapter; build.py runs them after every chapter.

Run alone:  python3 tools/layout_check.py build/mp_ch3.pdf [more PDFs...]
(each PDF needs its .log and .nav next to it, and its source mp_chN.tex in the repo root).

Every rule below reports the slide, the frame title and the source line of the frame.

  overfull-vbox    Frame content taller than the slide, so the bottom is clipped.
                   Any amount is an error.
  overfull-hbox    Text or math running past the right margin by more than HBOX_TOL_PT.
  short-slide      A slide of a frame that breaks across slides (allowframebreaks or
                   \\framebreak) fills less than MIN_FILL of the slide body: a few leftover
                   lines alone on a slide, or a manual break placed too early.
  stranded-lead-in A slide that continues on the next slide of the same frame ends with
                   ':', so the display it introduces landed on the next slide.

Slide fill is measured on a 72-dpi grayscale render (1 pixel = 1 pt): the title bar and the
footline are the full-width colored bands at the top and bottom of the page, and the body is
the white region between them.  Requires poppler (pdftoppm, pdftotext) and numpy.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# TeX reports every box more than \hfuzz = 0.1pt too wide.  Overflows below 1pt are not
# visible on a slide; anything larger shows as text touching or crossing the margin.
HBOX_TOL_PT = 1.0
# A slide filling under a quarter of the body holds about four text lines or fewer; the
# frame should be rebalanced (break elsewhere, or tighten the content) instead.  Measured
# fills have no natural gap, so this is a chosen line: every slide at or below it that was
# inspected looked nearly empty.
MIN_FILL = 0.25
# Rows are part of a colored bar when at least this fraction of their pixels is non-white.
BAR_ROW_FRACTION = 0.9
# A pixel darker than this (0 = black, 255 = white) counts as ink.
INK_LEVEL = 200


@dataclass
class Issue:
    rule: str
    chapter: str
    slide: int | None
    source_line: int | None
    frame_title: str
    detail: str

    def __str__(self) -> str:
        where = f'{self.chapter}.tex:{self.source_line}' if self.source_line else f'{self.chapter}.tex'
        slide = f' slide {self.slide}' if self.slide else ''
        return f'[{self.rule}] {where}{slide} "{self.frame_title}": {self.detail}'


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='replace')
    if out.returncode:
        raise RuntimeError(f'{cmd[0]} failed: {out.stderr.strip()}')
    return out.stdout


def _braced(text: str, start: int) -> tuple[str, int] | None:
    """Return the contents of the {...} group opening at text[start], and the index after it."""
    if start >= len(text) or text[start] != '{':
        return None
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == '\\':            # skip the escaped character (\{, \}, \\)
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return None


@dataclass
class Frame:
    line: int          # source line of \\begin{frame}
    title: str
    breaks: bool       # allowframebreaks option or a manual \\framebreak in the body


def _commented(text: str, pos: int) -> bool:
    line_start = text.rfind('\n', 0, pos) + 1
    return '%' in text[line_start:pos].replace('\\%', '')


def source_frames(tex: Path) -> list[Frame]:
    """Frames in the order Beamer typesets them: every \\begin{frame} of the document, plus one
    frame per \\section when the preamble's \\AtBeginSection inserts a contents frame."""
    text = tex.read_text(errors='replace')
    template = None                      # span of the \AtBeginSection{...} body, if any
    hook = re.search(r'\\AtBeginSection\s*(\[[^\]]*\])?\s*', text)
    if hook and not _commented(text, hook.start()):
        body = _braced(text, hook.end())
        if body and '\\begin{frame}' in body[0]:
            template = (hook.end(), body[1])
    events = []
    for m in re.finditer(r'\\begin\{frame\}', text):
        if _commented(text, m.start()) or (template and template[0] <= m.start() < template[1]):
            continue
        events.append((m.start(), m))
    if template:
        for m in re.finditer(r'\\section\s*\{', text):
            if not _commented(text, m.start()) and m.start() > template[1]:
                events.append((m.start(), None))
    frames = []
    for pos, m in sorted(events, key=lambda e: e[0]):
        line = text.count('\n', 0, pos) + 1
        if m is None:
            frames.append(Frame(line, '(section contents frame)', False))
            continue
        end = text.find('\\end{frame}', m.end())
        end = len(text) if end < 0 else end
        i, options = m.end(), ''
        if i < len(text) and text[i] == '<':            # overlay specification
            i = text.index('>', i) + 1
        if i < len(text) and text[i] == '[':
            close = text.index(']', i)
            options, i = text[i + 1:close], close + 1
        group = _braced(text, i)
        if group is None:
            ft = re.compile(r'\\frametitle\s*').search(text, i, end)
            group = _braced(text, ft.end()) if ft else None
        title = ' '.join(group[0].split()) if group else '(untitled)'
        breaks = 'allowframebreaks' in options or '\\framebreak' in text[i:end]
        frames.append(Frame(line, title, breaks))
    return frames


def slide_frames(nav: Path) -> list[list[int]]:
    """Slides of each frame, in order, from Beamer's \\slideentry records."""
    groups: dict[int, list[int]] = {}
    for first, page in re.findall(r'\\slideentry \{[^}]*\}\{[^}]*\}\{[^}]*\}\{(\d+)/(\d+)\}', nav.read_text()):
        groups.setdefault(int(first), []).append(int(page))
    return [sorted(set(groups[k])) for k in sorted(groups)]


def frame_at_line(frames: list[Frame], line: int) -> Frame | None:
    best = None
    for f in frames:
        if f.line <= line:
            best = f
    return best


def log_issues(log: Path, chapter: str, frames: list[Frame]) -> list[Issue]:
    issues = []
    for m in re.finditer(r'Overfull \\(vbox|hbox) \(([\d.]+)pt too (?:high|wide)\)([^\n]*)',
                         log.read_text(errors='replace')):
        kind, amount, rest = m.group(1), float(m.group(2)), m.group(3)
        if kind == 'hbox' and amount <= HBOX_TOL_PT:
            continue
        what = 'content taller than the slide' if kind == 'vbox' else 'line wider than the text area'
        at = re.search(r'lines? (\d+)', rest)
        if at is None:     # e.g. "has occurred while \\output is active": no source line given
            issues.append(Issue(f'overfull-{kind}', chapter, None, None, '(unknown frame)',
                                f'{what} by {amount:.1f}pt; the log gives no line ({rest.strip()})'))
            continue
        line = int(at.group(1))
        frame = frame_at_line(frames, line)
        issues.append(Issue(f'overfull-{kind}', chapter, None, line,
                            frame.title if frame else '(before first frame)',
                            f'{what} by {amount:.1f}pt'
                            + (f' (frame starts at line {frame.line})' if frame else '')))
    return issues


def _read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, w, h, _maxval, data = raw.split(maxsplit=4)
    if magic != b'P5':
        raise RuntimeError(f'unexpected image format in {path}')
    w, h = int(w), int(h)
    return np.frombuffer(data[:w * h], dtype=np.uint8).reshape(h, w)


def body_band(img: np.ndarray) -> tuple[int, int] | None:
    """(first, last) body row between the top bar and the footline, or None if not found."""
    bar = (img < 245).mean(axis=1) >= BAR_ROW_FRACTION
    h = len(bar)
    top = 0
    while top < h and bar[top]:
        top += 1
    bottom = h - 1
    while bottom > top and not bar[bottom] and bottom > h - 4:   # antialiased last row
        bottom -= 1
    if not bar[bottom]:
        return None
    while bottom > top and bar[bottom]:
        bottom -= 1
    if top == 0 or bottom <= top:
        return None
    return top, bottom


def slide_issues(pdf: Path, chapter: str, frames: list[Frame],
                 groups: list[list[int]]) -> list[Issue]:
    issues = []
    mapped = len(groups) == len(frames)
    if not mapped:
        issues.append(Issue('frame-map', chapter, None, None, '(all)',
                            f'{len(frames)} \\begin{{frame}} in source but {len(groups)} frames in '
                            f'the .nav file; every multi-slide frame is checked and reported '
                            f'by slide number only'))
    with tempfile.TemporaryDirectory() as tmp:
        _run(['pdftoppm', '-gray', '-r', '72', str(pdf), f'{tmp}/p'])
        images = sorted(Path(tmp).glob('p-*.pgm'), key=lambda p: int(p.stem.split('-')[1]))
        for k, pages in enumerate(groups):
            if len(pages) < 2:
                continue
            if mapped and not frames[k].breaks:      # overlay slides, not a broken frame
                continue
            fline, title = (frames[k].line, frames[k].title) if mapped else (None, f'frame {k + 1}')
            for pos, page in enumerate(pages):
                img = _read_pgm(images[page - 1])
                band = body_band(img)
                if band is None:
                    issues.append(Issue('unmeasured', chapter, page, fline, title,
                                        'could not locate the title bar and footline'))
                    continue
                top, bottom = band
                ink = np.nonzero((img[top:bottom + 1] < INK_LEVEL).any(axis=1))[0]
                fill = (ink[-1] + 1) / (bottom - top + 1) if len(ink) else 0.0
                if fill < MIN_FILL:
                    which = 'last slide' if pos == len(pages) - 1 else f'slide {pos + 1} of {len(pages)}'
                    issues.append(Issue('short-slide', chapter, page, fline, title,
                                        f'{which} of the frame is only {fill:.0%} full'))
                if pos < len(pages) - 1:
                    body = _run(['pdftotext', '-f', str(page), '-l', str(page), '-r', '72',
                                 '-x', '0', '-y', str(top), '-W', str(img.shape[1]),
                                 '-H', str(bottom - top + 1), '-layout', str(pdf), '-'])
                    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
                    if lines and lines[-1].endswith(':'):
                        issues.append(Issue('stranded-lead-in', chapter, page, fline, title,
                                            f'slide ends with "{lines[-1][-60:]}" and the display '
                                            f'it introduces is on slide {page + 1}'))
    return issues


def check_chapter(pdf: Path) -> list[Issue]:
    chapter = pdf.stem
    tex = ROOT / f'{chapter}.tex'
    frames = source_frames(tex)
    issues = log_issues(pdf.with_suffix('.log'), chapter, frames)
    issues += slide_issues(pdf, chapter, frames, slide_frames(pdf.with_suffix('.nav')))
    return issues


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    total = 0
    for arg in argv:
        issues = check_chapter(Path(arg).resolve())
        for issue in issues:
            print(issue)
        total += len(issues)
    print(f'{total} layout issue(s)')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

