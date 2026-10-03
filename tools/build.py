#!/usr/bin/env python3
"""Compile chapters 1--7 until references settle, check their layout, and merge their PDFs.

Run: python tools/build.py
Optional: --figures rebuilds the 18 figures with supplied TeX sources first.
The four legacy PDF-only diagrams are retained, not reconstructed.
Every chapter goes through tools/layout_check.py (clipped frames, text past the
margin, near-empty slides, lead-ins separated from their display); any issue
fails the build after all PDFs are written, with the slide and source line.
Requires pdflatex, poppler (pdftoppm, pdftotext) and numpy; PyMuPDF (or
poppler's pdfunite as fallback) for the combined PDF.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from layout_check import check_chapter

ROOT = Path(__file__).resolve().parents[1]
TITLES = {1:'Complex Analysis',2:'Gamma, Beta, Zeta',3:'Method of Frobenius',
          4:'Sturm--Liouville Theory',5:'Bessel Functions',6:'Orthogonal Polynomials',
          7:'Calculus of Variations'}

def compile_tex(source: Path, cwd: Path, output: Path, passes: int) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    text=''
    # Start with the requested passes; allow two more for references after re-pagination.
    for p in range(passes+2):
        run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',
                            '-file-line-error',f'-output-directory={output}',source.name],
                           cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,errors='replace')
        text=run.stdout
        (output/f'{source.stem}.pass{p+1}.txt').write_text(text)
        if run.returncode:
            raise RuntimeError(f'LaTeX failed: {source.name}; see {output}')
        rerun=any(msg in text for msg in ['Label(s) may have changed', 'Rerun to get', 'rerunfilecheck Warning'])
        if p+1>=passes and not rerun:
            break
    warnings=[line for line in text.splitlines() if any(s in line for s in
              ['Warning:', 'Overfull \\hbox', 'Overfull \\vbox', 'undefined references'])]
    # Overfull boxes (sub-point beamer artifacts) are reported, not fatal.
    # Fatal warnings are real LaTeX warnings: undefined references, labels changed.
    errors=[line for line in warnings if 'Overfull' not in line]
    return {'source':source.name,'passes':p+1,'warnings':warnings,'errors':errors}

def _merge_with_pdfunite(output: Path, parts: list[Path]) -> dict:
    """Fallback merger when PyMuPDF is unavailable. No bookmarks, just concatenation."""
    if not shutil.which('pdfunite'):
        raise RuntimeError('Chapter PDFs were compiled. Install PyMuPDF to merge them: python -m pip install pymupdf (or poppler for pdfunite)')
    target = output / 'mp_original_based_notes.pdf'
    run = subprocess.run(['pdfunite', *(str(p) for p in parts), str(target)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors='replace')
    if run.returncode:
        raise RuntimeError(f'pdfunite failed to merge chapters:\n{run.stdout}')
    counts = {}
    for n, p in enumerate(parts, start=1):
        if shutil.which('pdfinfo'):
            info = subprocess.run(['pdfinfo', str(p)], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors='replace')
            for line in info.stdout.splitlines():
                if line.startswith('Pages:'):
                    counts[str(n)] = int(line.split(':')[1].strip())
                    break
        counts.setdefault(str(n), 0)
    print('Merged without bookmarks (pdfunite fallback; install PyMuPDF for chapter bookmarks).')
    return counts

def merge_pdfs(output: Path) -> dict:
    parts = [output / f'mp_ch{n}.pdf' for n in range(1, 8)]
    missing = [str(p) for p in parts if not p.exists()]
    if missing:
        raise RuntimeError(f'Chapter PDFs missing, cannot merge: {missing}')
    try: import fitz
    except ImportError:
        return _merge_with_pdfunite(output, parts)
    merged=fitz.open();toc=[];counts={}
    for n in range(1,8):
        with fitz.open(output/f'mp_ch{n}.pdf') as part:
            offset=len(merged);counts[str(n)]=len(part)
            toc.append([1,f'Chapter {n}: {TITLES[n]}',offset+1])
            toc.extend([[level+1,title,page+offset] for level,title,page in part.get_toc() if page>0])
            merged.insert_pdf(part,links=True,annots=True)
    merged.set_toc(toc)
    merged.set_metadata({'title':'Mathematical Physics: Chapters 1--7',
                         'author':'Compiled by Jake W. Liu',
                         'subject':'Selective revision of the original lecture notes'})
    target=output/'mp_original_based_notes.pdf'
    merged.save(target,garbage=4,deflate=True);merged.close()
    return counts

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',action='store_true',help='rebuild available figure sources and update their PDFs under figs/')
    parser.add_argument('--output',type=Path,default=ROOT/'build',help='output directory; default is build/')
    args=parser.parse_args()
    if not shutil.which('pdflatex'):raise SystemExit('pdflatex was not found. Install a LaTeX distribution with Beamer, TikZ/PGFPlots, standalone, and Latin Modern.')
    missing=[tool for tool in ('pdftoppm','pdftotext') if not shutil.which(tool)]
    if missing:raise SystemExit(f'{", ".join(missing)} not found; install poppler for the layout check.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    report={'figures':[],'chapters':[]}
    if args.figures:
        dest=output/'figures';dest.mkdir(exist_ok=True)
        for f in sorted((ROOT/'figs').glob('*.tex')):
            print('Figure:',f.name,flush=True)
            report['figures'].append(compile_tex(f,ROOT/'figs',dest,1))
            shutil.copy2(dest/f'{f.stem}.pdf',ROOT/'figs'/f'{f.stem}.pdf')
    for n in range(1,8):
        f=ROOT/f'mp_ch{n}.tex'
        print('Chapter:',n,flush=True)
        result=compile_tex(f,ROOT,output,2)
        issues=check_chapter(output/f'mp_ch{n}.pdf')
        result['layout']=[asdict(i) for i in issues]
        for issue in issues:print('  layout:',issue,flush=True)
        report['chapters'].append(result)
    report['pages']=merge_pdfs(output)
    report['total_pages']=sum(report['pages'].values())
    (output/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Combined PDF:',output/'mp_original_based_notes.pdf')
    print('Total pages:',report['total_pages'])
    if any(c['errors'] for c in report['chapters']+report['figures']):
        raise SystemExit('Build completed with LaTeX warnings; inspect build_report.json.')
    layout=sum(len(c['layout']) for c in report['chapters'])
    if layout:
        raise SystemExit(f'Layout check failed: {layout} issue(s), listed above and in build_report.json. '
                         'Fix each frame, then rebuild.')
    print('Layout check passed for all chapters.')
    if any(c['warnings'] for c in report['chapters']+report['figures']):
        print('Note: minor overfull-box warnings present; see build_report.json.')

if __name__=='__main__':main()
