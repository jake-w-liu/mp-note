#!/usr/bin/env python3
"""Targeted checks for the second trimming pass, plus source-structure checks.

The algebra is independently transcribed, not extracted from arbitrary LaTeX.
Hashes identify the exact reviewed sources. These tests complement verify_math.py;
they do not constitute a proof of all theorems or of pedagogical continuity.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
RESULTS=[]
x,t,r,lam=s.symbols('x t r lambda', real=True)

def add(kind,name,ok,detail=''):
    RESULTS.append({'type':kind,'check':name,'passed':bool(ok),'detail':detail})
def exact(name,expr):
    v=s.simplify(s.expand(expr))
    add('mathematical',name,v==0,str(v))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def mathematical():
    # Airy: no indicial exponent is needed at an ordinary point.
    for a0,a1 in [(1,0),(0,1),(2,-3)]:
        a=[s.Rational(a0),s.Rational(a1),s.Integer(0)]+[s.Integer(0)]*19
        for k in range(19):a[k+3]=a[k]/((k+3)*(k+2))
        y=sum(v*x**j for j,v in enumerate(a))
        residual=s.Poly(s.diff(y,x,2)-x*y,x)
        for j in range(20):exact(f'Airy ({a0},{a1}) residual coefficient {j}',residual.nth(j))
        exact(f'Airy ({a0},{a1}) initial value',y.subs(x,0)-a0)
        exact(f'Airy ({a0},{a1}) initial slope',s.diff(y,x).subs(x,0)-a1)
    # Formal Laurent shifts used by the shorter recurrence derivation.
    c={j:s.Symbol(f'c{j:+d}') for j in range(-7,8)}
    g=sum(v*t**j for j,v in c.items())
    for m in range(-6,7):
        exact(f'[t^{m}] of t*g',s.expand(t*g).coeff(t,m)-c[m-1])
        exact(f'[t^{m}] of g/t',s.expand(g/t).coeff(t,m)-c[m+1])
        exact(f'[t^{m}] of t*dg/dt',s.expand(t*s.diff(g,t)).coeff(t,m)-m*c[m])
    # One-sided Bessel formulas arise by adding/subtracting, not new reindexing.
    A,B,J,D,nu=s.symbols('A B J D nu')
    eqsum=2*nu*J/x;eqdiff=2*D
    exact('Bessel lower neighbor elimination',(eqsum+eqdiff)/2-(nu*J/x+D))
    exact('Bessel upper neighbor elimination',(eqsum-eqdiff)/2-(nu*J/x-D))
    for m in range(5):
        for k in range(1,5):
            a=lambda j:(-1)**j/(s.factorial(j)*s.factorial(m+j)*2**(2*j+m))
            exact(f'Bessel Frobenius cancellation m={m},k={k}',4*k*(k+m)*a(k)+a(k-1))
    # The indentation is clockwise: theta goes from pi down to zero.
    exact('Clockwise upper indentation sign',s.integrate(s.I*r,(t,s.pi,0))+s.I*s.pi*r)
    # Product rule and integrating factor, including the spectral term.
    y=s.Function('y')(x)
    for p0,p1,Q,r0,r1 in [(s.Integer(1),-2*x,s.exp(-x*x),0,1),
                            (s.Integer(2),s.Integer(2),s.exp(x),-3,4),
                            (1+x*x,2*x,1+x*x,-3,2+x)]:
        exact(f'Integrating factor Q={Q}',s.diff(Q,x)-Q*p1/p0)
        lhs=Q/p0*(p0*s.diff(y,x,2)+p1*s.diff(y,x)+(r0+lam*r1)*y)
        rhs=s.diff(Q*s.diff(y,x),x)+Q*r0/p0*y+lam*Q*r1/p0*y
        exact(f'SL conversion Q={Q}',lhs-rhs)
    # Norm derivation includes degree zero separately, and its beta substitution.
    for n in range(9):
        P=s.legendre(n,x)
        exact(f'Legendre norm n={n}',s.integrate(P**2,(x,-1,1))-s.Rational(2,2*n+1))
        B=s.factorial(n)**2/s.factorial(2*n+1)
        exact(f'Legendre beta substitution n={n}',s.integrate((1-x*x)**n,(x,-1,1))-2**(2*n+1)*B)
    # No quotient a[k+1]/a[k] is formed, even beyond termination.
    for n in range(8):
        a=lambda k:(-1)**k*s.binomial(n,k)/s.factorial(k) if 0<=k<=n else s.Integer(0)
        for k in range(n+3):
            exact(f'Laguerre terminating recurrence n={n},k={k}',(k+1)**2*a(k+1)+(n-k)*a(k))
    # Gaussian factor in the oscillator and the polar change of area.
    f=s.exp(-x*x/2)
    exact('Oscillator product second derivative',s.diff(f*y,x,2)-f*(s.diff(y,x,2)-2*x*s.diff(y,x)+(x*x-1)*y))
    jac=s.Matrix([[s.diff(r*s.cos(t),r),s.diff(r*s.cos(t),t)],
                  [s.diff(r*s.sin(t),r),s.diff(r*s.sin(t),t)]]).det()
    exact('Polar area factor',jac-r)
    exact('Quartic beta answer retained',s.gamma(s.Rational(1,4))*s.sqrt(s.pi)/(4*s.gamma(s.Rational(3,4)))-s.beta(s.Rational(1,4),s.Rational(1,2))/4)

def structural():
    sources={n:(ROOT/f'mp_ch{n}.tex').read_text() for n in range(1,8)}
    for n,src in sources.items():
        labs=re.findall(r'\\label\{([^}]+)\}',src)
        refs=re.findall(r'\\(?:eqref|ref|pageref|autoref)\{([^}]+)\}',src)
        add('source',f'Chapter {n} unique labels',len(labs)==len(set(labs)))
        add('source',f'Chapter {n} references resolve',set(refs)<=set(labs),str(sorted(set(refs)-set(labs))))
        add('source',f'Chapter {n} required supplied style',r'\usepackage{Kyushu}' in src)
        add('source',f'Chapter {n} explicit breaks end paragraphs',not re.search(r'(?<!\\par)\\framebreak',src))
        add('source',f'Chapter {n} document environment',src.count(r'\begin{document}')==src.count(r'\end{document}')==1)
    assets=set()
    for src in sources.values():
        assets.update(re.findall(r'\\(?:safeincludegraphics|includegraphics)(?:\[[^\]]*\])?\s*\{([^}]+)\}',src))
    assets.discard('#2')
    for name in sorted(assets):
        p=ROOT/name
        add('source',f'Figure exists and is PDF: {name}',p.is_file() and p.read_bytes().startswith(b'%PDF-'))
    add('source','21 referenced figure PDFs retained',len(assets)==21,str(len(assets)))
    original_style='f8def167e9ea7ec09f708dfb4666cd20723102fc4f0409795bce47758ecf79d1'
    add('source','Supplied style byte-identical',digest(ROOT/'Kyushu.sty')==original_style)
    a=sources[3].index('{Choose the series by classifying the point}')
    b=sources[3].index('{The method of Frobenius: the recipe}')
    add('source','Classification test precedes recipe',a<b)
    airy=sources[3].split("{Example: Airy's equation by series}",1)[1].split(r'\end{frame}',1)[0]
    add('source','Airy uses ordinary series, not general-r detour',r'a_nx^n' in airy and 'k+r' not in airy)
    add('source','Airy includes index alignment and both initial values',all(z in airy for z in ['j(j-1)', 'a_{n-1}',r'a_0=y(0)',r"a_1=y'(0)"]))
    add('source','Taylor equality uses the remainder-to-zero criterion',
        all(z in sources[1] for z in [r'R_N(x)=f(x)-T_N(x;x_0)',
                                      r'tends to zero as',r'N\to\infty']))
    add('source','Both Bessel endpoint cases retained',all(z in sources[5] for z in [r'm\ge1&O(s^m)',r'm=0&1+O(s^2)']))
    add('source','Green identity retained',"Green's identity" in sources[4] and 'overline' in sources[4])
    add('source','Legendre norm explicitly handles n=0','For $n=0$' in sources[6] and r'I_0=2' in sources[6])
    add('source','No undefined generalized Laguerre symbol',r'L_n^{(\alpha)}' not in sources[6])
    add('source','Hermite extraction convention repaired',r'[t^n]/n!' not in sources[6])
    add('source','Spherical angular return resolves to the named section',
        all(z in sources[6] for z in [r'\label{sec:laplace-to-legendre}',
                                      r'\S\ref{sec:laplace-to-legendre}']))
    add('source','Integer Y defined by limit',r'\label{eq:Y-integer-limit}' in sources[5])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'review'/'pass2_checks.json')
    args=parser.parse_args()
    mathematical();structural()
    out={'scope':__doc__.strip(),'source_sha256':{p.name:digest(p) for p in sorted(ROOT.glob('mp_ch*.tex'))},
         'mathematical_checks':sum(r['type']=='mathematical' for r in RESULTS),
         'source_checks':sum(r['type']=='source' for r in RESULTS),
         'passed':sum(r['passed'] for r in RESULTS),'total':len(RESULTS),'results':RESULTS}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(f"{out['passed']}/{out['total']} passed: {out['mathematical_checks']} mathematical, {out['source_checks']} source checks")
    failed=[r for r in RESULTS if not r['passed']]
    if failed:print(json.dumps(failed,indent=2));raise SystemExit(1)
if __name__=='__main__':main()
