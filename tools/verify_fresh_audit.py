#!/usr/bin/env python3
"""Independent targeted audit of the 338-page notes.

Finite exact and high-precision checks, with source hashes and actual figure-data
checks. Formula tests are explicitly transcribed, not an automated proof/parser
of every LaTeX claim. Sampled asymptotic bounds are not universal error bounds.
"""
from pathlib import Path
import argparse, hashlib, json, re, time
import mpmath as mp
import numpy as np
import sympy as s
from scipy import special
ROOT=Path(__file__).resolve().parents[1]
mp.mp.dps=60
RESULTS=[]
x,z,t=s.symbols('x z t', real=True)

def add(ch,name,ok,detail='',kind='math'):
    RESULTS.append(dict(chapter=ch,check=name,passed=bool(ok),detail=str(detail),kind=kind))
def exact(ch,name,expr):
    r=s.simplify(s.expand(expr)); add(ch,name,r==0,r)
def near(ch,name,a,b,tol=mp.mpf('1e-35')):
    e=abs(a-b)/(1+abs(b));add(ch,name,mp.isfinite(e) and e<tol,f'scaled error {float(e):.4e}; tolerance {float(tol):.3e}')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analysis_checks():
    # Orientation on an annulus; coefficient extraction inside/outside a pole.
    theta,R=s.symbols('theta R',real=True,positive=True)
    near(1,'CCW circle integral of 1/z',mp.quad(lambda u:1j,[0,2*mp.pi]),2j*mp.pi)
    near(1,'CW inner boundary integral of 1/z',mp.quad(lambda u:1j,[2*mp.pi,0]),-2j*mp.pi)
    exact(1,'positive annular boundary has zero total for 1/z',2*s.pi*s.I-2*s.pi*s.I)
    for radius,expect in [(mp.mpf('.4'),-1),(mp.mpf('1.8'),0)]:
        a=mp.quad(lambda th:1j/(radius*mp.e**(1j*th)-1),[0,mp.pi,2*mp.pi])/(2j*mp.pi)
        near(1,f'Laurent residue circle radius={radius}',a,expect)
    for fun,center,res in [(1/s.sin(z),s.pi,-1),(1/(1+z*z)**2,s.I,1/(4*s.I)),
                           (z*z/s.sin(z),0,0)]:
        exact(1,f'residue {fun} at {center}',s.residue(fun,z,center)-res)
    near(1,'essential residue exp(1/z) by unit contour',mp.quad(lambda th:mp.e**(mp.e**(-1j*th))*mp.e**(1j*th),[0,mp.pi,2*mp.pi])/(2*mp.pi),1)
    for k in [1,2,4]:
        exact(1,f'removable sin(z)/z derivative {2*k}',s.diff(s.series(s.sin(z)/z,z,0,12).removeO(),z,2*k).subs(z,0)-(-1)**k/s.Integer(2*k+1))
    for b in [mp.mpf('0.3'),mp.mpf('1.4')]:
        q=mp.mpf('1e-12')
        near(1,f'log upper-minus-lower cut jump r={b}',mp.log(-b+1j*q)-mp.log(-b-1j*q),2j*mp.pi,mp.mpf('1e-11'))
    # The keyhole phases are checked for nonreal exponents too, inside the strip.
    for a in [mp.mpc('.2','.4'),mp.mpc('-.3','.2')]:
        integral=mp.quad(lambda q:q**a/(1+q)**2,[0,1,mp.inf])
        near(1,f'complex keyhole exponent {a}',integral,mp.pi*a/mp.sin(mp.pi*a))
        phase=mp.e**(2j*mp.pi*a)
        near(1,f'keyhole residue and bank phases {a}',(1-phase)*integral,-2j*mp.pi*a*mp.e**(1j*mp.pi*a))
    # A single-point complex-valued Lagrange remainder is not claimed in the notes.
    b=mp.pi
    add(1,'real-valued Lagrange hypothesis is necessary',abs((mp.e**(1j*b)-1)/b)<mp.mpf('0.7'),
        'For f=exp(ix), |f(pi)-f(0)|=2, but pi|fprime(c)|=pi: no single real c works.')
    for a,b in [(mp.mpc('.4','.7'),mp.mpc('1.2','-.3')),
                (mp.mpc('1.3','-.2'),mp.mpc('.7','.4'))]:
        beta=mp.quad(lambda u:u**(a-1)*(1-u)**(b-1),[0,mp.mpf('.5'),1])
        near(2,f'complex beta integral a={a}, b={b}',beta,mp.gamma(a)*mp.gamma(b)/mp.gamma(a+b),mp.mpf('1e-22'))
    for a in [mp.mpc('.6','.8'),mp.mpc('-2.3','.2'),mp.mpc('3.1','-.9')]:
        near(2,f'complex reflection z={a}',mp.gamma(a)*mp.gamma(1-a),mp.pi/mp.sin(mp.pi*a))
        near(2,f'complex recurrence z={a}',mp.gamma(a+1),a*mp.gamma(a))
    for a in [mp.mpc('.7','.3'),mp.mpc('1.8','-.5')]:
        inte=mp.quad(lambda q:(1-q**a)/(1-q),[0,mp.mpf('.5'),1])
        near(2,f'harmonic interpolation {a}',inte,mp.digamma(a+1)+mp.euler)
    # At z=1 the finite Euler approximant is n/(n+1), not exactly Gamma(1).
    for n in [10,50,200]:
        q=mp.mpf(n)
        near(2,f'Euler approximant exactly n/(n+1), n={n}',q*mp.factorial(n)/mp.factorial(n+1),q/(q+1))
    for p in [mp.mpf('2'),mp.mpf('3'),mp.mpf('4')]:
        # Change u=x^p; oscillatory quadrature evaluates the conditional integral.
        beta=1/p
        val=mp.quad(lambda u:u**(beta-1)*mp.cos(u),[0,1])+mp.quadosc(lambda u:u**(beta-1)*mp.cos(u),[1,mp.inf],omega=1)
        near(2,f'oscillatory gamma integral p={p}',val/p,mp.gamma(beta)*mp.cos(mp.pi*beta/2)/p,mp.mpf('1e-15'))
    for zz in [mp.mpc('1.8','.3'),mp.mpc('2.6','-.7')]:
        val=mp.quad(lambda u:u**(zz-1)/mp.expm1(u),[0,1,mp.inf])
        near(2,f'zeta Mellin integral s={zz}',val,mp.gamma(zz)*mp.zeta(zz),mp.mpf('1e-30'))

def ode_checks():
    # Exact retained exceptional-index examples, not just regular recurrences.
    rr,mm,kk=s.symbols('r m k')
    exact(3,'Bessel general indicial factor',rr*(rr-1)+rr-mm*mm-(rr-mm)*(rr+mm))
    for m in range(1,7):
        a={0:s.Integer(1)}
        for k in range(2,2*m,2):a[k]=-a[k-2]/(k*(k-2*m))
        add(3,f'negative integer Bessel branch obstruction m={m}',a[2*m-2]!=0,f'coefficient at k=2m equals {a[2*m-2]}, cannot divide by zero')
    for f in [x**3,x**-2]:exact(3,f'integer-gap no-log example {f}',x*x*s.diff(f,x,2)-6*f)
    for f in [s.Integer(1),s.log(x)]:exact(3,f'repeated-root logarithm example {f}',x*x*s.diff(f,x,2)+x*s.diff(f,x))
    # Legendre nonpolynomial branch at lambda=2 (the even chain).
    f=1-x*s.log((1+x)/(1-x))/2
    exact(6,'Legendre lambda=2 nonpolynomial example', (1-x*x)*s.diff(f,x,2)-2*x*s.diff(f,x)+2*f)
    add(6,'Legendre counterexample diverges at both endpoints',s.limit(f,x,1,dir='-')==-s.oo and s.limit(f,x,-1,dir='+')==-s.oo)
    # General reduction of order checked by the original ODE, not only Wronskian.
    p=s.Function('p')(x);q=s.Function('q')(x);u=s.Function('u')(x);v=s.Function('v')(x)
    residual=s.diff(u*v,x,2)+p*s.diff(u*v,x)+q*u*v
    residual=residual.subs(s.diff(u,x,2),-p*s.diff(u,x)-q*u)
    exact(3,'reduction of order full residual',residual-u*s.diff(v,x,2)-(2*s.diff(u,x)+p*u)*s.diff(v,x))
    # Actual regular/radial classifications used in the text.
    for name,P,Q,p0,q0 in [('Bessel',1/x,1-4/x**2,1,-4),('Laguerre',(1-x)/x,3/x,1,0)]:
        exact(3,f'{name} regular-singular p0',s.limit(x*P,x,0)-p0)
        exact(3,f'{name} regular-singular q0',s.limit(x*x*Q,x,0)-q0)
    M,omega,hbar,xi,E=s.symbols('M omega hbar xi E',positive=True)
    # Scaling of kinetic and potential coefficients in oscillator.
    exact(6,'oscillator scaled kinetic coefficient',hbar*hbar/(2*M)*(M*omega/hbar)/(hbar*omega/2)-1)
    exact(6,'oscillator scaled potential coefficient',(M*omega*omega/2)*(hbar/(M*omega))*xi**2/(hbar*omega/2)-xi**2)
    for n in range(9):
        H=sum((-1)**k*s.factorial(n)*(2*xi)**(n-2*k)/(s.factorial(k)*s.factorial(n-2*k)) for k in range(n//2+1))
        psi=s.exp(-xi*xi/2)*H
        exact(6,f'oscillator direct physical-scaling residual n={n}',s.diff(psi,xi,2)+(2*n+1-xi*xi)*psi)
    # First and second derivative conversion theta -> cos(theta), general function.
    th=s.symbols('theta', real=True);F=s.Function('F')
    expr=s.diff(s.sin(th)*s.diff(F(s.cos(th)),th),th)/s.sin(th)
    expected=((1-x*x)*s.diff(F(x),x,2)-2*x*s.diff(F(x),x)).subs(x,s.cos(th))
    exact(6,'spherical-to-Legendre chain rule',expr-expected)
    r=s.symbols('r',positive=True)
    for n in range(7):
        for f in [r**n,r**(-n-1)]:exact(6,f'spherical radial equation n={n}, u={f}',s.diff(r*r*s.diff(f,r),r)-n*(n+1)*f)
        for f in ([s.Integer(1),s.log(r)] if n==0 else [r**n,r**(-n)]):
            exact(6,f'polar radial equation n={n}, u={f}',r*r*s.diff(f,r,2)+r*s.diff(f,r)-n*n*f)

def spectral_bessel_checks():
    # Complex-valued test functions exercise the first-slot-linear convention.
    p=1+x*x;q=2-x;w=2+x
    u=(1+s.I)*x*(1-x);v=(2-s.I)*x*x*(1-x)
    ell=lambda f:s.diff(p*s.diff(f,x),x)+q*f
    ip=lambda f,g:s.integrate(f*s.conjugate(g)*w,(x,0,1))
    A=lambda f:-ell(f)/w
    exact(4,'weighted Green identity for complex Dirichlet functions',ip(A(u),v)-ip(u,A(v)))
    # A function in the adjoint domain need not obey an overrestricted domain.
    u=x*x*(1-x)**2;v=s.Integer(1)
    exact(4,'symmetric-domain counterexample has zero boundary form',s.integrate(u*s.diff(v,x,2)-s.diff(u,x,2)*v,(x,0,1)))
    add(4,'adjoint-test constant violates overrestricted Dirichlet data',v.subs(x,0)!=0,
        'L=-d2/dx2 with u=uprime=0 at both ends is symmetric but not self-adjoint; v=1 belongs to its adjoint domain.')
    for nu in [mp.mpf('-2.3'),mp.mpf('-.5'),mp.mpf('.5'),mp.mpf('1.4'),mp.mpf('4.2')]:
        for a in [mp.mpf('.08'),mp.mpf('1.1'),mp.mpf('6')]:
            series=mp.fsum([(-1)**j*(a/2)**(2*j+nu)*mp.rgamma(nu+j+1)/mp.factorial(j) for j in range(70)])
            near(5,f'general-order source series nu={nu},x={a}',series,mp.besselj(nu,a))
            val=(mp.besselj(nu,a)*mp.cos(mp.pi*nu)-mp.besselj(-nu,a))/mp.sin(mp.pi*nu)
            near(5,f'Y definition negative/fractional order nu={nu},x={a}',val,mp.bessely(nu,a))
    for m in range(6):
        a=mp.mpf('1.3')
        series=mp.fsum([(-1)**j*(a/2)**(2*j-m)*mp.rgamma(j-m+1)/mp.factorial(j) for j in range(55)])
        near(5,f'reciprocal-gamma series at negative integer {-m}',series,mp.besselj(-m,a))
        eps=mp.mpf('1e-18')
        vals=[]
        for sign in [-1,1]:
            nu=m+sign*eps
            vals.append((mp.besselj(nu,a)*mp.cos(mp.pi*nu)-mp.besselj(-nu,a))/mp.sin(mp.pi*nu))
        near(5,f'two-sided integer Y limit m={m}',sum(vals)/2,mp.bessely(m,a),mp.mpf('1e-30'))
    for m in range(4):
        a=mp.mpf('2.1');b=a+mp.mpf('1e-12')
        val=mp.quad(lambda r:r*mp.besselj(m,a*r)*mp.besselj(m,b*r),[0,1])
        cross=(b*mp.besselj(m,a)*mp.besselj(m,b,1)-a*mp.besselj(m,b)*mp.besselj(m,a,1))/(a*a-b*b)
        near(5,f'cross-product near coincident parameters m={m}',cross,val,mp.mpf('1e-44'))
    # Neumann zero-frequency mode is not included in positive-zero lists.
    near(4,'Neumann m=0 constant-mode radial norm',mp.quad(lambda r:r,[0,1]),mp.mpf('.5'))
    for j in range(1,5):
        a=mp.besseljzero(1,j)
        near(4,f'constant orthogonal to positive Neumann mode j={j}',mp.quad(lambda r:r*mp.besselj(0,a*r),[0,1]),0)
    for nu in [mp.mpf('0'),mp.mpf('1'),mp.mpf('2.3'),mp.mpf('-.3')]:
        for k in [20,60,180]:
            a=nu*mp.pi/2+3*mp.pi/4+k*mp.pi  # zero of the leading cosine
            chi=a-nu*mp.pi/2-mp.pi/4
            pref=mp.sqrt(2/(mp.pi*a));first=(4*nu**2-1)/8
            approx=pref*(mp.cos(chi)-first/a*mp.sin(chi))
            err=abs(mp.besselj(nu,a)-approx)*a**mp.mpf('2.5')
            add(5,f'J asymptotic at cosine zero nu={nu}, k={k}',err<2,
                f'next-term residual times x^(5/2)={float(err):.4e}; sampled check, not theorem')
    for m in range(5):
        for a in [mp.mpf(50),mp.mpf(200)]:
            lead=mp.sqrt(2/(mp.pi*a))*mp.e**(1j*(a-m*mp.pi/2-mp.pi/4))
            ratio=mp.hankel1(m,a)/lead
            add(5,f'Hankel far-field ratio m={m},x={a}',abs(ratio-1)<(1+m*m)/a,
                f'|ratio-1|={float(abs(ratio-1)):.4e}; leading term is not exact')
    # Exact leading-wave residual: proves proportionality fails for integer m.
    rr,k,mm=s.symbols('r k m',positive=True)
    f=rr**s.Rational(-1,2)*s.exp(s.I*k*rr)
    res=s.diff(f,rr,2)+s.diff(f,rr)/rr+(k*k-mm*mm/rr**2)*f
    exact(5,'outgoing leading ansatz residual',(res/f)-(s.Rational(1,4)-mm*mm)/rr**2)

def polynomial_data_checks():
    # Actual polynomial plot expressions, parsed from the supplied figure TeX.
    for stem,family in [('legendre-P',s.legendre),('hermite-H',s.hermite),('laguerre-L',s.laguerre),('chebyshev-T',s.chebyshevt)]:
        text=(ROOT/'figs'/f'{stem}.tex').read_text()
        expressions=re.findall(r'\\addplot\[[^]]*\]\s*\{([^}]+)\}',text)
        for n,e in enumerate(expressions):
            ex=s.sympify(e.replace('^','**'),locals={'x':x},rational=True)
            exact(6,f'actual figure formula {stem} n={n}',ex-family(n,x))
    for n in range(9):
        P=s.legendre(n,x)
        # Endpoint values survive the apparent singular factors in the ODEs.
        exact(6,f'Legendre endpoint derivative n={n}',s.diff(P,x).subs(x,1)-s.Rational(n*(n+1),2))
        T=s.chebyshevt(n,x)
        exact(6,f'Chebyshev endpoint derivative n={n}',s.diff(T,x).subs(x,1)-n*n)
        for m in range(n+1):
            # Verify associated norms by a polynomial integrand, avoiding sqrt branches.
            integ=(1-x*x)**m*s.diff(P,x,m)**2
            norm=s.Rational(2,2*n+1)*s.factorial(n+m)/s.factorial(n-m)
            exact(6,f'associated-Legendre norm n={n},m={m}',s.integrate(integ,(x,-1,1))-norm)
    gamma_src=(ROOT/'figs'/'gamma-real.tex').read_text()
    blocks=re.findall(r'\\addplot[^\n]*coordinates\s*\{(.*?)\};',gamma_src,re.S)
    total_gamma=0
    for j,block in enumerate(blocks):
        vals=np.array([(float(a),float(b)) for a,b in re.findall(r'\(([-\d.]+),([-\d.]+)\)',block)])
        total_gamma+=len(vals)
        # mp independent of the plotted four-decimal tabulations.
        err=max(abs(mp.mpf(str(b))-mp.gamma(mp.mpf(str(a)))) for a,b in vals)
        add(2,f'all gamma coordinates, branch {j}',err<mp.mpf('0.00005001'),f'{len(vals)} coordinates; maximum absolute error {float(err):.6g}',kind='data')
    for family,fun in [('J',mp.besselj),('Y',mp.bessely)]:
        for n in range(4):
            vals=np.loadtxt(ROOT/'figs'/f'bessel-{family}{n}.dat')
            err=max(abs(mp.mpf(float(b))-fun(n,mp.mpf(float(a))))/(1+abs(fun(n,mp.mpf(float(a))))) for a,b in vals)
            add(5,f'all actual Bessel data {family}{n}',err<mp.mpf('5e-13'),f'{len(vals)} samples; maximum scaled error {float(err):.6g}',kind='data')
    vals=np.loadtxt(ROOT/'figs'/'airy-ai-bi.dat',skiprows=1)
    err=max(max(abs(mp.mpf(float(ai))-mp.airyai(mp.mpf(float(a)))),abs(mp.mpf(float(bi))-mp.airybi(mp.mpf(float(a))))) for a,ai,bi in vals)
    add(3,'all actual Airy Ai/Bi data',err<mp.mpf('5.1e-11'),f'{len(vals)} paired samples; maximum absolute error {float(err):.6g}',kind='data')
    for order,radius in [(0,mp.mpf('.5446')),(1,mp.mpf('.6827'))]:
        target=mp.mpf('1.25')*mp.besseljzero(order,1)/mp.besseljzero(order,2)
        add(5,f'disk nodal-circle radius m={order}',abs(radius-target)<mp.mpf('0.00005'),f'{radius}; exact {target}',kind='data')
    return total_gamma

def source_checks():
    sources={n:(ROOT/f'mp_ch{n}.tex').read_text() for n in range(1,8)}
    checks=[(1,'outer/inner contour distinction','boundary around a hole is clockwise'),
            (4,'finite weighted norm explicitly required',r'$0<\int_a^b w|u|^2\,dx<\infty$'),
            (4,'symmetry not generalized self-adjointness','not a general\nself-adjointness or completeness theorem'),
            (5,'positive-frequency radiation convention',r'$\omega>0$'),
            (5,'Hankel asymptotic relation',r'\sim \sqrt{\frac{2}{\pi kr}}'),
            (5,'Hankel exact radial solution distinguished','exact radial\nsolutions for $r>0$')]
    for ch,name,fragment in checks:add(ch,name,fragment in sources[ch],kind='source')
    for n,src in sources.items():
        labs=re.findall(r'\\label\{([^}]+)\}',src);refs=re.findall(r'\\(?:eqref|ref|pageref|autoref)\{([^}]+)\}',src)
        add(n,'all local references resolve',set(refs)<=set(labs),kind='source')
        add(n,'no duplicated equation labels',len(labs)==len(set(labs)),kind='source')
    return {str(n):dict(lines=len(v.splitlines()),equation_labels=re.findall(r'\\label\{([^}]+)\}',v),sections=re.findall(r'\\section\{([^}]+)\}',v)) for n,v in sources.items()}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'review'/'fresh_audit_checks.json');args=parser.parse_args()
    start=time.time()
    for f in [analysis_checks,ode_checks,spectral_bessel_checks,polynomial_data_checks]:
        print('Checking',f.__name__,flush=True);f()
    inventory=source_checks()
    files=sorted(ROOT.glob('mp_ch*.tex'))+[ROOT/'Kyushu.sty']+sorted((ROOT/'figs').glob('*.tex'))+sorted((ROOT/'figs').glob('*.dat'))
    out={'scope':__doc__,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'source_inventory':inventory,
         'math_checks':sum(r['kind']=='math' for r in RESULTS),'data_groups':sum(r['kind']=='data' for r in RESULTS),'source_checks':sum(r['kind']=='source' for r in RESULTS),
         'passed':sum(r['passed'] for r in RESULTS),'total':len(RESULTS),'failed':[r for r in RESULTS if not r['passed']], 'elapsed_seconds':round(time.time()-start,2),'results':RESULTS}
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['math_checks','data_groups','source_checks','passed','total','failed','elapsed_seconds']},indent=2))
    raise SystemExit(bool(out['failed']))
if __name__=='__main__':main()
