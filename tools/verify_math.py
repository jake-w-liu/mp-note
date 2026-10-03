#!/usr/bin/env python3
"""Reproducible checks of formulas retained in chapters 1--7.

These checks are independent symbolic/numerical evaluations of transcribed
formulas, not a LaTeX parser or a proof of every theorem. Source hashes and
retained equation-label inventory bind the report to the version reviewed.
Requires Python 3.10+, sympy, mpmath, numpy, scipy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
import mpmath as mp
import numpy as np
import sympy as sp
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
mp.mp.dps = 45
results: list[dict] = []
x, z, t = sp.symbols('x z t', real=True)

def check(ch: int, name: str, ok: bool, detail: str = '') -> None:
    results.append({'chapter': ch, 'check': name, 'passed': bool(ok), 'detail': detail})

def exact(ch: int, name: str, expr) -> None:
    residual = sp.simplify(sp.expand(expr))
    check(ch, name, residual == 0, str(residual))

def near(ch: int, name: str, a, b, tol: float = 1e-12) -> None:
    err = float(abs(a-b)/(1+abs(b)))
    check(ch, name, math.isfinite(err) and err <= tol,
          f'scaled absolute error={err:.4e}; tolerance={tol:.2e}')

def moment_integral(poly, family: str):
    p = sp.Poly(sp.expand(poly), x)
    value = 0
    for (k,), c in p.terms():
        if family == 'P': moment = 0 if k % 2 else sp.Rational(2, k+1)
        elif family == 'H': moment = 0 if k % 2 else sp.gamma(sp.Rational(k+1, 2))
        elif family == 'L': moment = sp.factorial(k)
        else: raise ValueError(family)
        value += c*moment
    return value

def chapter1() -> None:
    for f in [sp.exp(x), sp.sin(x), sp.cos(x), 1/(1-x), sp.log(1+x)]:
        for a in [0, sp.Rational(1, 4)]:
            coeff = sum(sp.diff(f,x,k).subs(x,a)/sp.factorial(k)*(x-a)**k for k in range(6))
            exact(1, f'Taylor derivatives f={f}, a={a}',
                  coeff-sp.series(f,x,a,6).removeO())
    u,v = sp.symbols('u v', real=True)
    for f in [(u+sp.I*v)**2, sp.exp(u+sp.I*v)]:
        re,im=sp.expand_complex(f).as_real_imag()
        exact(1,'Cauchy--Riemann first '+str(f),sp.diff(re,u)-sp.diff(im,v))
        exact(1,'Cauchy--Riemann second '+str(f),sp.diff(re,v)+sp.diff(im,u))
    exact(1,'partial fraction identity',1/(z**2*(z-1))-(1/(z-1)-1/z-1/z**2))
    exact(1,'worked Cauchy integral residues',
          sp.residue(sp.exp(z)/(z**2*(z-1)),z,0)+sp.residue(sp.exp(z)/(z**2*(z-1)),z,1)-(sp.E-2))
    exact(1,'residue sin(z)/z^2',sp.residue(sp.sin(z)/z**2,z,0)-1)
    exact(1,'removable z^2/sin(z)',sp.residue(z**2/sp.sin(z),z,0))
    exact(1,'semicircle cosine residue',2*sp.pi*sp.I*sp.residue(sp.exp(sp.I*z)/(1+z**2),z,sp.I)-sp.pi/sp.E)
    roots=[(1+sp.I)/sp.sqrt(2),(-1+sp.I)/sp.sqrt(2)]
    exact(1,'rational quartic integral',2*sp.pi*sp.I*sum(1/(4*r**3) for r in roots)-sp.pi/sp.sqrt(2))
    for a in ['-0.6','-0.25','0','0.25','0.6']:
        aa=mp.mpf(a)
        rhs=mp.pi*aa/mp.sin(mp.pi*aa) if aa else mp.mpf(1)
        near(1,'keyhole integral a='+a,mp.quad(lambda q:q**aa/(1+q)**2,[0,1,mp.inf]),rhs,1e-15)
    for a in [mp.mpf('0.3'),mp.mpf('0.5'),mp.mpf('0.7')]:
        near(1,'keyhole simple denominator '+str(a),
             mp.quad(lambda q:q**(-a)/(1+q),[0,1,mp.inf]),mp.pi/mp.sin(mp.pi*a),1e-12)
    for theta in [-2.4,-0.5,0.4,2.6]:
        alpha=mp.e**(1j*(mp.pi-theta)/2)
        near(1,'saddle descent direction '+str(theta),mp.e**(1j*theta)*alpha**2,-1)
    for sign in [-1,1]:
        eps=mp.mpf('0.001')
        a=mp.mpf('1.3')
        # Exact damped Gaussian tends to the Fresnel phase; finite damping is not the limit.
        exact_gauss=mp.sqrt(mp.pi/(eps-sign*1j*a))
        target=mp.e**(sign*1j*mp.pi/4)*mp.sqrt(mp.pi/a)
        near(1,'damped Fresnel phase '+str(sign),exact_gauss,target,4e-4)
    q=sp.symbols('q', positive=True)
    f=sp.log(1+z)-q*sp.log(z)
    saddle=q/(1-q)
    exact(1,'binomial saddle stationary',sp.diff(f,z).subs(z,saddle))
    exact(1,'binomial saddle curvature',sp.diff(f,z,2).subs(z,saddle)-(1-q)**3/q)
    for n,m in [(200,80),(1000,350),(2000,1000)]:
        q=mp.mpf(m)/n
        est=mp.e**(-n*(q*mp.log(q)+(1-q)*mp.log(1-q)))/mp.sqrt(2*mp.pi*n*q*(1-q))
        near(1,f'binomial leading approximation n={n},m={m}',est,mp.binomial(n,m),0.002)

def chapter2() -> None:
    for a in [mp.mpf('0.3'),mp.mpf('0.5'),mp.mpf('1.4'),mp.mpf('2.7')]:
        near(2,'Gamma recurrence '+str(a),mp.gamma(a+1),a*mp.gamma(a),1e-35)
        near(2,'Euler gamma integral '+str(a),mp.quad(lambda q:q**(a-1)*mp.exp(-q),[0,1,mp.inf]),mp.gamma(a),1e-12)
    for a in [mp.mpf('0.2'),mp.mpf('0.5'),mp.mpf('0.8'),mp.mpf('1.3')]:
        near(2,'Gamma reflection '+str(a),mp.gamma(a)*mp.gamma(1-a),mp.pi/mp.sin(mp.pi*a),1e-35)
    for n in range(7):
        exact(2,f'Gamma residue at -{n}',sp.residue(sp.gamma(z),z,-n)-sp.Rational((-1)**n,sp.factorial(n)))
        exact(2,f'half-integer Gamma n={n}',sp.gamma(sp.Rational(1,2)+n)-sp.factorial(2*n)*sp.sqrt(sp.pi)/(4**n*sp.factorial(n)))
    near(2,'Gaussian integral',mp.quad(lambda q:mp.exp(-q*q),[-mp.inf,0,mp.inf]),mp.sqrt(mp.pi),1e-35)
    for a,b in [('0.5','0.5'),('0.75','1.25'),('1.3','2.2')]:
        a,b=mp.mpf(a),mp.mpf(b)
        near(2,f'beta gamma {a},{b}',mp.quad(lambda u:u**(a-1)*(1-u)**(b-1),[0,mp.mpf('.5'),1]),mp.gamma(a)*mp.gamma(b)/mp.gamma(a+b),1e-18)
    near(2,'quartic beta example',mp.quad(lambda u:1/mp.sqrt(1-u**4),[0,mp.mpf('.8'),1]),mp.gamma(mp.mpf('.25'))**2/(4*mp.sqrt(2*mp.pi)),1e-18)
    for n in [0,1,2,3,mp.mpf('0.5'),mp.mpf('2.4')]:
        near(2,'sine power beta n='+str(n),mp.quad(lambda q:mp.sin(q)**n,[0,mp.pi/2,mp.pi]),mp.sqrt(mp.pi)*mp.gamma((n+1)/2)/mp.gamma((n+2)/2),1e-15)
    near(2,'Basel value',mp.zeta(2),mp.pi**2/6,1e-35)
    for n in [30,100,500]:
        est=mp.sqrt(2*mp.pi*n)*(mp.mpf(n)/mp.e)**n
        near(2,'Stirling factorial n='+str(n),est,mp.factorial(n),0.003)
    for n in [1,5,20,100]:
        h=sum(mp.mpf(1)/k for k in range(1,n+1))
        check(2,'harmonic integral bounds n='+str(n),mp.log(n+1)<=h<=1+mp.log(n))

def chapter3() -> None:
    for n in range(2,16):
        for f in [sp.sin(x),sp.cos(x)]:
            a=sp.diff(f,x,n).subs(x,0)/sp.factorial(n)
            old=sp.diff(f,x,n-2).subs(x,0)/sp.factorial(n-2)
            exact(3,f'oscillator coefficient n={n},f={f}',n*(n-1)*a+old)
    for m in range(5):
        for j in range(1,9):
            a=(-1)**j/(sp.Integer(2)**(2*j+m)*sp.factorial(j)*sp.factorial(m+j))
            prev=(-1)**(j-1)/(sp.Integer(2)**(2*(j-1)+m)*sp.factorial(j-1)*sp.factorial(m+j-1))
            exact(3,f'Bessel Frobenius m={m},j={j}',(2*j)*(2*j+2*m)*a+prev)
    coeff=[]
    for seed in [(1,0),(0,1)]:
        a=[sp.Integer(seed[0]),sp.Integer(seed[1]),sp.Integer(0)]
        for k in range(1,25):a.append(a[k-1]/((k+2)*(k+1)))
        p=sum(aa*x**k for k,aa in enumerate(a))
        for k in range(24):exact(3,f'Airy recurrence seed={seed},k={k}',sp.expand(sp.diff(p,x,2)-x*p).coeff(x,k))
        coeff.append(p)
    A0=1/(mp.power(3,mp.mpf(2)/3)*mp.gamma(mp.mpf(2)/3))
    A1=-1/(mp.power(3,mp.mpf(1)/3)*mp.gamma(mp.mpf(1)/3))
    B0=mp.sqrt(3)*A0;B1=-mp.sqrt(3)*A1
    for name,a,b in [('Ai(0)',A0,mp.airyai(0)),("Ai'(0)",A1,mp.airyai(0,1)),('Bi(0)',B0,mp.airybi(0)),("Bi'(0)",B1,mp.airybi(0,1))]:near(3,name,a,b,1e-35)
    near(3,'Airy Wronskian',A0*B1-A1*B0,1/mp.pi,1e-35)
    for p in [x**3,x**(-2)]:exact(3,'Euler root gap '+str(p),x*x*sp.diff(p,x,2)-6*p)
    for p in [sp.Integer(1),sp.log(x)]:exact(3,'repeated Euler root '+str(p),x*x*sp.diff(p,x,2)+x*sp.diff(p,x))
    exact(3,'oscillator reduction of order',sp.cos(x)*sp.tan(x)-sp.sin(x))
    exact(3,'oscillator Wronskian',sp.cos(x)**2+sp.sin(x)**2-1)
    P=sp.Function('P')(x);v=sp.Function('v')(x);I=sp.Function('I')(x)
    y=v*I;w=v*sp.diff(y,x)-sp.diff(v,x)*y
    exact(3,'reduction-order product rule',w-v**2*sp.diff(I,x))
    for a in [mp.mpf('.4'),mp.mpf('1.3'),mp.mpf('3')]:
        near(3,'J half order '+str(a),mp.besselj(mp.mpf('.5'),a),mp.sqrt(2/(mp.pi*a))*mp.sin(a),1e-35)
        near(3,'J negative half order '+str(a),mp.besselj(mp.mpf('-.5'),a),mp.sqrt(2/(mp.pi*a))*mp.cos(a),1e-35)

def chapter4() -> None:
    exact(4,'basis conversion',1+2*x+3*x*x-(3+sp.Rational(5,2)*(2*x-1)+sp.Rational(1,2)*(6*x*x-6*x+1)))
    a=2+3*sp.I;b=1-2*sp.I;u=sp.Matrix([1,sp.I]);v=sp.Matrix([2+sp.I,3])
    ip=lambda p,q:(p.T*q.conjugate())[0]
    exact(4,'inner product first-slot linearity',ip(a*u,v)-a*ip(u,v))
    exact(4,'inner product second-slot conjugate',ip(u,b*v)-sp.conjugate(b)*ip(u,v))
    exact(4,'inner product positive',ip(sp.I*u,sp.I*u)-ip(u,u))
    for n in range(1,6):
        for m in range(1,6):
            value=sp.integrate(sp.sin(n*sp.pi*x)*sp.sin(m*sp.pi*x),(x,0,1))
            exact(4,f'sine orthogonality n={n},m={m}',value-(sp.Rational(1,2) if n==m else 0))
    u,v,p=sp.Function('u')(x),sp.Function('v')(x),sp.Function('p')(x)
    # v may represent the conjugate test function; identity is algebraic.
    exact(4,'Green identity derivative',u*sp.diff(p*sp.diff(v,x),x)-sp.diff(p*sp.diff(u,x),x)*v-sp.diff(p*(u*sp.diff(v,x)-sp.diff(u,x)*v),x))
    p0,p1,p2=[sp.Function(q)(x) for q in ['p0','p1','p2']]
    adj=sp.diff(p0*v,x,2)-sp.diff(p1*v,x)+p2*v
    expected=p0*sp.diff(v,x,2)+(2*sp.diff(p0,x)-p1)*sp.diff(v,x)+(sp.diff(p0,x,2)-sp.diff(p1,x)+p2)*v
    exact(4,'second order formal adjoint',adj-expected)
    exact(4,'Hermite integrating factor',sp.diff(sp.exp(-x*x),x)+2*x*sp.exp(-x*x))
    v0=sp.Integer(1);v1=x;v2=x*x-sp.Rational(1,3)
    for a,b in [(v0,v1),(v0,v2),(v1,v2)]:exact(4,'Gram Schmidt '+str(a)+','+str(b),sp.integrate(a*b,(x,-1,1)))
    exact(4,'Gram Schmidt P2',v2/v2.subs(x,1)-sp.legendre(2,x))

def chapter5() -> None:
    for n in range(5):
        for a in [mp.mpf('.3'),mp.mpf('1.2'),mp.mpf('4')]:
            J=lambda nu:mp.besselj(nu,a)
            near(5,f'negative integer J n={n},x={a}',J(-n),(-1)**n*J(n),1e-35)
            near(5,f'J order recurrence n={n},x={a}',J(n-1)+J(n+1),2*n/a*J(n),1e-35)
            near(5,f'J derivative n={n},x={a}',mp.diff(lambda q:mp.besselj(n,q),a),(J(n-1)-J(n+1))/2,1e-35)
            near(5,f'J integral n={n},x={a}',mp.quad(lambda q:mp.cos(n*q-a*mp.sin(q)),[0,mp.pi])/mp.pi,J(n),1e-35)
            near(5,f'JY Wronskian n={n},x={a}',J(n)*mp.diff(lambda q:mp.bessely(n,q),a)-mp.diff(lambda q:mp.besselj(n,q),a)*mp.bessely(n,a),2/(mp.pi*a),1e-35)
    for a in [mp.mpf('.7'),mp.mpf('2')]:
        for n in [0,1,2,3]:
            D=lambda nu:mp.diff(lambda q:mp.besselj(q,a),nu)
            near(5,f'integer-order lHopital n={n},x={a}',(D(n)+(-1)**n*D(-n))/mp.pi,mp.bessely(n,a),1e-30)
    for nu in [mp.mpf('.3'),mp.mpf('1.7')]:
        for a in [mp.mpf('.5'),mp.mpf('2')]:
            near(5,f'noninteger Neumann nu={nu},x={a}',(mp.besselj(nu,a)*mp.cos(mp.pi*nu)-mp.besselj(-nu,a))/mp.sin(mp.pi*nu),mp.bessely(nu,a),1e-35)
    for a,b in [(mp.mpf('.6'),mp.mpf('2.1')),(mp.mpf('2'),mp.mpf('3'))]:
        for m in range(4):
            val=mp.quad(lambda q:q*mp.besselj(m,a*q)*mp.besselj(m,b*q),[0,1])
            rhs=(b*mp.besselj(m,a)*mp.besselj(m,b,1)-a*mp.besselj(m,b)*mp.besselj(m,a,1))/(a*a-b*b)
            near(5,f'Bessel cross integral m={m},a={a},b={b}',val,rhs,1e-33)
    for m in range(4):
        zeros=[mp.besseljzero(m,k) for k in range(1,4)]
        for j,a in enumerate(zeros):
            near(5,f'Bessel simple zero m={m},j={j+1}',mp.besselj(m,a),0,1e-35)
            for k,b in enumerate(zeros):
                val=mp.quad(lambda q:q*mp.besselj(m,a*q)*mp.besselj(m,b*q),[0,1])
                rhs=mp.besselj(m+1,a)**2/2 if k==j else 0
                near(5,f'Fourier Bessel norm/orthogonality m={m},j={j+1},k={k+1}',val,rhs,1e-33)
            # Projection of its own mode must have coefficient one.
            norm=mp.quad(lambda q:q*mp.besselj(m,a*q)**2,[0,1])
            near(5,f'Fourier Bessel coefficient m={m},j={j+1}',2*norm/mp.besselj(m+1,a)**2,1,1e-33)
    for n in range(4):
        a=mp.mpf('0.0001')
        if n:
            leading=-mp.gamma(n)/mp.pi*(2/a)**n
            near(5,f'Y small argument n={n}',mp.bessely(n,a),leading,1e-6)
        else:near(5,'Y0 small argument constant',mp.bessely(0,a),2/mp.pi*(mp.log(a/2)+mp.euler),1e-7)
    for n in range(4):
        for a in [mp.mpf(200),mp.mpf(800)]:
            phase=a-n*mp.pi/2-mp.pi/4;amp=mp.sqrt(2/(mp.pi*a))
            near(5,f'J large argument leading n={n},x={a}',mp.besselj(n,a),amp*mp.cos(phase),0.003)
            near(5,f'Y large argument leading n={n},x={a}',mp.bessely(n,a),amp*mp.sin(phase),0.003)
    for family,fun in [('J',mp.besselj),('Y',mp.bessely)]:
        for n in range(4):
            table=np.loadtxt(ROOT/'figs'/f'bessel-{family}{n}.dat')
            # 21 evenly spaced samples include both endpoints and the near-zero Y region.
            err=max(float(abs(mp.mpf(float(table[i,1]))-fun(n,mp.mpf(float(table[i,0]))))/(1+abs(fun(n,mp.mpf(float(table[i,0])))))) for i in np.unique(np.linspace(0,len(table)-1,21,dtype=int)))
            check(5,f'figure table {family}{n}',err<5e-13,f'21-point mpmath cross-check; max scaled error={err:.4e}')

def chapter6() -> None:
    families={'P':sp.legendre,'H':sp.hermite,'L':sp.laguerre,'T':sp.chebyshevt}
    g={'P':(1-2*x*t+t*t)**sp.Rational(-1,2),'H':sp.exp(2*x*t-t*t),'L':sp.exp(-x*t/(1-t))/(1-t),'T':(1-x*t)/(1-2*x*t+t*t)}
    gs={k:sp.series(v,t,0,7).removeO().expand() for k,v in g.items()}
    for name,fun in families.items():
        for n in range(7):
            p=fun(n,x)
            if name=='P': ode=(1-x*x)*sp.diff(p,x,2)-2*x*sp.diff(p,x)+n*(n+1)*p
            elif name=='H': ode=sp.diff(p,x,2)-2*x*sp.diff(p,x)+2*n*p
            elif name=='L': ode=x*sp.diff(p,x,2)+(1-x)*sp.diff(p,x)+n*p
            else: ode=(1-x*x)*sp.diff(p,x,2)-x*sp.diff(p,x)+n*n*p
            exact(6,f'{name} ODE n={n}',ode)
            exact(6,f'{name} generating coefficient n={n}',gs[name].coeff(t,n)-(p/sp.factorial(n) if name=='H' else p))
            if name=='P':rod=sp.diff((x*x-1)**n,x,n)/(2**n*sp.factorial(n))
            elif name=='H':rod=(-1)**n*sp.exp(x*x)*sp.diff(sp.exp(-x*x),x,n)
            elif name=='L':rod=sp.exp(x)/sp.factorial(n)*sp.diff(x**n*sp.exp(-x),x,n)
            else:rod=None
            if rod is not None: exact(6,f'{name} Rodrigues n={n}',p-rod)
            if n>=1:
                pm,pp=fun(n-1,x),fun(n+1,x)
                if name=='P':r=(n+1)*pp-(2*n+1)*x*p+n*pm
                elif name=='H':r=pp-2*x*p+2*n*pm
                elif name=='L':r=(n+1)*pp-(2*n+1-x)*p+n*pm
                else:r=pp-2*x*p+pm
                exact(6,f'{name} recurrence n={n}',r)
            if name=='P':series=sum((-1)**k*sp.factorial(2*n-2*k)*x**(n-2*k)/(2**n*sp.factorial(k)*sp.factorial(n-k)*sp.factorial(n-2*k)) for k in range(n//2+1))
            elif name=='H':series=sum((-1)**k*sp.factorial(n)*(2*x)**(n-2*k)/(sp.factorial(k)*sp.factorial(n-2*k)) for k in range(n//2+1))
            elif name=='L':series=sum((-1)**k*sp.binomial(n,k)*x**k/sp.factorial(k) for k in range(n+1))
            else:series=None
            if series is not None:exact(6,f'{name} explicit coefficients n={n}',p-series)
            if name=='T':
                for theta in [0,.4,1.3,math.pi]:near(6,f'Chebyshev cosine n={n},theta={theta}',float(p.subs(x,math.cos(theta))),math.cos(n*theta),1e-12)
            if name=='H':
                psi=sp.exp(-x*x/2)*p
                exact(6,f'oscillator bound state n={n}',sp.diff(psi,x,2)+(2*n+1-x*x)*psi)
                exact(6,f'Hermite derivative n={n}',sp.diff(p,x)-(2*n*fun(n-1,x) if n else 0))
        if name in ['P','H','L']:
            for n in range(6):
                for m in range(6):
                    norm={'P':sp.Rational(2,2*n+1),'H':2**n*sp.factorial(n)*sp.sqrt(sp.pi),'L':sp.Integer(1)}[name]
                    exact(6,f'{name} weighted orthogonality n={n},m={m}',moment_integral(fun(n,x)*fun(m,x),name)-(norm if n==m else 0))
    for n in range(6):
        for m in range(6):
            val=mp.quad(lambda q:mp.cos(n*q)*mp.cos(m*q),[0,mp.pi])
            rhs=mp.pi if n==m==0 else (mp.pi/2 if n==m else 0)
            near(6,f'Chebyshev weighted orthogonality n={n},m={m}',val,rhs,1e-35)
    for n in range(1,6):
        for m in range(n+1):
            p=(-1)**m*(1-x*x)**sp.Rational(m,2)*sp.diff(sp.legendre(n,x),x,m)
            exact(6,f'associated Legendre ODE n={n},m={m}',(1-x*x)*sp.diff(p,x,2)-2*x*sp.diff(p,x)+(n*(n+1)-m*m/(1-x*x))*p)
    for n in range(1,9):
        for k in range(1,n+1):
            a=math.cos((2*k-1)*math.pi/(2*n))
            near(6,f'Chebyshev node n={n},k={k}',float(sp.chebyshevt(n,x).subs(x,a)),0,5e-12)
    # Verify all existing Airy plot samples against the independent library.
    table=np.loadtxt(ROOT/'figs'/'airy-ai-bi.dat',skiprows=1)
    expected=special.airy(table[:,0])
    err=max(np.max(abs(table[:,1]-expected[0])),np.max(abs(table[:,2]-expected[2])))
    check(3,'original Airy plot data',err<5.1e-11,f'all {len(table)} points; absolute error={err:.4e}; 10-digit source rounding')

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'review'/'math_checks.json')
    args=parser.parse_args();start=time.time()
    for function in [chapter1,chapter2,chapter3,chapter4,chapter5,chapter6]:
        print('Checking',function.__name__,flush=True)
        function()
    files=[ROOT/f'mp_ch{n}.tex' for n in range(1,8)]+[ROOT/'Kyushu.sty']
    files+=sorted((ROOT/'figs').glob('*.dat'))+sorted((ROOT/'figs').glob('*.tex'))
    report={'scope':'Finite symbolic/numerical checks of independently transcribed formulas; not a proof of every statement.',
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'equation_labels':{str(n):re.findall(r'\\label\{([^}]+)\}',(ROOT/f'mp_ch{n}.tex').read_text()) for n in range(1,8)},
            'count':len(results),'passed':sum(r['passed'] for r in results),
            'failed':[r for r in results if not r['passed']],
            'elapsed_seconds':round(time.time()-start,2),'checks':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['count','passed','failed','elapsed_seconds']},indent=2))
    raise SystemExit(0 if not report['failed'] else 1)
if __name__=='__main__':main()
