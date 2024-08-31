import sympy as sp

x = sp.symbols('x')
expr = x**(1/3)
print(f'expr = {expr}')
expr = x**(sp.Integer(1)/sp.Integer(3))
print(f'expr = {expr}')
print(f'expr = {expr**(sp.Integer(1)/5)}')

print(f'sp.rational(1/7) = {sp.Rational(1, 7)}')
print(f'sp.rational(1/12) = {(x**sp.Rational(1, 4))**sp.Rational(1, 3)}')

# CONSTANTS
print(f'3*sp.pi = {3*sp.pi}, sp.E + sp.pi = {sp.E+sp.pi}, sp.oo = {sp.oo}')
print(f'sp.oo > 999999 = {sp.oo > 999999}')
print(f'sp.oo + sp.pi  = {sp.oo + sp.pi}')
print(f'sp.oo + sp.oo  = {sp.oo + sp.oo}')
print(f'sp.oo - sp.oo  = {sp.oo - sp.oo}')
print(f'sp.I = {sp.I}, sp.I**2 = {sp.I**2}')
print(f'sp.exp(x) = {sp.exp(x)}')
# Euler's Identity => e^(i*x)=cos(x)+i.sin(x), cos(pi)=-1, sin(pi)=0 => e^(i*pi)=-1
print(f'sp.E = {sp.exp(sp.I*sp.pi)}')
print(f'sp.ln(x) = {sp.ln(x)}, sp.log(x) = {sp.log(x)} :: Both are same')
print(f'sp.exp(sp.ln(x)) = {sp.exp(sp.ln(x))}, sp.log(sp.exp(x)) = {sp.log(sp.exp(x))}')
print(f'sp.sin(x)**2 + sp.cos(x)**2 = {(sp.sin(x))**2 + (sp.cos(x))**2}')
