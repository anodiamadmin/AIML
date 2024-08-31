import sympy as sp

# Derivatives
x, y, z = sp.symbols('x y z', real=True)
C = sp.symbols('C', real=True)

# Indefinite integrals
f = sp.cos(2*x) * sp.sin(2*x)
print(f'f = {f}')
print(f'sp.integrate(f) = {sp.integrate(f)}\n'
      f'sp.integrate(f).diff(x) = {sp.integrate(f).diff(x)}')
print(f'sp.integrate(f) = {sp.integrate(f) + C} '
      f':: Adding constant bias C to indefinite integral')
print(f'sp.integrate(f) = {sp.integrate(f, x)}')

g = x**y
h = sp.integrate(g, x)
print(f'sp.integrate(g, x) = {h}, type(h) = {type(h)}')
print(f'(h*f).simplify() = {(h*f).simplify()}')
print(f'sp.integrate(g, x, y) = {sp.integrate(g, x, y)}')

# DEFINITE INTEGRALS
f1 = sp.sin(3*x) * sp.cos(2*x)
print(f'sp.integrate(f1, (x, 0, sp.pi)) = {sp.integrate(f1, (x, 0, sp.pi))}')

# Improper Definite INTEGRALS one side of the function is open / limit tends to sp.oo
f2 = sp.exp(-x)
print(f'sp.integrate(f2, (x, 0, sp.oo)) = {sp.integrate(f2, (x, 0, sp.oo))}')

# Multivariable integrals
f3 = x*y*z**2
print(f'sp.integrate(f3, x, y) = {sp.integrate(f3, x, y)}')

# Multivariable definite integrals
print(f'sp.integrate(f3, (x, 0, 1), (y, 0, 1), (z, 0, 1)) = '
      f'{sp.integrate(f3, (x, 0, 1), (y, 1, 5), (z, 0, 3))}')
