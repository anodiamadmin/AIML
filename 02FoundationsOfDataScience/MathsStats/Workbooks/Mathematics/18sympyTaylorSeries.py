import sympy as sp

# All arbitrary functions can be expanded into infinite series Polynomials
# This is called the Taylor Series expansion
# Polynomials are sometimes easier to deal with than arbitrary functions
x = sp.symbols('x')
expr = sp.sin(x)

print(f'Taylor Series Expansion of sin(x) at x=pi/2, and up to term 5 (default) terms:\n'
      f'sp.series(expr, x, sp.pi/2) = {sp.series(expr, x, sp.pi/2)}')

expr1 = sp.exp(sp.sin(x))
print(f'Taylor Series Expansion of exp.sin(x) at x=0, and up to term 8 terms:\n'
      f'sp.series(expr1, x, 0, n=8) = {sp.series(expr1, x, 0, n=8)}')

print(f'Taylor Series Expansion of exp.sin(x)*sin(x) at x=0:\n'
      f'sp.series(expr1, x, 0, n=6)*sp.series(sp.sin(x), x, 0, n=6) = '
      f'{(sp.series(expr1, x, 0, n=8)*sp.series(sp.sin(x), x, 0, n=6)).expand()}')

print(f'Differentiate Taylor Series exp.sin(x) at x=0 n=8:\n'
      f'sp.diff(sp.series(expr1, x, 0, n=8)) = '
      f'{sp.diff(sp.series(expr1, x, 0, n=8))}')

print(f'Integrate Taylor Series exp.sin(x) at x=0 n=8:\n'
      f'sp.integrate(sp.series(expr1, x, 0, n=8)) = '
      f'{sp.integrate(sp.series(expr1, x, 0, n=8))}')

print(f'Remove O: Integrate Taylor Series exp.sin(x) at x=0 n=8:\n'
      f'sp.integrate(sp.series(expr1, x, 0, n=8).removeO() = '
      f'{sp.integrate(sp.series(expr1, x, 0, n=8)).removeO()}')
