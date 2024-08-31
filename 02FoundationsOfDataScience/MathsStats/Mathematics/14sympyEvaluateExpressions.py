import sympy as sp

# Complex(by default), Real, +ve, Integer Symbols
x0, x1, x2 = sp.symbols('x0:3')

expr = x0**2 + 5

print(f'expr = {expr}\nexpr.subs(x0, 2)= {expr.subs(x0, 2)}\n'
      f'expr.subs(x0, sp.Pi)= {expr.subs(x0, sp.pi)}\n'
      f'sp.N(expr.subs(x0, sp.Pi))= {sp.N(expr.subs(x0, sp.pi))}')
print(f'expr.subs(x0, x1**2)= {expr.subs(x0, x1**2)}')

print(f'evaluate 3Pi to 4 places of decimals = '
      f'{sp.N(sp.N(sp.pi, 41)*x1.subs(x1, 3), 40)}')
