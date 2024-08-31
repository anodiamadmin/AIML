import sympy as sp

# Limits to real numbers
x = sp.symbols('x')

expr = x**2
print(f'sp.limit(expr, x, 2) = {sp.limit(expr, x, 2)} == '
      f'expr.subs(x, 2) = {expr.subs(x, 2)}')

# Limit at sp.oo
exponent = x/sp.exp(x)
print(f'sp.limit(exponent, x, sp.oo) = {sp.limit(exponent, x, sp.oo)} == '
      f'exponent.subs(x, sp.oo) = {exponent.subs(x, sp.oo)}')

# Limit from Right and Left
expr = 1/x
print(f'Left Limit: sp.limit(expr, x, 0, \'-\') = {sp.limit(expr, x, 0, '-')}')
print(f'Right Limit: sp.limit(expr, x, 0, \'+\') = {sp.limit(expr, x, 0, '+')}')
print(f'Both Direction Limit: sp.limit(expr, x, 0, \'+-\') = '
      f'{sp.limit(expr, x, 0, '+-')}')
