import sympy as sp

x, y = sp.symbols('x, y')

poly = x**2 + 2*x - 15
print(f'poly = {poly}')
factorised = sp.factor(poly)
print(f'factorised = {factorised}')
expanded = sp.expand(factorised)             # sp.expand(expr) method for SIMPLIFICATION
print(f'Simplified: expanded = {expanded}')

expr = sp.exp(x + y)
print(f'expr = {expr}')
expr = sp.exp(x+y).expand()                  # expr.expand() for SIMPLIFICATION
print(f'Simplified: expr.expand() = {expr}')

trig = (sp.cos(x) + sp.sin(y))**2 - sp.cos(x)**2
print(f'trig = {trig}')
simplified = sp.expand(trig)              # sp.simplify(expr) method for SIMPLIFICATION
print(f'Simplified: sp.expand(trig) = {simplified}')

trig_and_exp = sp.exp(x + y) + (sp.cos(x) + sp.sin(y))**2 - sp.cos(x)**2
print(f'Expanded: sp.expand(trig_and_exp) = {trig_and_exp.expand()}')
print(f'Expand trig part only = {trig_and_exp.expand(power_exp=False)}')

p = x**3 + 10*x**2 + 31*x + 30
q = x**2 + 12*x + 35
print(f'sp.factor(p) = {sp.factor(p, x)}; sp.factor(q) = {sp.factor(q, x)}')
fraction = p/q
print(f'Fraction cancel simplify: {sp.cancel(fraction), fraction.simplify()}')
print(f'Fraction apart helps in integration: {sp.apart(fraction)}')

# sp.simplify() is versatile but slow
expr = sp.exp(x + y) + (sp.cos(x) + sp.sin(x))**2 - sp.cos(x) - sp.exp(x)
print(f'expr = {expr}, expr.simplify() = {expr.simplify()}')
