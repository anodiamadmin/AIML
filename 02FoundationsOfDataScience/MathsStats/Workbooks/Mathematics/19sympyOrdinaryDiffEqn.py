import sympy as sp

x = sp.symbols('x')

# Create function f(x)
f = sp.Function('f')(x)

print(f'f.diff() = {f.diff()}')
print(f'f.integrate() = {f.integrate()}')

my_diff_eq = sp.Eq(x*f.diff(x, x)+f.diff(x), x**3)
print(f'\nmy_diff_eq :: {my_diff_eq}')
print(f'my_diff_eq.lhs = {my_diff_eq.lhs}\n'
      f'my_diff_eq.rhs = {my_diff_eq.rhs}')

sol = sp.dsolve(my_diff_eq, f)
print(f'Solution for f :: {sol} :: type(sol)={type(sol)}')
C2, _, C1 = tuple(sol.rhs.free_symbols)
# sol = sol.rhs.subs(C2, 0).subs(C1, 1)
sol = sol.rhs.subs({C2: 0, C1: 1})
print(f'sol = {sol}')
