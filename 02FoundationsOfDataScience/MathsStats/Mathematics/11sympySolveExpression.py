import sympy as sp

x, y, z = sp.symbols('x, y, z')

# Define an equation: '=' used for assignment, '==' for comparison in Python
# Therefor the Eq class constructor sp.Eq() used for solving sympy equations
equation = sp.Eq(x**2 + 2*x - 15, 0)   # lhs = rhs: sp.Eq(lhs, rhs)
print(f'equation = {equation}, type(equation) = {type(equation)}')

# sp.solveset(eqn) function used for solving linear equations
solutions = sp.solveset(equation, x)
print(f'solutions = {solutions}, type(solutions) = {type(solutions)}')
print(f'solutions.args[0] = {solutions.args[0]}')
for i, solution in enumerate(list(solutions)):
    print(f'solution[{i}] = {solution}')

print(f'sp.solveset(x**2-9) = {sp.solveset(x**2 - 9)}')
print(f'sp.solveset(cos(x)-sin(x)) = {sp.solveset(sp.cos(x)-sp.sin(x))}\n'
      f'type(sp.solveset(cos(x)-sin(x))) = {type(sp.solveset(sp.cos(x)-sp.sin(x)))}')

numerically_solvable_eqn = sp.Eq(sp.cos(x), x)
print(f'numerically_solvable_eqn = {numerically_solvable_eqn}\n'
      f'sp.solveset(numerically_solvable_eqn) = {sp.solveset(numerically_solvable_eqn)}\n'
      f'type(sp.solveset(numerically_solvable_eqn)) = {type(sp.solveset(numerically_solvable_eqn))}')

# Solving system of linear equations
eq1 = sp.Eq(40*x + 2*y, 3*z + x)
eq2 = sp.Eq(7*x + 13*y, 5*z + 2*x)
eq3 = sp.Eq(2*x + y, 6*z)
sol = sp.linsolve([eq1, eq2, eq3], x, y, z)
print(f'eq1 = {eq1}, eq2 = {eq2}, eq3 = {eq3}')
print(f'sol = {sol}, type(sol) = {type(sol)}')
sol_part = sp.linsolve([eq1, eq2,], x, y, z)
print(f'sol_part = {sol_part}, type(sol_part) = {type(sol_part)}')
