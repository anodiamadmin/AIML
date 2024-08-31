import sympy as sp

# Define symbolic variables
x = sp.symbols('x')

# expression = 4 * sp.ln(x) - x
# expression = 20 * x - 2 * x**2
# expression = -5 * x
# expression = 10 * x - x**2 - 2 * 2
expression = -sp.ln(1-x)
print(f'expression = {expression}')
print(f'expression.diff(x) = {expression.diff(x)}')
