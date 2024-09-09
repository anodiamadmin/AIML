import sympy as sp

# Define symbolic variables
x, beta = sp.symbols('x beta')

# expression = 4 * sp.ln(x) - x
# expression = 20 * x - 2 * x**2
# expression = -5 * x
# expression = 10 * x - x**2 - 2 * 2
# expression = -sp.ln(1-x)
# expression = 1/beta * sp.exp(-x/beta)
expression = sp.sqrt(2*x)
print(f'expression = {expression}')
print(f'expression.diff(x) = {expression.diff(x)}')
