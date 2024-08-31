import sympy as sp

# Define symbolic variables
x, theta = sp.symbols('x theta')

# expression = 3*x**3
# expression = x*sp.exp(-1*theta*x)
expression = (x**2)*sp.exp(-x)
# expression = 3/2*(x**2)
print(f'expression = {expression}')
print(f'expression.integral(x) = {expression.integrate(x)}')
