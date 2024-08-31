import sympy as sp

# Complex(by default), Real, +ve, Integer Symbols
z = sp.symbols('z')
print(f'exp(log(z)) = {sp.exp(sp.log(z))}')
print(f'log(exp(z)) = {sp.log(sp.exp(z))} :: not = z always. z is complex by default')

# real number solutions
x_vector = sp.symbols('x0:10', real=True)    # x0 to x9 are real numbers
print(f'x_vector = {x_vector}')
print(f'log(exp(x0)) = {sp.log(sp.exp(x_vector[0]))} :: x0 to x9 are real numbers')
print(f'sqrt(x1**2) = {sp.sqrt(x_vector[1]**2)} :: x0 to x0 are real numbers')

# +ve symbols
x0, x1, x2 = sp.symbols('x0:3', positive=True)    # x0 to x2 are real numbers
print(f'sqrt(x1**2) = {sp.sqrt(x1**2)} :: x0 to x0 are real numbers')

# Integer symbols
n = sp.symbols('n', Integer=True)    # Integer
print(f'(-1)**(2*n) = {sp.simplify((-1)**(2*n))}')    # should give 1 as result
