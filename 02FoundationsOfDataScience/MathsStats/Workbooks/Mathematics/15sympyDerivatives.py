import sympy as sp

# Derivatives
x0, x1 = sp.symbols('x0:2')

expr = sp.exp(2*x0) + sp.sin(x0)
print(f'expr = {expr}')
print(f'sp.diff(expr) = {sp.diff(expr)} :: sp.diff(expr, x0) = {sp.diff(expr, x0)}')
print(f'sp.diff(sp.diff(expr, x0), x0) = {sp.diff(sp.diff(expr, x0), x0)}')
print(f'sp.diff(expr, x0, x0) = {sp.diff(expr, x0, x0)}')
print(f'sp.diff(expr, x0, 2) = {sp.diff(expr, x0, 2)}')
print(f'sp.diff(expr, x0, 7) = {sp.diff(expr, x0, 7)}')

# Rewrite a fn in terms of another fn
expr = sp.exp(sp.sinh(x0)) / sp.exp(sp.exp(x0)/2)
print(f'expr = {expr}')
print(f'expr.rewrite(sp.exp) = {expr.rewrite(sp.exp)}')
print(f'expr.rewrite(sp.exp).simplify() = {expr.rewrite(sp.exp).simplify()}')

# Partial derivatives
expr_2var = sp.cos(x0) + sp.cot(2*x1)**2
print(f'expr_2var = {expr_2var}')
diff_wrt_x1 = expr_2var.diff(x1)
print(f'diff_wrt_x1 = {diff_wrt_x1}')

# rewrite and simplify to pick the simple forms Use the order properly
diff_wrt_x1_simple = expr_2var.diff(x1).simplify()
print(f'diff_wrt_x1_simple = {diff_wrt_x1_simple}')
sin_rewrite_diff_wrt_x1_simple = diff_wrt_x1_simple.rewrite(sp.sin)
print(f'sin_rewrite_diff_wrt_x1_simple = {sin_rewrite_diff_wrt_x1_simple}')

# Multiple partial derivatives
print(f'expr_2var = {expr_2var}')
part_derivative = expr_2var.diff(x0, x0, x0, x1, x1)
# part_derivative = expr_2var.diff(x1, x0, x0, x1, x0)  # Order of derivative does not matter
# part_derivative = expr_2var.diff(x0, 3, x1, 2)        # Same effect as above
print(f'part_derivative = expr_2var.diff(x0, x0, x0, x1, x1) = {part_derivative}')
