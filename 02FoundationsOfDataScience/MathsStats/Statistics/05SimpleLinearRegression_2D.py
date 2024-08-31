import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simple Linear Regression
# # One predictor or independent variable - 'x' to predict dependent the target or variable - 'y'
# # Linear => y = b0 + b1 * x : b0 = intercept, b1 = slope
# Multiple Linear regression => multiple predictors or independent variables - 'x1, x2, ... xn'
# # to predict dependent the target or variable - 'y'
# # y = b0 + b1 * x1 + b2 * x2 + ... + bn * xn
# # Draw diagram for concept of error term - 'e' and cost function - 'J'= sum(e^2) and minimizing 'J'

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df_used_cars = pd.read_csv('./data/01UsedCarsData.csv')
X = df_used_cars[['highway-mpg']].to_numpy()
Y = df_used_cars[['price']].to_numpy()
lr_obj = LinearRegression()
lr_obj.fit(X, Y)
Y_hat = lr_obj.predict(X)
df_X_Y_Yhat = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten(), 'Y_hat': Y_hat.flatten()}, columns=['X', 'Y', 'Y_hat'])
print(f'df_X_Y_Yhat\n{df_X_Y_Yhat}')
# print(f'intercept=lr_obj.intercept_={lr_obj.intercept_[0]}::slope=lr_obj.coef_={lr_obj.coef_[0][0]}')
equation = f'Y_hat = {round(lr_obj.intercept_[0], 3)} + X * ({round(lr_obj.coef_[0][0], 3)})'
# print(f'{equation}')
Y_hat_27_5 = lr_obj.intercept_[0] + 27.5*lr_obj.coef_[0][0]
# print(f'Y_hat_27_5 = {Y_hat_27_5}')
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Regression - Predict Price from Fuel Consumption')
ax.scatter(df_X_Y_Yhat.X, df_X_Y_Yhat.Y)
ax.plot(df_X_Y_Yhat.X, df_X_Y_Yhat.Y_hat, c='red', alpha=0.5, linewidth=1, label='SLR')
ax.axvline(27.5, c='k', alpha=0.5, linewidth=1, linestyle='--')
ax.axhline(Y_hat_27_5, c='k', alpha=0.5, linewidth=1, linestyle='--')
ax.scatter(27.5, Y_hat_27_5, c='orange')
ax.set_ylabel('USD')
ax.set_xlabel('Highway Miles Per Gallon')
ax.legend(loc='upper right')
ax.grid(True)
fig.text(.5, .9, bbox=dict(facecolor='red', alpha=0.5), s=equation)
plt.tight_layout()
plt.savefig('./plots/05SimpleLinearRegression_2D.png')
plt.show()
