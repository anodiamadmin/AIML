import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simple Linear Regression
# # One predictor or independent variable - 'x' to predict dependent the target or variable - 'y'
# # Linear => y = b0 + b1 * x : b0 = intercept, b1 = slope
# Multiple Linear regression => multiple predictors or independent variables - 'x1, x2, ... xn'
# # to predict dependent the target or variable - 'y'
# # y = b0 + b1 * x1 + b2 * x2 + ... + bn * xn

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df_used_cars = pd.read_csv('./data/01UsedCarsData.csv')
X = df_used_cars[['highway-mpg', 'horsepower']].to_numpy()
Y = df_used_cars[['price']].to_numpy()
lr_obj = LinearRegression()
lr_obj.fit(X, Y)
Y_hat = lr_obj.predict(X)
df_X_Y_Yhat = pd.DataFrame({'X0': np.round(X[:, 0].flatten()).astype(int),
                            'X1': np.round(X[:, 1].flatten()).astype(int),
                            'Y': Y.flatten(), 'Y_hat': Y_hat.flatten()},
                           columns=['X0', 'X1', 'Y', 'Y_hat'])
# print(f'df_X_Y_Yhat\n{df_X_Y_Yhat}')
# print(f'intercept=lr_obj.intercept_={lr_obj.intercept_[0]}'
#       f'::slope=lr_obj.coef_={lr_obj.coef_[0][0]}'
#       f'::slope=lr_obj.coef_={lr_obj.coef_[0][1]}')
equation = (f'Y_hat = {round(lr_obj.intercept_[0], 3)} + X0 * ({round(lr_obj.coef_[0][0], 3)})'
            f' + X1 * ({round(lr_obj.coef_[0][1], 3)})')
# print(f'{equation}')
Y_hat_27_5_130 = lr_obj.intercept_[0] + 27.5*lr_obj.coef_[0][0] + 130*lr_obj.coef_[0][1]
# print(f'Y_hat_27_5_130 = {Y_hat_27_5_130}')
X0_range = np.linspace(df_X_Y_Yhat[['X0']].to_numpy().min(),
                       df_X_Y_Yhat[['X0']].to_numpy().max(), 10)
X1_range = np.linspace(df_X_Y_Yhat[['X1']].to_numpy().min(),
                       df_X_Y_Yhat[['X1']].to_numpy().max(), 10)
X_0, X_1 = np.meshgrid(X0_range, X1_range)
Y_hat_range = lr_obj.intercept_[0] + X_0 * lr_obj.coef_[0][0] + X_1 * lr_obj.coef_[0][1]
# print(f'Y_hat_range = {Y_hat_range}')

plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=plt.figaspect(0.4))
elevation = 20
roll = 0

azimuth = 33
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter3D(df_X_Y_Yhat.X0, df_X_Y_Yhat.X1, df_X_Y_Yhat.Y, color='red', marker='o')
ax.scatter3D(27.5, 130, Y_hat_27_5_130, color='black', marker='d')
ax.plot_surface(X_0, X_1, Y_hat_range, color='purple', alpha=0.5, linewidth=1, label='SLR')
ax.view_init(elevation, azimuth, roll)
ax.set_title("Multiple Linear Regression")
ax.set_xlabel("<-- X0")
ax.set_ylabel("X1 -->")
ax.set_zlabel("<-- Y")

azimuth = 36
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(df_X_Y_Yhat.X0, df_X_Y_Yhat.X1, df_X_Y_Yhat.Y, color='red', marker='o')
ax.scatter3D(27.5, 130, Y_hat_27_5_130, color='black', marker='d')
ax.plot_surface(X_0, X_1, Y_hat_range, color='purple', alpha=0.5, linewidth=1, label='SLR')
ax.view_init(elevation, azimuth, roll)
ax.set_title("Multiple Linear Regression")
ax.set_xlabel("<-- X0")
ax.set_ylabel("X1 -->")
ax.set_zlabel("<-- Y")

plt.tight_layout()
plt.savefig('./plots/06MultipleLinearRegression_3D.png')
plt.show()
