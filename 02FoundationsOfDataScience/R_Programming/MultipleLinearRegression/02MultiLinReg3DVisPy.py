import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df_used_cars = pd.read_excel('HousingPrices.xlsx')
X = df_used_cars[['Location', 'Rooms']].to_numpy()
Y = df_used_cars[['Price']].to_numpy()

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
print(f'{equation}')
Y_hat_Dandenong0_Rooms9 = lr_obj.intercept_[0] + 0*lr_obj.coef_[0][0] + 9*lr_obj.coef_[0][1]
print(f'Y_hat_Dandenong0_Rooms9 = {Y_hat_Dandenong0_Rooms9}')
X0_range = np.linspace(df_X_Y_Yhat[['X0']].to_numpy().min(),
                       df_X_Y_Yhat[['X0']].to_numpy().max(), 10)
X1_range = np.linspace(df_X_Y_Yhat[['X1']].to_numpy().min(),
                       df_X_Y_Yhat[['X1']].to_numpy().max(), 10)
X_0, X_1 = np.meshgrid(X0_range, X1_range)
Y_hat_range = lr_obj.intercept_[0] + X_0 * lr_obj.coef_[0][0] + X_1 * lr_obj.coef_[0][1]
# print(f'Y_hat_range = {Y_hat_range}')

plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter3D(df_X_Y_Yhat.X0, df_X_Y_Yhat.X1, df_X_Y_Yhat.Y, color='red', marker='o')
ax.scatter3D(0, 9, Y_hat_Dandenong0_Rooms9, color='black', marker='d', s=100)
ax.plot_surface(X_0, X_1, Y_hat_range, color='purple', alpha=0.5, linewidth=1, label='SLR')
ax.set_title(f"Multiple Linear Regression\n{equation}")
ax.set_ylabel("Bedrooms -->")
ax.set_zlabel("Price Au$1000 -->")
ax.set_xlim(0, 1)
ax.set_ylim(0, 14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 5, 10, 15])
ax.set_xticklabels(['Dandenong', 'Sunshine'])

plt.savefig('./plots/A2Eco.png')
plt.show()
