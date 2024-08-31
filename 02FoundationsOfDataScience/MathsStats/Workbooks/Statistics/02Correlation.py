import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Correlation: interdependency of 2 variables (numeric), one variable is the output, another is a feature
# e.g. Time spent on social media Vs exam score
# Does not give causation/ causality. i.e. It is not known if time spent on social media is the cause
# of the exam score or if the exam score is the cause of the time spent on social media

# Pearson Correlation:
# # Correlation coefficient:
# # #         ~ +1 => large +ve correlation
# # #         ~ -1 => large -ve correlation
# # #         ~ 0 => no correlation
# # P-value:
# # #         < 0.001 => Strong certainty
# # #         < 0.05  => Moderate certainty
# # #         < 0.1   => Weak certainty
# # #         > 0.1   => No certainty
def correlation_strength(pearson_coeff=0, p_value=0):
    strength = 'No'
    if pearson_coeff < -0.85:
        strength = 'Strong -ve'
    elif pearson_coeff < -0.65:
        strength = '-ve'
    elif pearson_coeff < -0.2:
        strength = 'Weak -ve'
    elif pearson_coeff < 0.2:
        strength = 'No'
    elif pearson_coeff < 0.65:
        strength = 'Weak +ve'
    elif pearson_coeff < 0.85:
        strength = '+ve'
    else:
        strength = 'Strong +ve'
    certainty = 'No'
    if p_value < 0.001:
        certainty = 'Strong'
    elif p_value < 0.05:
        certainty = 'Moderate'
    elif p_value < 0.1:
        certainty = 'Weak'
    else:
        certainty = 'No'
    return strength, certainty

def draw_image(img_obj, X='engine-size', Y='price', x_pos=.1, y_pos=.8):
    pearson_coeff, p_value = stats.pearsonr(df_used_cars[X], df_used_cars[Y])
    strength, certainty = correlation_strength(pearson_coeff, p_value)
    text = (f'{strength} Correlation\nPearson-coeff={round(pearson_coeff, 2)}\n'
              f'{certainty} Certainty\nP-value~={round(p_value, 6)}')
    img_obj.set_title(f'{X} vs {Y}')
    sns.regplot(x=X, y=Y, data=df_used_cars, ax=img_obj)
    img_obj.set_ylim(0, )
    fig.text(x_pos, y_pos, text, bbox=dict(facecolor='yellow', alpha=0.5))
    return

# pd.options.display.width = None
# pd.options.display.max_columns = None
# pd.set_option('display.max_rows', 3000)
# pd.set_option('display.max_columns', 3000)
df_used_cars = pd.read_csv('./data/01UsedCarsData.csv')
# print(df_used_cars.head(50))

fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(11, 6))

draw_image(ax[0][0], X='engine-size')
draw_image(ax[0][1], X='highway-mpg', x_pos=.8)
draw_image(ax[1][0], X='peak-rpm', x_pos=.35, y_pos=.3)
draw_image(ax[1][1], X='horsepower', x_pos=.6, y_pos=.3)

plt.tight_layout()
plt.savefig('./plots/02Correlation.png')
plt.show()
