import matplotlib.pyplot as plt
import pandas as pd
import seaborn

plt.style.use('seaborn-v0_8')

df_scores = pd.read_csv('./data/testBatting.csv')
# df_years, df_sachin, df_sourav, df_dravid, df_laxman = (df_scores['Year'], df_scores['Sachin'],
#                                                         df_scores['Sourav'], df_scores['Dravid'],
#                                                         df_scores['Laxman'])

plt.figure(figsize=(10, 5))

seaborn.regplot(x=df_scores.Year, y=df_scores.Sachin, data=df_scores,
                label='Sachin', color='green', marker='v')
plt.legend(loc='upper left')
plt.title("Sachin's Test Scores")
plt.xlim([1988, 2014])
plt.ylim([50, 1750])
plt.xlabel("Year")
plt.ylabel("Yearly Total")

plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig('./plots/29RegressionPlotSeaborn.png')
