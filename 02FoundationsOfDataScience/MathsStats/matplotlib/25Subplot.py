import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-v0_8')
# print(plt.style.available)

df_scores = pd.read_csv('./data/testBatting.csv')
df_years, df_sachin, df_sourav, df_dravid, df_laxman = (df_scores['Year'], df_scores['Sachin'],
                                                        df_scores['Sourav'], df_scores['Dravid'],
                                                        df_scores['Laxman'])

# fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12 , 6))
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_figheight(6)
fig.set_figwidth(12)

# ax1[0].plot(df_years, df_sachin, label='Sachin', color='red', linewidth=2, linestyle='--', marker='v')
# ax1[1].plot(df_years, df_sourav, label='Sourav', color='blue', linewidth=2, linestyle=':', marker='o')
# ax2[0].plot(df_years, df_dravid, label='Dravid', color='green', linewidth=2, linestyle='-', marker='*')
# ax2[1].plot(df_years, df_laxman, label='Laxman', color='purple', linewidth=2, linestyle='-.', marker='>')

ax[0][0].plot(df_years, df_sachin, label='Sachin', color='red', linewidth=2, linestyle='--', marker='v')
ax[0][0].legend(loc='upper left')
ax[0][0].set_title("Sachin's Test Scores")
# ax[0][0].set_xlim([1988, 2014])
ax[0][0].set_ylabel("Yearly Total")
# ax[0][0].set_xlabel("Year")

ax[0][1].plot(df_years, df_sourav, label='Sourav', color='blue', linewidth=2, linestyle=':', marker='o')
ax[0][1].legend(loc='upper right')
ax[0][1].set_title("Sourav's Test Scores")
# ax[0][1].set_xlim([1988, 2014])
# ax[0][1].set_ylabel("Yearly Total")
# ax[0][1].set_xlabel("Year")

ax[1][0].plot(df_years, df_dravid, label='Dravid', color='green', linewidth=2, linestyle='-', marker='*')
ax[1][0].legend(loc='lower left')
ax[1][0].set_title("Dravid's Test Scores")
# ax[1][0].set_xlim([1988, 2014])
# ax[1][0].set_ylabel("Yearly Total")
ax[1][0].set_xlabel("Year")

ax[1][1].plot(df_years, df_laxman, label='Laxman', color='purple', linewidth=2, linestyle='-.', marker='>')
ax[1][1].legend(loc='lower right')
ax[1][1].set_title("Laxman's Test Scores")
# ax[1][1].set_xlim([1999, 2006])
ax[1][1].set_xlim([1988, 2014])
ax[1][1].set_ylim([50, 1750])
# ax[1][1].set_ylabel("Yearly Total")
# ax[1][1].set_xlabel("Year")

plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/25SubPlot.png')
plt.show()

