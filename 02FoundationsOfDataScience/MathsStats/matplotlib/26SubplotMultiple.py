import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-v0_8')

df_scores = pd.read_csv('./data/testBatting.csv')
df_years, df_sachin, df_sourav, df_dravid, df_laxman = (df_scores['Year'], df_scores['Sachin'],
                                                        df_scores['Sourav'], df_scores['Dravid'],
                                                        df_scores['Laxman'])

fig1, ax1 = plt.subplots(figsize=(7, 4))
fig2, ax2 = plt.subplots(figsize=(7, 4))
fig3, ax3 = plt.subplots(figsize=(7, 4))
fig4, ax4 = plt.subplots(figsize=(7, 4))

ax1.plot(df_years, df_sachin, label='Sachin', color='red', linewidth=2, linestyle='--', marker='v')
ax1.legend(loc='upper left')
ax1.set_title("Sachin's Test Scores")
ax1.set_xlim([1988, 2014])
ax1.set_ylim([50, 1750])
ax1.set_xlabel("Year")
ax1.set_ylabel("Yearly Total")

ax2.plot(df_years, df_sourav, label='Sourav', color='blue', linewidth=2, linestyle=':', marker='o')
ax2.legend(loc='upper right')
ax2.set_title("Sourav's Test Scores")
ax2.set_xlim([1988, 2014])
ax2.set_ylim([50, 1750])
ax2.set_xlabel("Year")
ax2.set_ylabel("Yearly Total")

ax3.plot(df_years, df_dravid, label='Dravid', color='green', linewidth=2, linestyle='-', marker='*')
ax3.legend(loc='lower left')
ax3.set_title("Dravid's Test Scores")
ax3.set_xlim([1988, 2014])
ax3.set_ylim([50, 1750])
ax3.set_xlabel("Year")
ax3.set_ylabel("Yearly Total")

ax4.plot(df_years, df_laxman, label='Laxman', color='purple', linewidth=2, linestyle='-.', marker='>')
ax4.legend(loc='lower right')
ax4.set_title("Laxman's Test Scores")
ax4.set_xlim([1988, 2014])
ax4.set_ylim([50, 1750])
ax4.set_xlabel("Year")
ax4.set_ylabel("Yearly Total")

plt.grid(True)
plt.tight_layout()
plt.show()

fig1.savefig('./plots/26SubPlot_Sachin.png')
fig2.savefig('./plots/26SubPlot_Sourav.png')
fig3.savefig('./plots/26SubPlot_Dravid.png')
fig4.savefig('./plots/26SubPlot_Laxman.png')