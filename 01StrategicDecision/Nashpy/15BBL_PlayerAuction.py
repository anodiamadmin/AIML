import numpy as np
from matplotlib import pyplot as plt

analysis_ranking = np.linspace(0, 14, 15)
fan_base_rating = np.linspace(2, 5, 16)
analysis_mesh, fan_base_mesh = np.meshgrid(analysis_ranking, fan_base_rating)
player_list = ['Khan', 'Shangha', 'Ross', 'Farooqi', 'Davies', 'Gilkes', 'McAndrew', 'Tanveer',
               'Hales', 'Sams', 'Dogget', 'Warner', 'Green', 'Cutting', 'Cummins']


def valuation_function(analysis_score, fan_base_score, franchise_budget=0., sponsor_agreement=0.):
    return (3 + 0.4 * franchise_budget * analysis_score,
            np.exp((fan_base_score - 2) * sponsor_agreement) - 1)


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(10, 6))

# calculate common_value from proper sport's analysis and franchise's budget
common_value, private_value = valuation_function(analysis_mesh, fan_base_mesh,
                                                 franchise_budget=2, sponsor_agreement=1.)
team_auction = common_value + private_value
# print(f'Common Value: {common_value}')
# print(f'Private Value: {private_value}')
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(analysis_mesh, fan_base_mesh, np.zeros(common_value.shape), color='grey',
                linewidth=.4, alpha=.3)
ax.plot_surface(analysis_mesh, fan_base_mesh, team_auction, color='green',
                linewidth=.4, alpha=.3, label='Private Value')
ax.plot_surface(analysis_mesh, fan_base_mesh, common_value, color='yellow',
                linewidth=.4, alpha=.2, label='Common Value')

scale = np.linspace(.5, 1.5, 15)
actual_fan_base_rating = np.round(np.random.normal(3.5, .5, 15) * scale, 1)
if any(actual_fan_base_rating < 2):
    actual_fan_base_rating[actual_fan_base_rating < 2] = 2
if any(actual_fan_base_rating > 5):
    actual_fan_base_rating[actual_fan_base_rating > 5] = 5

for player in range(len(player_list)):
    cv, pv = valuation_function(player, actual_fan_base_rating[player],
                                franchise_budget=2, sponsor_agreement=1.)
    auction_value = cv + pv
    # print(f'Player {player} -> {actual_fan_base_rating[player - 1]} -> {cv} -> {pv} '
    #       f'-> {auction_value}')
    ax.plot([player, player], [actual_fan_base_rating[player], actual_fan_base_rating[player]],
            [0, cv], color='yellow', linewidth=3)
    ax.scatter(player, actual_fan_base_rating[player], cv, color='k', s=20, alpha=1, marker='d')
    ax.plot([player, player], [actual_fan_base_rating[player], actual_fan_base_rating[player]],
            [cv, auction_value], color='green', linewidth=3)
    ax.plot([player, player], [2, 2], [0, cv], color='grey', linewidth=.5, alpha=.5)
    ax.plot([player, player], [2, 5], [cv, cv], color='yellow', linewidth=.6, alpha=1)
    c_val, p_val = valuation_function(np.ones(len(fan_base_rating))*player, fan_base_rating,
                                      franchise_budget=2, sponsor_agreement=1.)
    ax.plot(np.ones(len(fan_base_rating))*player, fan_base_rating,
            c_val + p_val, color='green', linewidth=.5, alpha=1)
    ax.scatter(player, 2, cv, color='yellow', s=20, alpha=1, marker='d', edgecolors='grey')
    ax.scatter(player, actual_fan_base_rating[player], auction_value, color='red', s=100, alpha=1,
               marker='*')

ax.legend(loc='upper left', fontsize=8)
ax.set_title("BBL Auction Price - Sydney Thunders")
ax.set_ylabel("Fan Base Rating ->", fontsize=8)
ax.set_zlabel("Final Bid Value (K AU$)  ->\n=  Common Value (player ranking)"
              "\n  +  Private Value (fan base rating)",
              fontsize=8)
ax.set_xticks(analysis_ranking)
ax.set_xticklabels(player_list, fontsize=6, rotation=90)
ax.set_yticks(fan_base_rating[0:len(fan_base_rating):5])
ax.set_yticklabels(np.round(fan_base_rating[0:len(fan_base_rating):5], 1), fontsize=6)
zticks = np.arange(0, 35, 5)
zlabels = [str(i*40) for i in zticks]
ax.set_zticks(zticks)
ax.set_zticklabels(zlabels, fontsize=6)
plt.savefig('./image/15BBL_Player_Auction.png')
plt.show()
