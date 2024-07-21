import numpy as np
from matplotlib import pyplot as plt
players, strategies, strategies_row, strategies_col, colors = [], [], [], [], []
game_name, payoff_quantity_name = None, None
symmetric = True

# ######################################
# payoff_matrix = [[(-8, -8), (0, -10)],          # ** Mandatory field **
#                  [(-10, 0), (-1, -1)]]
# game_name = 'Prisoner\'s Dilemma'
# players = ['John', 'Clerk']
# strategies = ['Confess', 'Deny']
# payoff_quantity_name = 'Sentence-Years'
######################################
######################################
payoff_matrix = [[(5, -5), (0, 0), (0, 0)],          # ** Mandatory field **
                 [(0, 0), (0, 0), (0, 0)],
                 [(0, 0), (0, 0), (-5, 5)]]
game_name = 'Politics'
players = ['L-Party', 'R-Party']
strategies = ['Left', 'Mid', 'Right']
payoff_quantity_name = 'Vote Share'
######################################

if not game_name:
    game_name = 'Generic Game'
if not payoff_quantity_name:
    payoff_quantity_name = 'Payoff'
if not players:
    players = ['Row: Player', 'Col: Player']
else:
    players = [i + j for i, j in zip(['Row: ', 'Col: '], players)]
players = [player[:12] for player in players]
if not colors:
    colors = ['red', 'blue']
game = np.array(payoff_matrix)
rows = game.shape[0]
cols = game.shape[1]
rsps = game.shape[2]

if symmetric:
    if not strategies:
        strategies = [f'St: {row}' for row in range(rows)]
    strategies = [strategy[:4] for strategy in strategies]
    strategies_row = strategies
    strategies_col = strategies
else:
    if not strategies_row:
        strategies_row = [f'{players[0][:4]}: {row}' for row in range(rows)]
    if not strategies_col:
        strategies_col = [f'{players[1][:4]}: {col}' for col in range(cols)]

def get_indices(element, lst):
    return [i for i, x in enumerate(lst) if x == element]


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')

R, P = np.meshgrid(np.linspace(0, rows, 101),
                   np.linspace(np.min(game) - (np.max(game) - np.min(game)) * 0.2,
                               np.max(game) + (np.max(game) - np.min(game)) * 0.2, 101))
C = np.zeros((101, 101))
for col in range(cols):
    br_of_row_player_for_this_col = np.max(game[:, col, 0])
    ax.plot(R, C+col+0.5, P, color=colors[1], linewidth=0.3, alpha=0.2)
    ax.plot3D(np.arange(rows) + 0.5, np.full(rows, col + 0.5), game[:, col, 0],
              color=colors[0], linewidth=1, alpha=0.5)
    for row in range(rows):
        if game[row, col, 0] == br_of_row_player_for_this_col:
            ax.scatter(row + 0.5, col + 0.5, game[row, col, 0], color=colors[0], s=200, marker='*',
                       linewidth=.5, alpha=0.5)
        else:
            ax.scatter(row+0.5, col+0.5, game[row, col, 0], color=colors[0], s=30, marker='o',
                       alpha=0.5)

C, P = np.meshgrid(np.linspace(0, cols, 101),
                   np.linspace(np.min(game) - (np.max(game) - np.min(game)) * 0.2,
                               np.max(game) + (np.max(game) - np.min(game)) * 0.2, 101))
R = np.zeros((101, 101))
for row in range(rows):
    br_of_col_player_for_this_row = np.max(game[row, :, 1])
    ax.plot(R+row+0.5, C, P, color=colors[0], linewidth=0.3, alpha=0.2)
    ax.plot3D(np.full(rows, row + 0.5), np.arange(rows) + 0.5, game[row, :, 1],
              color=colors[1], linewidth=1, alpha=0.5)
    for col in range(cols):
        if game[row, col, 1] == br_of_col_player_for_this_row:
            ax.scatter(row + 0.5, col + 0.5, game[row, col, 1], color=colors[1], s=200, marker='*',
                       linewidth=.5, alpha=0.5)
        else:
            ax.scatter(row+0.5, col+0.5, game[row, col, 1], color=colors[1], s=30, marker='o',
                       alpha=0.5)

nth_nash = 0
for col in range(cols):
    br_of_row_player_at_rows = get_indices(np.max(game[:, col, 0]), game[:, col, 0])
    for row in br_of_row_player_at_rows:
        br_of_col_player_at_cols = get_indices(np.max(game[row, :, 1]), game[row, :, 1])
        if col in br_of_col_player_at_cols:
            nth_nash += 1
            ax.plot3D([row+0.5, row+0.5], [col+0.5, col+0.5],
                      [np.min(game) - (np.max(game) - np.min(game)) * 0.2,
                       np.max(game) + (np.max(game) - np.min(game)) * 0.2],
                      linewidth=3, label=f'PSNE # {nth_nash}',
                      color=f'#{str(hex(np.random.randint(16, 240))[2:])}'
                            f'{str(hex(np.random.randint(16, 240))[2:])}'
                            f'{str(hex(np.random.randint(16, 240))[2:])}')

ax.view_init(elev=45, azim=-30, roll=0)
ax.legend(loc='upper left', fontsize=8)
ax.set_title(f"Duopoly: {game_name}")
ax.set_xlabel(f'{players[0]}', fontsize=10)
ax.set_ylabel(f'{players[1]}', fontsize=10)
ax.set_zlabel(f'{payoff_quantity_name}', fontsize=10)
ax.set_xlim(0, rows)
ax.set_ylim(0, cols)
ax.set_zlim(np.min(game) - (np.max(game) - np.min(game)) * 0.2,
            np.max(game) + (np.max(game) - np.min(game)) * 0.2)
ax.set_xticks(np.arange(rows)+0.5)
ax.set_xticklabels(strategies_row, fontsize=8)
ax.set_yticks(np.arange(cols)+0.5)
ax.set_yticklabels(strategies_col, fontsize=8)
z_ticks = np.round(np.linspace(np.min(game) - (np.max(game) - np.min(game)) * 0.2,
                               np.max(game) + (np.max(game) - np.min(game)) * 0.2,
                               8))
ax.set_zticks(z_ticks)
ax.set_zticklabels(z_ticks, fontsize=8)
plt.savefig(f'./image/16VNash-{game_name}.png')
plt.show()
