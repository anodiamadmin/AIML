import numpy as np
from matplotlib import pyplot as plt

# game = [[(31, 2), (11, 1)],
#         [(10, 0), (12, 3)]]
game = [[(10, 5), (2, 2)],      # Battle of sexes
        [(0, 0), (5, 10)]]

colors = ['red', 'blue']
game_name = 'Basic Game'
players = ['Row Player', 'Column Player']
strategies = ['Str-1', 'Str-2']


def get_indices(element, lst):
    return [i for i, x in enumerate(lst) if x == element]


game = np.array(game)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
rows = game.shape[0]
cols = game.shape[1]
rsps = game.shape[2]

R, C = np.meshgrid(np.linspace(0, rows, 101),
                   np.linspace(np.min(game) - (np.max(game) - np.min(game)) * 0.2,
                               np.max(game) + (np.max(game) - np.min(game)) * 0.2, 101))
E = np.zeros((101, 101))
for col in range(cols):
    br_of_row_player_for_this_col = np.max(game[col, :, 0])
    ax.plot(E+col+0.5, R, C, color=colors[1], linewidth=0.3, alpha=0.3)
    for row in range(rows):
        ax.plot3D(np.full(cols, row + 0.5), np.arange(cols) + 0.5, game[row, :, 0],
                  color=colors[0], linewidth=1)
        if game[row, col, 0] != br_of_row_player_for_this_col:
            ax.scatter(row+0.5, col+0.5, game[row, col, 0], color=colors[0], s=30, marker='o')
        else:
            ax.scatter(row+0.5, col+0.5, game[row, col, 0], color=colors[0], s=200, marker='*',
                       edgecolors=colors[1], linewidth=.5)

R, C = np.meshgrid(np.linspace(0, cols, 101),
                   np.linspace(np.min(game) - (np.max(game) - np.min(game)) * 0.2,
                               np.max(game) + (np.max(game) - np.min(game)) * 0.2, 101))
E = np.zeros((101, 101))
for row in range(rows):
    br_of_col_player_for_this_row = np.max(game[:, row, 1])
    ax.plot(R, E+row+0.5, C, color=colors[0], linewidth=0.3, alpha=0.3)
    for col in range(cols):
        ax.plot3D(np.arange(rows) + 0.5, np.full(rows, col + 0.5), game[:, col, 1],
                  color=colors[1], linewidth=1)
        if game[col, row, 1] != br_of_col_player_for_this_row:
            ax.scatter(col+0.5, row+0.5, game[col, row, 1], color=colors[1], s=30, marker='o')
        else:
            ax.scatter(col+0.5, row+0.5, game[col, row, 1], color=colors[1], s=200, marker='*',
                       edgecolors=colors[0], linewidth=.5)

nth_nash = 0
for col in range(cols):
    br_of_row_player_for_this_col = np.max(game[col, :, 0])
    br_of_row_player_at_rows = get_indices(np.max(game[col, :, 0]), game[col, :, 0])
    print(f'Row Player A: {br_of_row_player_for_this_col} at rows {br_of_row_player_at_rows}')
    for row in br_of_row_player_at_rows:
        br_of_col_player_for_this_row = np.max(game[:, row, 1])
        br_of_col_player_at_cols = get_indices(np.max(game[:, row, 1]), game[:, row, 1])
        if col in br_of_col_player_at_cols:
            nth_nash += 1
            print(f'Pure Nash Equilibrium Found at: ({row+1}, {col+1})')
            ax.plot3D([row+0.5, row+0.5], [col+0.5, col+0.5],
                      [np.min(game) - (np.max(game) - np.min(game)) * 0.2,
                       np.max(game) + (np.max(game) - np.min(game)) * 0.2],
                      linewidth=3, label=f'PSNE # {nth_nash}',
                      color=f'#{str(hex(np.random.randint(0, 16777215)))[2:]}')

ax.view_init(elev=45, azim=-30, roll=0)
ax.legend(loc='upper left', fontsize=10)
ax.set_title(f"Duopoly: {game_name}")
ax.set_xlabel(f'{players[0]}', fontsize=10)
ax.set_ylabel(f'{players[1]}', fontsize=10)
ax.set_zlabel("Payoff", fontsize=10)
ax.set_xlim(0, rows)
ax.set_ylim(0, cols)
ax.set_zlim(np.min(game) - (np.max(game) - np.min(game)) * 0.2,
            np.max(game) + (np.max(game) - np.min(game)) * 0.2)
ax.set_xticks(np.arange(rows)+0.5)
ax.set_xticklabels(strategies, fontsize=8)
ax.set_yticks(np.arange(cols)+0.5)
ax.set_yticklabels(strategies, fontsize=8)
ax.set_zticklabels(fontsize=8)
plt.savefig(f'./image/16VNash-{game_name}.png')
plt.show()
