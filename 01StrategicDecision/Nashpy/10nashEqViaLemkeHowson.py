import nashpy as nash

A = [[3, 1], [0, 2]]
B = [[2, 1], [0, 3]]
battle_of_sexes_game = nash.Game(A, B)
print(f'battle_of_sexes_game:\n{battle_of_sexes_game}')

print(f'\nbattle_of_sexes_game.lemke_howson(initial_dropped_label=0): '
      f'{battle_of_sexes_game.lemke_howson(initial_dropped_label=0)}')

print(f'\nlist(battle_of_sexes_game.lemke_howson_enumeration()):\n'
      f'{list(battle_of_sexes_game.lemke_howson_enumeration())}')

print(f'\nElements of battle_of_sexes_game.lemke_howson_enumeration():')
for i, eq in enumerate(battle_of_sexes_game.lemke_howson_enumeration()):
    print(f'element{i}: {eq}')
