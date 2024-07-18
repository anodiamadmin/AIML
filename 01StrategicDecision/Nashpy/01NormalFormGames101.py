import nashpy as nash
print(f'Nash Version = {nash.__version__}')

print('\n----- Basic Example -----')
A = [[31, 11], [10, 12]]
B = [[2, 1], [0, 3]]
basic_game = nash.Game(A, B)
print(f'Basic game:\n{basic_game}')
for nash_equilibrium in next(basic_game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(f'nash_equilibrium = {nash_equilibrium}')
        print(f'len([nash_equilibrium]) = {len([nash_equilibrium])}')

print('\n----- Battle of Sexes -----')
A = [[10, 2], [0, 5]]
B = [[5, 2], [0, 10]]
battle_of_sexes_game = nash.Game(A, B)
print(f'Battle of Sexes:\n{battle_of_sexes_game}')
for nash_equilibrium in next(battle_of_sexes_game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(f'nash_equilibrium = {nash_equilibrium}')
        print(f'len([nash_equilibrium]) = {len([nash_equilibrium])}')

print('\n----- Prisoner\'s Dilemma -----')
A = [[-3, 0], [-4, -1]]
B = [[-3, -4], [0, -1]]
prisoners_dilemma_game = nash.Game(A, B)
print(f'Prisoner\'s Dilemma:\n{prisoners_dilemma_game}')
for nash_equilibrium in next(prisoners_dilemma_game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(f'nash_equilibrium = {nash_equilibrium}')
        print(f'len([nash_equilibrium]) = {len([nash_equilibrium])}')

print('\n----- Matching Pennies Game -----')
A = [[1, -1], [-1, 1]]
B = [[-1, 1], [1, -1]]
matching_pennies_game = nash.Game(A, B)
print(f'Matching Pennies Game:\n{matching_pennies_game}')
for nash_equilibrium in next(matching_pennies_game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(f'nash_equilibrium = {nash_equilibrium}')
        print(f'len([nash_equilibrium]) = {len([nash_equilibrium])}')

print('\n----- Single Matrix Matching Pennies Game -----')
A = [[5, 6], [3, 4]]
single_matrix_matching_pennies_game = nash.Game(A)
print(f'Single Matrix Matching Pennies Game:\n{single_matrix_matching_pennies_game}')
for nash_equilibrium in next(single_matrix_matching_pennies_game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(f'nash_equilibrium = {nash_equilibrium}')
        print(f'len([nash_equilibrium]) = {len([nash_equilibrium])}')

print('\n----- Market Share for Low Price Game -----')
print(f'Firm1 & Firm2 sales identical apples at $10 or $5 to 100 customers.'
      f'80 customers care for low unit sale price, 20 are oblivious to price.')
F1 = [[500, 100], [450, 250]]
F2 = [[500, 450], [100, 250]]
market_share_low_price_game = nash.Game(F1, F2)
print(f'Market Share for Low Price Game:\n{market_share_low_price_game}')
for nash_equilibrium in next(market_share_low_price_game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(f'nash_equilibrium = {nash_equilibrium}')
        print(f'len([nash_equilibrium]) = {len([nash_equilibrium])}')

print('\n----- Bertrand Competition Game -----')
print(f'Firm1 & Firm2 sales identical apples @ Cost Price = $5/ apple or @ $10/ apple.'
      f'100 customers, all care for low unit sale price.')
F1 = [[250, 500], [0, 500]]
F2 = [[250, 0], [500, 500]]
bertrand_competition_game = nash.Game(F1, F2)
print(f'Bertrand Competition Game:\n{bertrand_competition_game}')
for nash_equilibrium in next(bertrand_competition_game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(f'nash_equilibrium = {nash_equilibrium}')
        print(f'len([nash_equilibrium]) = {len([nash_equilibrium])}')
