# python.exe -m pip install --upgrade pip
# pip install --upgrade distribute
# pip install --upgrade setuptools
# pip install axelrod
# pip install --upgrade axelrod
# pip install --force-reinstall -U axelrod
# pip install dask
# pip install dask[dataframe]

import axelrod as axl
import matplotlib.pyplot as plt

print(f'axl version = {axl.__version__}\n'
      f'len(axl.strategies) = {len(axl.strategies)}')

player1 = axl.TitForTat()
player2 = axl.Alternator()
match = axl.Match((player1, player2), turns=6)
print(match.play())
print(f'match.final_score_per_turn() = {match.final_score_per_turn()}')

players = [s() for s in axl.demo_strategies]
print(f'axl.demo_strategies = {axl.demo_strategies}')
print(f'players = {players}')
tournament = axl.Tournament(players, turns=10, repetitions=5)
results = tournament.play()
print(f'results = {results}')

plot = axl.Plot(results)
plot.boxplot()
plt.show()
