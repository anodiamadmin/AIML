import pandas as pd
import matplotlib.pyplot as plt
from pywaffle import Waffle       # pip install pywaffle

# creation of a dataframe
data = {'phone': ['Xiaomi', 'Samsung', 'Apple', 'Nokia', 'Realme'], 'stock': [44, 12, 8, 5, 3]}
df = pd.DataFrame(data)

# To plot the waffle Chart
fig = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=df.stock,
    labels=list(df.phone),
    colors=['#dfa963', '#a9df63', '#a963df', '#df63a9', '#63dfa9'],
    figsize=(8, 4),
    legend={'loc': 'lower left', 'bbox_to_anchor': (1, 1)},
)

plt.style.use('fivethirtyeight')
plt.title("Market Share of Mobiles")
plt.tight_layout()
plt.savefig('./plots/27WaffleChart.png')
plt.show()