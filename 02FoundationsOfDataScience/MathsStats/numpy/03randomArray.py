import numpy as np
import matplotlib.pyplot as plt
np.random.seed(41)
plt.style.use('ggplot')

fig, ax = plt.subplots(3, 3, sharex=False, sharey=False)
fig.set_figheight(7)
fig.set_figwidth(10)

sample_size = 2000

# randint()
randint_arr = np.random.randint(0, 10, sample_size)
ax[0][0].hist(randint_arr, bins=np.arange(-1, 11, 1), edgecolor='black', color='pink', linewidth=1)
ax[0][0].set_title("randint()")
ax[0][0].set_xlabel("Values-->")
ax[0][0].set_ylabel("Frequency-->")
ax[0][0].set_xlim(-1, 11)
ax[0][0].set_ylim(-1, 600)
ax[0][0].set_xticks(ticks=range(-1, 11, 1))

# rand()
rand_arr = np.random.rand(sample_size)
ax[0][1].hist(rand_arr, bins=np.linspace(-.1, 1.1, 13), edgecolor='black',
              color='#ccccff', linewidth=1)
ax[0][1].set_title("rand()")
ax[0][1].set_xlim(-.1, 1.1)
ax[0][1].set_ylim(-1, 600)

# ranf()
ranf_arr = np.random.ranf(sample_size)
ax[0][2].hist(ranf_arr, bins=np.linspace(-.1, 1.1, 13), edgecolor='black',
              color='#ffddbb', linewidth=1)
ax[0][2].set_title("ranf()")
ax[0][2].set_xlim(-.1, 1.1)
ax[0][2].set_ylim(-1, 600)

# poisson(2)
print(f'POISSON Prob dist of number of events in fixed interval of time. Given a constant mean'
      f'\nA call-center receives λ = 3 calls per min (mean). Calls are independent (receiving '
      f'one does not\nchange the probability of when the next one will arrive).'
      f'\nThe number k of calls received during any minute has a Poisson probability distribution.'
      f'\nReceiving k = 1 to 4 calls then has a probability of about 0.77, while receiving 0 or at '
      f'least 5 calls has a probability of about 0.23.'
      f'\nAnother example: number of radioactive decay events during a fixed observation period')
psn2_arr = np.random.poisson(2, sample_size)
ax[1][0].hist(psn2_arr, bins=np.linspace(-1, 21, 23), edgecolor='black',
              color='#ccccff', linewidth=1)
ax[1][0].set_title("poisson(2)")
ax[1][0].set_xlim(-1, 21)
ax[1][0].set_ylim(-1, 600)

# poisson(5)
psn5_arr = np.random.poisson(5, sample_size)
ax[1][1].hist(psn5_arr, bins=np.linspace(-1, 21, 23), edgecolor='black',
              color='#ccffcc', linewidth=1)
ax[1][1].set_title("poisson(5)")
ax[1][1].set_xlim(-1, 21)
ax[1][1].set_ylim(-1, 600)

# poisson(9)
psn9_arr = np.random.poisson(9, sample_size)
ax[1][2].hist(psn9_arr, bins=np.linspace(-1, 21, 23), edgecolor='black',
              color='#ffcccc', linewidth=1)
ax[1][2].set_title("poisson(9)")
ax[1][2].set_xlim(-1, 21)
ax[1][2].set_ylim(-1, 600)

# randn()
randn_arr = np.random.randn(sample_size)
ax[2][0].hist(randn_arr, bins=np.linspace(-4, 4, 81), edgecolor='black',
              color='#ffddbb', linewidth=1)
ax[2][0].set_title("randn()")
ax[2][0].set_xlim(-4, 4)
ax[2][0].set_ylim(-1, 150)

# normal()
norm_arr = np.random.normal(10, 3, sample_size)
ax[2][1].hist(norm_arr, bins=np.linspace(-2, 22, 81), edgecolor='black',
              color='#ffddbb', linewidth=1)
ax[2][1].set_title("normal()")
ax[2][1].set_xlim(-2, 22)
ax[2][1].set_ylim(-1, 150)

# uniform()
uniform_arr = np.random.uniform(-5, 5, sample_size)
ax[2][2].hist(uniform_arr, bins=np.linspace(-5.1, 5.1, 103), edgecolor='black',
              color='#ffddbb', linewidth=1)
ax[2][2].set_title("uniform()")
ax[2][2].set_xlim(-5.5, 5.5)
ax[2][2].set_ylim(-1, 50)

plt.grid(True)
plt.tight_layout()
plt.show()
