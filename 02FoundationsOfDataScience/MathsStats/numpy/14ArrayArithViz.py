from math import sin
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(41)
plt.style.use('ggplot')

fig, ax = plt.subplots(1, 3, sharex=False, sharey=False)
fig.set_figheight(6)
fig.set_figwidth(10)

x_arry = np.linspace(0, 10, 101)[1:]  # to avoid division by zero, we start from 1, not 0
CONST = 4
arrConst = np.full(len(x_arry), CONST)
yEqualsXarry = np.copy(x_arry)
sinArry = np.sin(x_arry)
rndArry = np.random.normal(0, 1, len(x_arry))
intYary = np.linspace(2, 12, 11)
intXary = np.linspace(0, 10, 11)
'''
+, -, *, /, **, %, 1/
'''
ax[0].plot(x_arry, arrConst, label='CONST=3', linestyle=':', color='#ff9922', linewidth=1)
ax[0].plot(x_arry, yEqualsXarry, label='y=x', linestyle=':', color='#ff2299', linewidth=1)
ax[0].plot(x_arry, sinArry, label='y=sin(x)', linestyle=':', color='#2299ff', linewidth=1)
ax[0].plot(x_arry, rndArry, label='rand', linestyle=':', color='green', linewidth=1)
ax[0].plot(intXary, intYary, label='Int[x]', marker='*', linestyle=':', color='#9922ff', linewidth=1)
ax[0].legend(loc='upper left')
ax[0].set_title("CONST, [x], sin[x], rand[]")
# ax[0].set_xlabel("X-->")
# ax[0].set_ylabel("Y-->")
ax[0].set_xlim(-1, 11)
ax[0].set_ylim(-4, 16)
ax[0].set_xticks(ticks=range(-1, 11, 1))
'''
"Array +,-,*,/,**,%,1/ CONSTANT"
'''
sinArrAddConst = sinArry + CONST
noiseArrDivConst = rndArry / CONST
inverseX = CONST / yEqualsXarry
modAry = intYary % CONST
ax[1].plot(x_arry, sinArrAddConst, label='sin[x]+CONST', linestyle=':', color='#ff9922', linewidth=1)
ax[1].plot(x_arry, noiseArrDivConst, label='rand[]/CONST', linestyle=':', color='#ff2299', linewidth=1)
ax[1].plot(x_arry, inverseX, label='y=1/[x]', linestyle=':', color='#2299ff', linewidth=1)
ax[1].plot(intXary, modAry, label='[x]%CONST', marker='*', linestyle=':', color='#9922ff', linewidth=1)
ax[1].legend(loc='upper right')
ax[1].set_title("Array +,-,*,/,**,%,1/ CONSTANT")
# ax[1].set_xlabel("X-->")
# ax[1].set_ylabel("Y-->")
ax[1].set_xlim(-1, 11)
ax[1].set_ylim(-4, 16)
ax[1].set_xticks(ticks=range(-1, 11, 1))
'''
Array +, -, *, /, **, %, 1/ Array
'''
modulator = np.sin(x_arry*CONST*2)*2 + 3*CONST
audio = sinArry + 2*CONST
am = np.sin(x_arry*CONST)*2 * sinArry + CONST
noisyAm = np.sin(x_arry*CONST)*2 * sinArry + noiseArrDivConst
ax[2].plot(x_arry, modulator, label='modulator', linestyle=':', color='#9922ff', linewidth=1)
ax[2].plot(x_arry, audio, label='audio', linestyle=':', color='#ff9922', linewidth=1)
ax[2].plot(x_arry, am, label='am', linestyle=':', color='green', linewidth=1)
ax[2].plot(x_arry, noisyAm, label='noisy am', linestyle=':', color='red', linewidth=1)
ax[2].legend(loc='upper left')
ax[2].set_title("Array +,-,*,/,**,%,1/ Array")
# ax[2].set_xlabel("X-->")
# ax[2].set_ylabel("Y-->")
ax[2].set_xlim(-1, 11)
ax[2].set_ylim(-4, 16)
ax[2].set_xticks(ticks=range(-1, 11, 1))

plt.grid(True)
plt.tight_layout()
plt.show()