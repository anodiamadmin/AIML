# Random Experiment: outcome cannot be predicted: e.g. Coin flips, Die rolls, Runs scored in next over
# Probability = number of favorable outcomes / total number of possible outcomes
# # # A fraction between 0 and 1 :::: P(DiceFace = 5) E [0, 1] :: DiceFace = 5 is an event or outcome
# Sum(Probability(All_Events)) = 1 ::: integral(P(x).dx) = 1
# Random Variable: mathematical quantity associated with the outcome of a random event/ experiment
# Probability Distribution: mathematical function that maps random variables to probabilities
# Input(Event) ---> Probability Distribution Function ---> Output(Probability of occurrence of the event)
# If the sample size is very large, Frequency Distribution of occurrence will follow the Probability Distribution
# Frequency Distribution VS Probability Distribution VS Likely-hood
# Samples are independent of each other, height of person 1 has no impact on height of person 2

# Experiment: Survey height and eye color of 1000 random people in street.
# PersonID   EyeColor   Height
#  1          Brown      165
#  2          Blue       170
#  3          Hazel      168
#   ...........................
# Q1: What are mean & st div of heights?   mean = 168.5cm  sigma = 7.3cm   Height => Continuous Var - Numeric
# Q2: How many people surveyed per hour?   average = 4 people per hour     survey/hr => Discrete Var - Numeric
# Q3: What % of people have Hazel eyes?    1.4                             EyeColor => Discrete Var - Categorical
# Probability Mass Function (PMF): Probability distribution of discrete random variables

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(41)
plt.style.use('ggplot')
fig1, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(10, 5))

sample_size = 100000               # total number of people surveyed
height_mean = 168.5                # mean height of people observed = 168.5cm
height_std = 7.3                   # standard deviation of height of people observed = 7.3cm
mean_survey_per_hr = 2.8           # average number of people surveyed per hour

height_dist_array = np.random.normal(height_mean, height_std, sample_size)
height_bins_array = np.linspace(np.floor(height_mean-4*height_std).astype(int),
                                np.ceil(height_mean+4*height_std).astype(int),
                                np.ceil(8*height_std).astype(int)+1)
ax[0].hist(height_dist_array, bins=height_bins_array, edgecolor='blue', color='#ffddbb', linewidth=0.5,)
# integral[156:165](P(x).dx) = area under curve between 156 and 165
ax[0].hist(height_dist_array, color='purple', alpha=0.5,
           bins=np.linspace(156, 165, 10), label='Prob of 156-165cm height')
ax[0].set_title("Height: Normal (Gaussian) PDF - Dist")
ax[0].set_xlabel('Height (cm)')
ax[0].set_ylabel(f'Probability = Number of people / {sample_size}')
ax[0].set_xlim(height_mean-4*height_std, height_mean+4*height_std)
ax[0].set_ylim(-.002*sample_size, 0.07*sample_size)
ax[0].set_xticks(height_bins_array[1:61:10])
ax[0].axvline(height_mean, color='green', linestyle='--', linewidth=1, label='Mean')
ax[0].axvline(height_mean+height_std, color='green', linestyle=':', linewidth=1, label='St-div')
ax[0].axvline(height_mean-height_std, color='green', linestyle=':', linewidth=1)
ax[0].legend(loc='upper right')

survey_per_hr_array = np.random.poisson(mean_survey_per_hr, sample_size)
ax[1].hist(survey_per_hr_array, bins=np.linspace(0, 10, 11),
           edgecolor='black', color='#ccccff', linewidth=1)
ax[1].set_title(f"Survey / Hr: Poisson's PMF Dist: Mean={mean_survey_per_hr}")
ax[1].set_xlabel('People surveyed per hour')
ax[1].set_ylabel(f'Probability = Number of surveys / {sample_size}')
ax[1].set_xlim(0, 11)
ax[1].set_ylim(-.01*sample_size, 0.3*sample_size)

plt.grid(True)
plt.tight_layout()
plt.show()
