import numpy as np
from matplotlib import pyplot as plt

# Mean and Median
# Mean = Sum of all elements divided by number of elements
# Median = Middle element of the array when the array is sorted and length is ODD number
# Median = average of 2 Middle elements when the array is sorted and length is EVEN number
# smallArray1 = np.array([2, 2, 4, 4, 5, 6, 7, 8, 9, 10])
# print(f"smallArray1:: Length=", len(smallArray1), " Mean=", np.mean(smallArray1), " Median=", np.median(smallArray1))
# smallArray2 = np.array([2, 2, 4, 4, 5, 6, 7, 8, 9])
# print(f"smallArray2:: Length=", len(smallArray2), " Mean=", np.mean(smallArray2), " Median=", np.median(smallArray2))

plt.style.use('fivethirtyeight')
mu, sigma = 160.95, 5.59
heights = np.random.normal(mu, sigma, 10000)
bins = np.arange(68.58, 182.88, 2.54)
meanHeight = np.mean(heights)
medianHeight = np.median(heights)

# Mean and Median are sensitive to outliers
# Mean is affected by outliers than Median
# Wrongly Added Outliers: 100 6 yr old girls with mean height = 102.5 & SD = 4.3 cm
heightsWithOutlier = np.append(heights, np.random.normal(102.5, 4.3, 1000))
meanHeightWithOutlier = np.mean(heightsWithOutlier)
medianHeightWithOutlier = np.median(heightsWithOutlier)

plt.hist(heightsWithOutlier, bins=bins, edgecolor='black', color='pink', linewidth=1)

plt.axvline(medianHeight, color='green', linestyle='-', linewidth=1, label='Median Height')
plt.axvline(meanHeight, color='blue', linestyle='--', linewidth=1, label='Mean Height')

plt.axvline(meanHeightWithOutlier, color='red', linestyle='--', linewidth=1, label='Mean Affected by Outlier')
plt.axvline(medianHeightWithOutlier, color='k', linestyle='--', linewidth=1, label='Median Unaffected by Outlier')

plt.title("Heights of 18YO Girls")
plt.xlabel("Height in cm")
plt.ylabel("Number of Girls")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/10MeanVsMedian.png')
plt.show()