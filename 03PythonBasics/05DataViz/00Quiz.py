# Ans: 1.d, 2.c, 3.b, 4.c, 5.b, 6.a, 7.c, 8.d, 9.d, 10.a,
# 11.D, 12.A, 13.A, 14.A, 15.D, 16.D, 17.D, 18.A, 19.C, 20.D

# Q01: Which one of the following is true of the relationship between Matplotlib, pandas, and seaborn?
# a. Matplotlib and pandas are both built on top of seaborn.
# b. Matplotlib is built on top of both pandas and seaborn.
# c. pandas, but not seaborn, is built on top of Matplotlib.
# d. Both pandas and seaborn are built on top of Matplotlib.
# Ans: d

# Q02: Which one of the following is true of the relationship between figures and axes?
# a. A figure need not have an axes in order to plot on it.
# b. A figure can only have one axes.
# c. A figure can have multiple axes and those axes only belong to that figure.
# d. A figure can have multiple axes and those axes can belong to other figures too.
# Ams: C

# Q03: If you call the plot method on a panda's series, with no arguments, what will pandas create?
# A. If the values are numerical, it will create a box plot of the values to show a summary of their
# distribution. Otherwise, it will create a frequency histogram of the values.
# B. If the values are numerical, it will create a line plot of the values. Otherwise, it will create
# a frequency histogram of the values.
# C. If the values are numerical, it will create a box plot of the values to show a summary of their
# distribution. Otherwise, it will raise an error.
# D. If the values are numerical, it will create a line plot of the values. Otherwise, it will raise an error.
# Ans: B

# Q04: If you call the plot method on a panda's data frame, with no arguments, what will pandas create?
# A. For each numerical column it will create a line plot of the values. For each other column it will create a
# frequency histogram of the values. All plots will be on the same axes.
# B. For each numerical column it will create a line plot of the values. For each other column it will create a
# frequency histogram of the values. The line plots will all be together on one axes. The histograms will all
# be together on a separate axes.
# C. For each numerical column it will create a line plot of the values. All other columns will be ignored. All
# plots will be on the same axes.
# D. For each numerical column it will create a line plot of the values. All other columns will be ignored. Each
# plot will be on separate axes.
# Ans: c
# import pandas as pd
# import matplotlib.pyplot as plt
# # Example DataFrame
# df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': ['a', 'b', 'c', 'd', 'e']})
# # Default plotting
# df.plot()
# plt.show()

# Q05: Which one of the following is not a valid value for the kind parameter of pandas' plot method?
# A. "density"
# B. "hbar"
# C. "hexbin"
# D. All of the above
# Ans: B

# Q06: Suppose you have a series with 30 values, showing a company's total sales revenue for each of the last 30 years.
# Which one of the following kinds of plot would best show how those totals have been evolving over time?
# A. line plot
# B. box plot
# C. histogram
# D. pair plot
# Ans: A

# Q07: Suppose you have a data frame with 300 rows and 2 columns. Each row represents a cow that you have sold
# from your farm. The first column is called "Weight" and contains the cow's weight at sale; the second column
# is called "Price" and contains the cow's sale price. You are interested in the strength of the relationship
# between these two values. Which one of the following plots would best help you visualize it?
# A. unstacked bar plot of the two columns on one axes.
# B. stacked bar plot of the two columns on one axes.
# C. scatter plot of one column versus the other.
# D. box plot of each column on separate axes.
# Ans: C

# Q08: In which one of the following situations would it be good to use a hexbin plot?
# A. You have a series of numeric values and would like to show their frequency distribution, but many of the
# values are close together.
# B. You have a series of numeric values and would like to compare their relative sizes.
# C. You have a data frame with two columns and would like to show how the values in one column are related to
# the values in the other column, but the values are not numerical.
# D. You have a data frame with two numerical columns and would like to show how the values in one column are
# related to the values in the other column, but the points of a scatter graph would overlap too much.
# Ans: D

# Q09: Suppose you're creating a figure using the following statement:
# df.plot(
#     kind = 'scatter',
# )
# If you want six subplots, arranged in three rows of two, which one of the following options should you add?
# A. subplots=[2, 3]
# B. subplots=[3, 2]
# C. subplots=True, layout=[2, 3]
# D. subplots=True, layout=[3, 2]
# Ans: D
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # Example DataFrame
# df = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10), 'C': np.random.rand(10),
#                    'D': np.random.rand(10), 'E': np.random.rand(10), 'F': np.random.rand(10)})
# # Creating a scatter plot with subplots
# df.plot(kind='line', subplots=True, layout=[3, 2], figsize=(10, 8))
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

# Q10: Suppose you've created a figure with subplots arranged in two rows of two. If you want to use 'Jan',
# 'Feb','Mar', and 'Apr' as titles for the subplots, which one of the following options should you add?
# A. title = ['Jan', 'Feb', 'Mar', 'Apr']
# B. titles = ['Jan', 'Feb', 'Mar', 'Apr']
# C. title = [['Jan', 'Feb'], ['Mar', 'Apr']]
# D. titles = [['Jan', 'Feb'], ['Mar', 'Apr']]
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # Example DataFrame with 4 columns
# df = pd.DataFrame(
#     {'Jan': np.random.rand(10), 'Feb': np.random.rand(10), 'Mar': np.random.rand(10), 'Apr': np.random.rand(10)})
# # Creating a figure with subplots
# df.plot(kind='line', subplots=True, layout=(2, 2), figsize=(8, 6), title=['Jan', 'Feb', 'Mar', 'Apr'])
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()
# Ans: A

# Q11: Suppose you're creating a figure with subplots arranged in three rows of two.
# If you want the subplots to share both their x and y-axis which one of the following options
# should you add?
# A. share = True
# B. share = [True, True]
# C. [sharex, sharey] = [True, True]
# D. Sharex=True, sharey=True
# Ans: D
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # Example DataFrame
# df = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10), 'C': np.random.rand(10),
#                    'D': np.random.rand(10), 'E': np.random.rand(10), 'F': np.random.rand(10)})
# # Creating subplots with shared axes
# df.plot(subplots=True, layout=(3, 2), sharex=True, sharey=True, figsize=(8, 10))
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

# Q12: Suppose you're creating a plot using the following statement:
# df.plot(
#     kind = 'bar',
# )
# If you want the x-axis tick labels to go from 0 to 2 in steps of 0.2,
# which one of the following options could you add?
# A. xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
# B. xticks = range(0, 2, 0.2)
# C. xticks = [x/10 for x in range(0, 20)]
# D. You could use any of the above.
# Ans: A
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # Example DataFrame
# df = pd.DataFrame({'Category': ['A', 'B', 'C', 'D'], 'Values': [10, 15, 7, 12]})
# # Plot a bar chart with custom x-axis ticks
# df.plot(kind='bar', xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# # df.plot(kind='bar', xticks=np.arange(0, 2, 0.2))
# # df.plot(kind='bar', xticks=[x/10 for x in range(0, 20)])
# plt.show()

# Q13: Suppose you have a data frame with two numeric columns and you are using its plot method to
# create a line plot of each column. Suppose you want the first line to be dotted and the second
# line to be dashed. Which one of the following options should you add to your call to plot?
# A. style = [':', '--']
# B. style = ['--', ':']
# C. linestyte = [':', '--']
# D. linestyle = ['--', ':']
# import pandas as pd
# import matplotlib.pyplot as plt
# # Example DataFrame
# df = pd.DataFrame({'Column1': [1, 2, 3, 4, 5], 'Column2': [5, 4, 3, 2, 1]})
# # Plot with custom line styles
# df.plot(style=[':', '--'], title='Custom Line Styles')
# plt.show()
# Ans: A

# Q14: Suppose you are calling the plot method of a data frame to produce a plot,
# and you would like to show gridlines on the plot. Which one of the following options should you add?
# A. grid = True
# B. gridlines = True
# C. grid = 'On'
# D. hidegrid = False
# Ans: A

# Q15: Suppose you have data frame df with three columns, labeled "cat", "num1", and "num2".
# The first column contains categories ("A", "B", or "C"), and the second and third columns contain
# numerical values. You are creating a scatter plot of the second two columns, but you want to color
# the points according to the category in the first column. You try the following statement, but it
# generates an error:
# df.plot(kind = 'scatter', x = 'num1', y = 'num2', c = 'cat',)
# Why does it generate an error?
# A. You cannot adjust the color of the points in a scatter plot.
# B. The parameter name is C, not c.
# C. The parameter name is hue, not c.
# D. The values in the "cat" column cannot be interpreted as colors.
# Ans: D

# Q16: Suppose you decide to use a seaborn relplot for the task in the previous question.
# To color the points according to the category in the first column, which option
# should you add to your call of relplot?
# A. c = 'cat'
# B. C = 'cat'
# C. color = 'cat'
# D. hue = 'cat'
# Ans: D
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# # Example DataFrame
# df = pd.DataFrame({'cat': ['A', 'B', 'A', 'C', 'B', 'C'], 'X': [1, 2, 3, 4, 5, 6], 'Y': [6, 5, 4, 3, 2, 1]})
# # Seaborn relplot with points colored by 'cat'
# sns.relplot(data=df, x='X', y='Y', hue='cat', kind='scatter')
# # sns.relplot(data=df, x='X', y='Y', C='cat', kind='scatter')
# # sns.relplot(data=df, x='X', y='Y', c='cat', kind='scatter')
# # sns.relplot(data=df, x='X', y='Y', color='cat', kind='scatter')
# plt.show()

# Q17: Which one of the following is true of panda's scatter_matrix function and its scatter matrices?
# A. scatter_matrix is not a method of a series or data frame but of pandas.plotting
# B. To be more informative, scatter_matrix creates histogram plots down the main diagonal of a
# scatter matrix rather than scatter plots.
# C. The scatter plots above the main diagonal of a scatter matrix are just a reflection of the
# scatter plots below the main diagonal, so do not carry much new information.
# D. All of the above are true.
# And: D

# Q18: Suppose that s1 and s2 are both series with ten numerical values each,
# and that you have the following two statements in your program:
# s1.plot()
# s2.plot()
# When the first statement is executed a line plot is created on an axes.
# What happens when the second statement is executed?
# A. Another line plot is created on the same axes as the first line plot.
# B. Another line plot is created on the same figure as the first line plot, but on a different axes.
# C. Another line plot is created, but on a different figure from the first line plot.
# D. An error occurs - there needs to be a call to a clearing function before a second plot can be created.
# Ans: A
# import pandas as pd
# import matplotlib.pyplot as plt
# Example Series
# s1 = pd.Series([1, 2, 3, 4, 5])
# s2 = pd.Series([5, 4, 3, 2, 1])
# # Plot the first Series
# s1.plot(label='s1', legend=True)
# # Plot the second Series on the same axes
# s2.plot(label='s2', legend=True)
# plt.title('Line Plots of s1 and s2 on the Same Axes')
# plt.show()

# Q19: Suppose that df1 and df2 are both data frames, each with two columns of ten numerical values,
# and that you have the following two statements in your program:
# df1.plot()
# df2.plot()
# When the first statement is executed two line plots are created on an axes.
# What happens when the second statement is executed?
# A. Another two line plots are created on the same axes as the first two line plots.
# B. Another two line plots are created on the same figure as the first two line plots, but on a different axes.
# C. Another two line plots are created, but on a different figure from the first two line plots.
# D. An error occurs - there needs to be a call to a clearing function before a second plot can be created.
# Ans C
# import pandas as pd
# import matplotlib.pyplot as plt
# # Example DataFrames
# df1 = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
# df2 = pd.DataFrame({'C': [2, 3, 4, 5, 6], 'D': [6, 5, 4, 3, 2]})
# # Plot the first DataFrame
# df1.plot()
# plt.title('Figure 1: df1')
# # Plot the second DataFrame
# df2.plot()
# plt.title('Figure 2: df2')
# plt.show()

# Q20: Suppose you are plotting a data frame using its plot method. You want the figure to be the size of an
# A4 page, which is 21 cm wide and 29.7 cm tall, but with landscape orientation. Which one of the following
# options could you add to your plot call to achieve this?
# A. figsize=[21, 29.7]
# B. figsize=[29.7, 21]
# C. figsize=[8.3, 11.7]
# D. figsize=[11.7, 8.3]
# Ans: D
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # Example DataFrame
# df = pd.DataFrame({
#     'A': np.random.rand(10),
#     'B': np.random.rand(10)
# })
# # Plot with A4 landscape size
# df.plot(
#     kind='line',
#     figsize=[11.7, 8.3],  # A4 in landscape orientation
#     title='A4 Landscape Plot'
# )
# plt.show()
