# import pandas as pd

# Ans: 1.C, 2.A, 3.C, 4.D, 5.D, 6.D, 7.C, 8.A, 9.C, 10.D, 11.A, 12.D, 13.D, 14.D, 15.C,
# 16.D, 17.C, 18.B, 19.D, 20.B

# Q01: Suppose we create a series using the following code:
# s = pd.Series({'a': 4, 'e': 3, 'i': 3, 'o': 2, 'u': 3})
# Which one of the following is true?
# a> The values of s are 0, 1, 2, 3, and 4.
# b> The labels of the values of s are 0, 1, 2, 3, and 4.
# c> The labels of the values of s are 'a', 'e', 'i', 'o', and 'u'.
# d> The positions of the values of s are 'a', 'e', 'i', 'o', and 'u'.
# Ans: c

# Q02: Suppose that s is a series with ten elements whose labels are strings.
# Which one of the following expressions could you use to select the first five elements?
# a> s.iloc[0:5]
# b> s.iloc[0:6]
# c> s.loc[0:5]
# d> s.loc[0:6]
# Ans: a

# Q03: Suppose that s is a series, two of whose elements have the labels 'a' and 'b'.
# Which one of the following expressions could you use to select those two elements?
# a> s.loc['a', 'b']
# b> s.loc('a', 'b')
# c> s.loc[['a', 'b']]
# d> s.loc(['a', 'b'])
# Ans: c

# Q04: Suppose you want to replace all occurrences of 'x' in a series with 'X',
# and all occurrences of 'y' with 'Y'. Which one of the following statements could you use?
# a> s.replace('x', 'X').replace('y', 'Y')
# b> s.replace(['x', 'y'], ['X', 'Y'])
# c> s.replace({'x': 'X', 'y': 'Y'}))
# d> You could use any of the above.
# Ans: d

# Q05: Suppose that s is a series. Which one the following expressions gives the sum
# of the values in the series?
# a> sum(s)
# b> s.sum()
# c> s.sum(axis=0)
# d> All of the above do.
# Ans: d

# Q06: Suppose that s is a series whose values are strings. Which one the following statements
# could you use to convert each value to upper case?
# a> s.str.upper()
# b> s.apply(str.upper)
# c> s.apply(lambda x: x.upper())
# d> You could use any of the above.
# Ans: d

# Q07: Why does the following piece of code cause an error?
# import pandas as pd
# s1 = pd.Series([1, 1])
# s2 = pd.Series([1, 'a'])
# if (s1 == s2):
#     print("s1 equals s2")
# a> 1 and 'a' cannot be compared for equality
# b> sl and s2 cannot be compared for equality
# c> sl == s2 does not evaluate to a single truth value
# d> None of the above
# Ans: c

# Q08: Suppose we create a data frame using the following statement:
# df = pd.DataFrame({
#     'John': {'A': 12, 'B': 20},
#     'Mary': {'A': 15, 'B': 17},
# })
# print(df)
# What will be the row labels of the resulting data frame?
# a> 'A' and 'B'
# b> 'John' and 'Mary'
# c> 0 and 1
# d> 1 and 2
# Ans: a

# Q09: Suppose we create a data frame using the following statement:
# df = pd.DataFrame([
#     {'A': 12, 'B': 20},
#     {'A': 15, 'B': 17},
# ], index=[1, 2])
# print(df)
# What will be the row labels of the resulting data frame?
# a> 'A' and 'B'
# b> 0 and 1
# c> 1 and 2
# d> It wont have column labels
# Ans: c

# Q10: Suppose that df is a data frame with multiple rows and multiple columns.
# Which one of the following expressions results in a data frame?
# a>  df.iloc[0, :]
# b>  df.iloc[:, 0]
# c>  df.iloc[:, :]
# d>  All of the above do.
# Ans: d

# Q11: Suppose that df is a data frame that has a column labelled 'A'.
# Which one of the following statements could you use to remove this column from the data frame?
# a> del df['A']
# b> df.drop('A')
# c> df['A'] = None
# d> All of the above do.
# Ans: a
# import pandas as pd
# df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# print(df)  # Original DataFrame
# del df['A']
# df = df.drop('A')
# df['A'] = None
# print(df)  # DataFrame with 'A' column filled with None

# Q12: Suppose that df is a data frame that has three columns labelled 'A', 'B', and 'C', in that order.
# Which one of the following statements could you use to relabel column 'C' to 'D'?
# a> df.columns = ['A', 'B', 'D']
# b> df = df.rename(columns={'C': 'D'})
# c> df.rename(columns={'C': 'D'}, inpIace=True)
# d> You could use any of the above
# Ans: d

# Q13: Suppose that df is a data frame that has a column labelled 'State' that contains some
# missing values (NaN). Which one of the following statements could you use to remove the rows
# that are missing a value in this column?
# a> df.dropna(subset=['State'], inplace=True)
# b> df.dropna(subset=['State'])
# c> df=df[df['State'].notnull()]
# d> You could use any of the above
# Ans: d

# Q14: Suppose that df is a data frame that has a column labelled 'Sales' in which the word 'zero' appears.
# Suppose you want to replace all occurrences of the word 'zero' in this column with the number 0.
# Which one of the following statements could you use?
# a> df=df['Sales'].replace('zero',0)
# b> df['Sales']=pd.replace('zero', 0)
# c> df['Sales'].replace('zero', 0)
# d> df['Sales']=df['Sales'].replace('zero', 0)
# Ans: d

# Q15: Suppose that df is a data frame that has a column labelled 'grade'.
# Which one of the following expressions could you use to select all rows which have the
# value 'CR' in this column?
# a> df['grade' == 'CR']
# b> df['grade'] == 'CR'
# c> df[df['grade'] == 'CR']
# d> Any of the above.
# Ans: c

# Q16: Suppose that df is a data frame that has a columns labelled 'Year' and 'Number'.
# Which one of the following expressions could you use to select all rows in which 'Year' is
# either 2010 or 2011 and 'Number' is greater than 100?
#
# a> df[((df['Year'] == 2010) | (df['Year'] == 2011)) & (df['Number'] > 100)]
# b> df[df['Year'].isin([2010, 2011]) & (df['Number'] > 100)]
# c> df[((df['Year'] == 2010) & (df['Number'] > 100)) | ((df['Year'] == 2011) & (df['Number'] > 100))]
# d> Any of the above.
# Ans: d

# Q17: Suppose that df is a data frame that has columns labelled 'A' and 'B'.
# Suppose you want to sort the rows of the data frame, by column 'A' in descending order
# then by column 'B' in ascending order. Which one of the following statements could you use?
#
# A> df = df.sort_values([by='A', ascending=False], [by='B', ascending=True])
# B> df = df.sort_values(by=[['A', False], ['B', True]])
# C> df = df.sort_values(by=['A', 'B'], ascending=[False, True])
# D> None of the above
# Ans: c

# Q18: Suppose that df1 is this data frame:
#      X    Y
# 0  1.0  1.0
# 1  1.0  1.0
# And that df2 is this data frame:
#      X    Z
# 0  1.0  1.0
# 1  1.0  1.0
# Which one of the data frames below does the following expression return?
# df1.append(df2)
# a>
#       X   Y   Z
# 0   1.0 1.0 NaN
# 1   1.0 1.0 NaN
# 0   1.0 NaN 1.0
# 1   1.0 NaN 1.0
# b>
#       X   Y   Z
# 0   1.0 1.0 NaN
# 1   1.0 1.0 NaN
# 2   1.0 NaN 1.0
# 3   1.0 NaN 1.0
# c>
#       X   Y   X   Z
# 0   1.0 1.0 1.0 1.0
# 1   1.0 1.0 1.0 1.0
# d>
#       X   Y   X   Z
# 0   1.0 1.0 NaN NaN
# 1   1.0 1.0 NaN NaN
# 0   NaN NaN 1.0 1.0
# 1   NaN NaN 1.0 1.0
#
# import pandas as pd
# df1 = pd.DataFrame([
#     {'X': 1.0, 'Y': 1.0},
#     {'X': 1.0, 'Y': 1.0},
# ])
# df2 = pd.DataFrame([
#     {'X': 1.0, 'Z': 1.0},
#     {'X': 1.0, 'Z': 1.0},
# ])
# print(df1)
# print(df2)
# print(df1.append(df2, axis=1))
# Ans: b

# Q19: Suppose that df1 is this data frame:
#      0    1
# A  1.0  1.0
# B  1.0  1.0
# And that df2 is this data frame:
#      0    1
# A  1.0  1.0
# C  1.0  1.0
# Which one of the expressions below returns the following data frame?
#      0    1    2    3
# A  1.0  1.0  1.0  1.0
# B  1.0  1.0  NaN  NaN
# C  NaN  NaN  1.0  1.0
# import pandas as pd
# df1 = pd.DataFrame([
#     [1.0, 1.0],
#     [1.0, 1.0],
# ], index=['A', 'B'])
# df2 = pd.DataFrame([
#     [1.0, 1.0],
#     [1.0, 1.0],
# ], index=['A', 'C'])
# # df3 = df1.append(df2, axis=1)
# # df3 = df1.concat(df2, axis=1)
# # df3 = pd.concat([df1, df2], axis=1)
# df3 = pd.concat([df1, df2], axis=1, ignore_index=True)
# print(df1)
# print(df2)
# print(df3)
# a> df1.append(df2, axis=1)
# b> df1.concat(df2, axis=1)
# c> pd.concat([df1, df2], axis=1)
# d> pd.concat([df1, df2], axis=1, ignore_index=True)
# Ans: d

# Q20: Suppose that df1 is a data frame  containing all students in a certain course with a column
# labelled "student_id", and that df2 is another data frame containing the phone number of students
# at UNSW with a column labelled "student_id". Suppose you want to merge df1 and df2 using
# the "student_id" column to generate a list of all students in the course and their phone number,
# making sure that all rows of df1 are included, even if they have no matching row in df2.
# Which one of the following statements should you use?
# a> df1 = df1.merge(df2, on='student_id', how='inner')
# b> df1 = df1.merge(df2, on='student_id', how='left')
# c> df1 = df1.merge(df2, on='student_id', how='right')
# d> df1 = df1.merge(df2, on='student_id', how='outer')
# Ans: b
