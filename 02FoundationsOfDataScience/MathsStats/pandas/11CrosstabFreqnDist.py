import pandas as pd
import numpy as np
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
df_emp = pd.read_excel('./data/11CrosstabFreqnDist.xlsx', sheet_name='11CrosstabFreqnDist')
# print(df_emp)

df_frequency_dist_handedness = (df_emp[['EmpId', 'Right_Left_Handed']]
                                .groupby(['Right_Left_Handed']).count())
print(df_frequency_dist_handedness)

df_frequency_dist_dept = df_emp[['EmpId', 'Department']].groupby(['Department']).count()
print(df_frequency_dist_dept)

# Cross Tab = Contingency Table = Frequency Distribution
# pandas function pd.crosstab(), not dataframe function df.crosstab()
df_cross_tab_dept_handedness = pd.crosstab(df_emp['Department'], df_emp['Right_Left_Handed'],
                                   margins=True)    # margins = False by default, no sum row/ col
print(df_cross_tab_dept_handedness)

df_cross_tab_sex_dept = pd.crosstab(df_emp['Sex'], df_emp['Department'])
print(df_cross_tab_sex_dept)

df_cross_tab_sex_dept_handedness = pd.crosstab(df_emp.Sex,
                                               [df_emp.Department, df_emp.Right_Left_Handed],
                                               margins=True)
print(f'\n{df_cross_tab_sex_dept_handedness}')

df_cross_tab_sex_dept_handedness = pd.crosstab([df_emp.Sex, df_emp.Department],
                                               df_emp.Right_Left_Handed, margins=True)
print(f'\n{df_cross_tab_sex_dept_handedness}')

df_cross_tab_sex_handedness_avg_age = pd.crosstab(df_emp.Sex, df_emp.Right_Left_Handed,
                                                  values=df_emp.Age, aggfunc=np.average)
print(f'\n{df_cross_tab_sex_handedness_avg_age}')
