import pandas as pd
import os

os.chdir('D:\\anodiam\\AnodiamContent\\AI_Robotics\\Youtubing\\05-DataAnalysis\\ProjExercise\\pandas\\data')
# print(f'$>> pwd\n{os.getcwd()}')
# print(f'$>> ls\n{os.listdir()}')

filename = 'Asocial_media_pd.csv'

df_social_media = pd.read_csv(filename, encoding='utf-8', header=None)
# header=None => no headers present
# Set the header names as a list of integers
headers = ['TopicName', 'Date', 'Level', 'Views', 'Likes', 'Comments', 'Shares', 'Rating']
df_social_media.columns = headers
print(f'df_social_media.head(3)\n{df_social_media.head(3)}')
print(f'df_social_media.shape = {df_social_media.shape}')

X_df_feature_matrix = df_social_media.iloc[:, :-1]
print(f'X_df_feature_matrix.head(3)\n{X_df_feature_matrix.head(3)}')
print(f'X_df_feature_matrix.shape = {X_df_feature_matrix.shape}')

Y_df_target_vector = df_social_media.iloc[:, -1]
print(f'Y_df_target_vector.head(3)\n{Y_df_target_vector.head(3)}')
print(f'Y_df_target_vector.shape = {Y_df_target_vector.shape}')