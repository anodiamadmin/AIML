import pandas as pd

url = './data/Asocial_media_pd.csv'

df_social_media = pd.read_csv(url, encoding='utf-8', header=None)
# header=None => no headers present
# Set the header names as a list of integers
headers = ['TopicName', 'Date', 'Level', 'Views', 'Likes', 'Comments', 'Shares', 'Rating']

df_social_media.columns = headers
print(f'df_social_media.head(3)\n{df_social_media.head(3)}')

df_likes = df_social_media['Likes']
print(f'\ndf_likes.head(3)\n{df_likes.head(3)}')