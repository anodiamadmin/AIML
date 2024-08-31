import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

url = './data/Asocial_media_pd.csv'
save_file = 'data/Asocial_media_save.csv'

# # DATA LOAD
df_social_media = pd.read_csv(url, encoding='utf-8', header=None)
# header=None => no headers present and sets the header names as a list of integers
# # SANITY CHECKS
# print(df_social_media.head())
# print(df_social_media.tail(3))
# print(df_social_media.info())
# print(df_social_media.describe(include='all'))
# print(df_social_media.dtypes)

# # ASSIGN HEADERS
headers = ['TopicName', 'Date', 'Level', 'Views', 'Likes', 'Comments', 'Shares', 'Rating']
df_social_media.columns = headers

# # MISSING VALUES
df_social_media.dropna(subset='Views', axis=0, inplace=True) # drop rows with missing values
df_social_media.dropna(subset='Level', axis=0, inplace=True)
df_social_media['Level'].replace('?', 'Pro')  # replace values
df_social_media['Shares'] = df_social_media['Shares'].replace('?', 0)  # replace values
df_social_media['Shares'] = df_social_media['Shares'].replace(np.nan, 0)  # replace values
df_social_media['Views'] = df_social_media['Views'].replace('?', 0)  # replace values

# # ADJUST DATA TYPES
# print(df_social_media.dtypes)
df_social_media['Views'] = df_social_media['Views'].astype(int)
df_social_media['Shares'] = df_social_media['Shares'].astype(int)
# print(df_social_media.dtypes)

# # ADJUSTING MISSING VALUES
meanViews = int(df_social_media['Views'].astype(str).astype(int).mean()) # mean of Views
df_social_media['Views'] = df_social_media['Views'].replace(0, meanViews)  # replace values
meanShares = int(df_social_media['Shares'].astype(str).astype(int).mean()) # mean of Likes
df_social_media['Shares'] = df_social_media['Shares'].replace(0, meanShares)  # replace values

# # RENAME COLUMNS
df_social_media.rename(columns={'Views': 'view_count', 'Shares': 'share_count'}, inplace=True)

# # DATA WRANGLING (SCALING)
# pd.options.display.max_columns = df_social_media.shape[1]
# print(df_social_media.describe(include='all'))
# print(df_social_media.describe())
# # Simple Feature Scaling => x = x/max(x)
df_social_media['view_count'] = df_social_media['view_count']/df_social_media['view_count'].max()
# # Min-Max Feature Scaling => x = (x-min(x))/(max(x)-min(x))
df_social_media['Likes'] = (df_social_media['Likes'] - df_social_media['Likes'].min())/(df_social_media['Likes'].max() - df_social_media['Likes'].min())
# # Zscore/ Standard Score Feature Scaling => x = (x-mean(x))/stdiv(x)
df_social_media['Comments'] = (df_social_media['Comments'] - df_social_media['Comments'].mean())/df_social_media['Comments'].std()
# # Simple Feature Scaling => x = x/max(x)
df_social_media['share_count'] = df_social_media['share_count']/df_social_media['share_count'].max()
# print(df_social_media.describe())

# # Categorical -> Numeric (One-Hot Encoding)
df_OneHotLevels = pd.get_dummies(df_social_media['Level'])
df_social_media = pd.concat([df_social_media, df_OneHotLevels], axis=1)

# # SANITY CHECKS & Descriptive Statistics
# print(df_social_media.describe())           # stats of all numeric columns only  # NaN values are EXCLUDED
print(df_social_media.describe(include='all'))     # stats of all columns including categorical vars
# print(df_social_media.head())
# print(df_social_media.tail(3))
# print(df_social_media.info())
# print(df_social_media.dtypes)


# OUTLIER ANALYSIS: Box Plots

# # SAVE/ UPLOAD FILE
df_social_media.to_csv(save_file, index=False) # index=False => no index column in the

# # # BINNING (SCALING)
# bins = np.geomspace(df_social_media['view_count'].min(), df_social_media['view_count'].max(), 6)
bins = np.linspace(1, 6, 6)
plt.hist(df_social_media['Rating'], bins=bins, edgecolor='black', color='purple', linewidth=1)

plt.title("Rating of YouTube Videos")
plt.xlabel("Rating")
plt.ylabel("Number of Videos")
width = 0.5
plt.xticks(ticks=bins+width,  labels=['Poor', 'Average', 'Good', 'Great', 'Viral', ''], rotation=30)
plt.yticks(ticks=np.arange(0, 150, 25), labels=np.arange(0, 150, 25), rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/A02PreProcessingHist.png')
plt.show()