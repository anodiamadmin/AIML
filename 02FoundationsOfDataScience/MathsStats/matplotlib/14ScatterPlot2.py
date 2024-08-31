import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

# Create Social Media CSV file
#############################
videos = 200
shares = (np.random.poisson(2, videos)
          + np.random.poisson(1, videos) * np.random.poisson(2, videos)
          + np.random.randint(-3, 3, videos))
comments = (np.random.poisson(3, videos) + shares * np.random.poisson(2, videos)
            + np.random.randint(-9, 9, videos))
likes = (np.random.poisson(4, videos) + comments * np.random.poisson(3, videos)
         + np.random.randint(-27, 27, videos))
views = (np.random.poisson(5, videos) + likes * np.random.poisson(4, videos)
         + np.random.randint(-81, 81, videos))
social_media = np.transpose(np.array([views, likes, comments, shares]))
# df_social_media = pd.DataFrame(social_media, columns=['Views', 'Likes', 'Comments', 'Shares'])
# df_social_media.to_csv('./data/social_media.csv', index=False)

# # Plot Social Media CSV Data
# ############################
# df = pd.read_csv('./data/social_media.csv')
# views, likes, comments, shares = df['Views'].to_numpy(), df['Likes'].to_numpy(), df['Comments'].to_numpy(), df['Shares'].to_numpy()

plt.scatter(views, likes, label="Size~=Shares", s=shares, cmap="Blues", c=comments,
            edgecolors="black", linewidths=1, alpha=0.5)
plt.title("Social Media Analysis")
plt.xlabel("Views")
plt.ylabel("Likes")
plt.xscale('log')
plt.yscale('log')
cbar = plt.colorbar()
cbar.set_label("Comments")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/14ScatterPlot2.png')
plt.show()