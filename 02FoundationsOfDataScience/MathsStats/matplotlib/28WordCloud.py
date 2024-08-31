import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator     # pip install wordcloud
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

ronaldinho_mask = np.array(Image.open("./data/ronaldinho_silhouette.png"))
# print(f"{ronaldinho_mask}")
ronaldinho_mask = np.where(ronaldinho_mask > 127, 0, 255)
# print(f"{ronaldinho_mask}")
# print(f"{ronaldinho_mask.shape}")

df = pd.read_csv("./data/ronaldinho-data-130k-v2.csv", index_col=0)
# Start with one review:
# text = df.description[0]
text = " ".join(review for review in df.description)
print(f"There are {len(text)} words in the combination of all review.")

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "football", "now", "soccer", "wine", "flavor", "flavors"])

# Generate a word cloud image
# wordcloud = WordCloud().generate(text)
# wordcloud = WordCloud(max_font_size=50, max_words=1000, stopwords=stopwords, background_color="white").generate(text)
wordcloud = WordCloud(contour_width=3, contour_color='firebrick', max_font_size=50, mask=ronaldinho_mask,
                       background_color="white", max_words=1000, stopwords=stopwords).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("./plots/28WordCloud.png")