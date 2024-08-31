import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

plt.style.use('ggplot')

df_resp = pd.read_csv('./data/favLanguages.csv')
ids = df_resp['Responder_id']
lang_resp = df_resp['LanguagesWorkedWith']

lang_counter = Counter()

for resp in lang_resp:
    lang_counter.update(resp.split(';'))

print(lang_counter.most_common(15))

languages = []
popularity = []

for item in lang_counter.most_common(15):
    languages.append(item[0])
    popularity.append(item[1])

languages.reverse()
popularity.reverse()

print(languages)
print(popularity)

plt.barh(languages, popularity)

plt.title("Most Popular Languages")
# plt.ylabel("Programming Languages")
plt.xlabel("Number of People Who Use")

plt.tight_layout()
plt.savefig('./plots/03BarchartHorizontal.png')
plt.show()