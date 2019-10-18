from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(palette='bright')
import pandas as pd

data = pickle.load(open('features/val.dat', 'rb'))

dimred = MDS(2)
reduced_data = dimred.fit_transform(data['data'])

df = pd.DataFrame()
df['x_1'] = reduced_data[:, 0]
df['x_2'] = reduced_data[:, 1]
df['cls'] = data['labels']

sns.scatterplot(x='x_1', y='x_2', data=df, 
                legend='full', hue='cls',
                palette=sns.color_palette('bright'))
plt.show()