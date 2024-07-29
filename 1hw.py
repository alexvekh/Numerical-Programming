import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Завантаження даних iris
iris = load_iris()
    # print(iris.DESCR)


# Створення DataFrame з атрибутів та цільових значень
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# # Додавання стовпця з цільовими значеннями
df['target'] = iris.target

print(df.head())

palette = {0: 'red', 1: 'green', 2: 'blue'}  # Визначення палітри кольорів
sns.pairplot(df, hue='target', diag_kind='kde', palette=palette)
    # sns.pairplot(df, hue='target', palette= "tab10", diag_kind='hist') 
plt.suptitle('Парний розподіл для кожної пари ознак', y=1.02)
plt.show()

# Включення теплової карти кореляції допомагає зрозуміти взаємозв'язки між змінними, що може бути корисним для інтерпретації результатів кластеризації.

pearsoncorr = df.drop('target', axis=1).corr()

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=1)
fig.tight_layout()

from sklearn.preprocessing import StandardScaler

# Стандартизація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('target', axis=1))

# Перетворення стандартизованих даних у DataFrame для зручності
df_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
df_scaled = df_scaled.drop(columns=['petal width (cm)'])  # Видалення неважливих стовпців
df_scaled['target'] = df['target']

# Перегляд перших кількох рядків стандартизованого DataFrame
print(df_scaled.head())

# Базові статистичні характеристики стандартизованих даних
print(df_scaled.describe())

from sklearn.cluster import SpectralClustering

# Виконання спектральної кластеризації
spectral_clustering = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0)
clusters = spectral_clustering.fit_predict(X_scaled)

# Додавання спрогнозованих кластерів до DataFrame
df_scaled['cluster'] = clusters

# Перегляд перших кількох рядків DataFrame з кластерами
print(df_scaled.head())

from sklearn.metrics import confusion_matrix

# Порівняння спрогнозованих кластерів з дійсними класами
conf_matrix = confusion_matrix(df_scaled['target'], df_scaled['cluster'])
print('Confusion Matrix:')
print(conf_matrix)

plt.figure(figsize=(12, 6))

# Розподіл спостережень за справжніми класами
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue='target', data=df_scaled, palette='viridis')
plt.title('Справжні класи')

# Розподіл спостережень за спрогнозованими кластерами
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue='cluster', data=df_scaled, palette='viridis')
plt.title('Спрогнозовані кластери')

plt.show()