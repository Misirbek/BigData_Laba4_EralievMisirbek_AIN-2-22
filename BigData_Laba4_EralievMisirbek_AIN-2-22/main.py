import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics

digits = load_digits()

print(f"Размер данных: {digits.data.shape}")
print(f"Размер целевых меток: {digits.target.shape}")

plt.gray()
plt.matshow(digits.images[0])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

plt.figure(figsize=(10, 4))
for index, (image, label) in enumerate(zip(digits.images[:5], digits.target[:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title(f'Цифра: {label}')
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Прогнозы: {y_pred[:5]}")
print(f"Реальные значения: {y_test[:5]}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность k-NN классификатора на тестовых данных: {accuracy * 100:.2f}%")

print(f"Точность через метод score: {knn.score(X_test, y_test) * 100:.2f}%")

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 9))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title("Матрица неточностей")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, digits.target_names, rotation=45)
plt.yticks(tick_marks, digits.target_names)

for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center")

plt.ylabel('Верная метка')
plt.xlabel('Прогнозируемая метка')
plt.show()
