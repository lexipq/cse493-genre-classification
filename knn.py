import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns


# (preprocessing) https://www.kaggle.com/code/jvedarutvija/music-genre-classification
# (visualization tsne) https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html

df = pd.read_csv('data/features_3_sec.csv')
df = df.drop(labels='filename', axis=1)

# get all target classes and convert -> index [0-9]
classes = df.iloc[:,-1]
converter = LabelEncoder()
y = converter.fit_transform(classes)

# standardize all features: zero mean unit variance
norm = StandardScaler()
data = norm.fit_transform(np.array(df.iloc[:,:-1], dtype=float))
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)


# 5-NN classifier
k_neighbors = 5
knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(X_train, y_train)

y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

training_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"full-dimension training accuracy {training_accuracy:.3f}")
print(f"full-dimension test accuracy {test_accuracy:.3f}")


# t-SNE reduction visualization
tsne_out = TSNE(n_components=2).fit_transform(data)
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(tsne_out, y, test_size=0.3)

# visualize reduced data
fig, axes = plt.subplots(1, 2)
# plt.scatter(x=tsne_out[:,0], y=tsne_out[:,1], c=y)
sns.scatterplot(x=tsne_out[:,0], y=tsne_out[:,1], hue=classes, palette='muted', ax=axes[0])

knn_2d = KNeighborsClassifier(n_neighbors=k_neighbors)
knn_2d.fit(X_train_reduced, y_train)

y_pred_train = knn_2d.predict(X_train_reduced)
y_pred_test = knn_2d.predict(X_test_reduced)

training_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"reduced-dimension training accuracy {training_accuracy:.3f}")
print(f"reduced-dimension test accuracy {test_accuracy:.3f}")

disp = DecisionBoundaryDisplay.from_estimator(
    knn_2d,
    tsne_out,
    response_method="predict",
    alpha=0.5,
    ax=axes[1]
)

disp.ax_.scatter(tsne_out[:, 0], tsne_out[:, 1], c=y, edgecolor="k")
plt.title("Decision Boundary of t-SNE KNN with k=5")
plt.tight_layout()
plt.show()
