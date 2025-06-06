import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# (preprocessing) https://www.kaggle.com/code/jvedarutvija/music-genre-classification
# (visualization pca) https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
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

# split dataset: X_train(6993, 58) y_train(6993,) total(9990)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

# 5-NN classifier
k_neighbors = 5
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

training_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training accuracy {training_accuracy:.3f}")
print(f"Test accuracy {test_accuracy:.3f}")

X_embedded = TSNE().fit_transform(data)
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.title('t-SNE Visualization')
plt.show()

# Fit PCA on training data
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Train KNN on PCA-reduced train data
knn_2d = KNeighborsClassifier()
knn_2d.fit(X_train_reduced, y_train)

y_pred_train = knn_2d.predict(X_train_reduced)
y_pred_test = knn_2d.predict(X_test_reduced)

training_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training accuracy {training_accuracy:.3f}")
print(f"Test accuracy {test_accuracy:.3f}")

# Plot decision boundary on test reduced data
disp = DecisionBoundaryDisplay.from_estimator(
    knn_2d,
    X_test_reduced,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    alpha=0.5,
)

disp.ax_.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=y_test, edgecolor='k')
plt.title("Decision Boundary of KNN with PCA with k=5")
plt.show()
