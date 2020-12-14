
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

f = np.load('mnist.npz')

X_train = f['x_train']
X_test = f['x_test']
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
y_train = f['y_train']
y_test = f['y_test']

clf = KNeighborsClassifier(n_neighbors=3, n_jobs=8)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)

print("score:", score)

# Predictions for test data
predicted = clf.predict(X_test)
# Print confusion matrix
confusion = confusion_matrix(y_test, predicted)
print("Matrice de confusion:\n", confusion)
