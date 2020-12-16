
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Digits de MNIST
fichier = np.load('mnist.npz')

train, test = 10000, 1000
X_train = fichier['x_train']
X_train = X_train[:train]
X_train = X_train.reshape(train, 784)
y_train = fichier['y_train']
y_train = y_train[:train]

X_test = fichier['x_test']
X_test = X_test[:test]
X_test = X_test.reshape(test, 784)
y_test = fichier['y_test']
y_test = y_test[:test]

classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)
print("score:", score)

# Predictions for test data
predicted = classifier.predict(X_test)

# Print confusion matrix
confusion = metrics.confusion_matrix(y_test, predicted)
print("Matrice de confusion:\n", confusion)

disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")

plt.show()
