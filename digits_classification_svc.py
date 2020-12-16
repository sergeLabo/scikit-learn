"""
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

This example is commented in the
:ref:`tutorial section of the user manual <introduction>`.

"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, metrics

# Digits de MNIST
fichier = np.load('mnist.npz')

train, test = 10000, 100
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

# Définition du graphique 3x8 images
_, axes = plt.subplots(3, 8, figsize=(20,10))

# Affichage des 8 premières images de train
img_train = list(zip(X_train[:8], y_train[:8]))
for ax, (image, label) in zip(axes[0, :], img_train):
    ax.set_axis_off()
    ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'{label}')

# Affichage des 8 premières images de test
img_test = list(zip(X_test[:8], y_test[:8]))
for ax, (image, label) in zip(axes[1, :], img_test):
    ax.set_axis_off()
    ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'{label}')

# C-Support Vector Classification
classifier = svm.SVC()
print("fit ...")
classifier.fit(X_train, y_train)

print("predict ...")
predicted = classifier.predict(X_test)

print("Images and predictions de test ...")

for i in range(8):
    print(y_test[i], predicted[i])

print(predicted)
images_and_predictions = list(zip(X_test[:8], predicted[:8]))

for ax, (image, prediction) in zip(axes[2, :], images_and_predictions):
    ax.set_axis_off()
    ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'{prediction}')

print("Classification report for classifier\n")
print(classifier, metrics.classification_report(y_test, predicted))

disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)

disp.figure_.suptitle("Confusion Matrix")

print(f"\nConfusion matrix:\n{disp.confusion_matrix}")

plt.show()
