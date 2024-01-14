import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load your data
X_train = np.loadtxt('F:\Animal Detection Project\ImageClassificationCNNKerasDataset\input.csv', delimiter=',')
Y_train = np.loadtxt('F:\Animal Detection Project\ImageClassificationCNNKerasDataset\labels.csv', delimiter=',')

X_test = np.loadtxt('F:\Animal Detection Project\ImageClassificationCNNKerasDataset\input_test.csv', delimiter=',')
Y_test = np.loadtxt('F:\Animal Detection Project\ImageClassificationCNNKerasDataset\labels_test.csv', delimiter=',')

# Flatten the image data
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)

# Split the data into training and testing sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Define the SVM model
model = SVC(kernel='linear', random_state=42)

# Train the model
model.fit(X_train, Y_train)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(Y_val, y_val_pred)
print("Validation Accuracy:", accuracy)

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_test_pred)
print("Test Accuracy:", accuracy)

# Display a random test image
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2].reshape(100, 100, 3))
plt.show()

# Make a prediction
pred = model.predict(X_test[idx2].reshape(1, -1))
if pred == 0:
    pred_label = 'dog'
else:
    pred_label = 'cat'

print("Our model says it is a:", pred_label)