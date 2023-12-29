# ENEL 525 Final Project
# Cheyenne Goh (30040528)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

IMG_SIZE = 200 # Image dimension
CLASS_SIZE = 700 # Number of images in each class
EPOCHS = 12 # Number of epochs

# Load a dataset
path = '/content/flowers'
labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

dataset = []
targets = []

for label in labels:
  directory = os.path.join(path, label)
  count = 0

  for filename in os.listdir(directory):
    if count >= CLASS_SIZE:
      break;

    img = cv2.imread(os.path.join(directory, filename))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.normalize(img, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)

    dataset.append(img)
    targets.append(label)

    count += 1

# Build additional test cases
test_cases_dataset = []
test_cases_targets = []

test_cases = [
    ['/content/daisy0.jpg', 'daisy'],
    ['/content/daisy1.jpg', 'daisy'],
    ['/content/dandelion0.jpg', 'dandelion'],
    ['/content/dandelion1.jpg', 'dandelion'],
    ['/content/rose0.jpg', 'rose'],
    ['/content/rose1.jpg', 'rose'],
    ['/content/sunflower0.jpg', 'sunflower'],
    ['/content/sunflower1.jpg', 'sunflower'],
    ['/content/tulip0.jpg', 'tulip'],
    ['/content/tulip1.jpg', 'tulip']
]

for test_case in test_cases:
  img = cv2.imread(test_case[0])
  img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
  img = cv2.normalize(img, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)

  test_cases_dataset.append(img)
  test_cases_targets.append(test_case[1])

# Converting the dataset/targets into numpy arrays
dataset = np.array(dataset)
test_cases_dataset = np.array(test_cases_dataset)

# Data verification step (Images)
for i in range(0, len(dataset), CLASS_SIZE):
  idx = rd.randint(i, i + CLASS_SIZE)
  plt.imshow(dataset[idx])
  plt.title("Class: " + targets[idx])
  plt.show()

# Split the dataset into training, validation and testing
X_train, X_test, y_train, y_test = train_test_split(dataset, targets, test_size=0.15)

# Convert targets to one-hot encoding
le = LabelEncoder()
y_train = tf.keras.utils.to_categorical(le.fit_transform(y_train), 5)

# Create and train the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', activation='relu', input_shape=(200, 200, 3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

  tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

  tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

  tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_split=(0.15 / 0.7),
    epochs=EPOCHS
)

# Model summary
model.summary()

# Prediction
y_pred = model.predict(X_test)
y_pred = le.inverse_transform(np.argmax(y_pred, axis=1))

y_pred_test_cases = model.predict(test_cases_dataset)
y_pred_test_cases = le.inverse_transform(np.argmax(y_pred_test_cases, axis=1))

for i in range(len(test_cases_dataset)):
  plt.imshow(test_cases_dataset[i])
  plt.title("Target: " + test_cases_targets[i] + ", Prediction: " + y_pred_test_cases[i])
  plt.show()

# Compute and view the confusion matrix
confusion_mx = confusion_matrix(y_test, y_pred, labels=labels)
print(confusion_mx)

# Compute accuracy
accuracy = ((confusion_mx[0, 0] + confusion_mx[1, 1] + confusion_mx[2, 2] + confusion_mx[3, 3] + confusion_mx[4, 4]) / len(y_test)) * 100
print("Accuracy =", accuracy, "%")

# Accuracy/loss plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Validation Loss'])
plt.show()