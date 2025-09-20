import tensorflow as tf 
from tensorflow.keras import datasets, layers, models 
import matplotlib.pyplot as plt
import numpy as np
from random import randint, uniform


#num1= randint(-100, 100)
#print(f"This is the number:{num1}")

#num2= uniform(-100, 100)

#print(f"this is the float number:{num2}")

#Load CIFAR-10 dataser
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

#Normalizar valores en escala de 0 y 1
X_train, X_test = X_train / 255.0, X_test / 255.0

#Dataser classes 
class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Dataset size",X_train.shape, y_train.shape)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

#Model Convolutional Neural Network
model = models.Sequential( [
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])  

tes_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(test_acc)

313/313



