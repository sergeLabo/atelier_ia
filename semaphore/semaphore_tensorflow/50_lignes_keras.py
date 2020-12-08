import os
import numpy as np
from tensorflow import keras

epochs = 2
fichier = np.load('./semaphore.npz')
x_train = fichier['x_train']
y_train = fichier['y_train']
x_test = fichier['x_test']
y_test = fichier['y_test']
train_images = x_train.reshape(60000, 40, 40)
train_labels = y_train
test_images = x_test.reshape(10000, 40, 40)
test_labels = y_test
class_names = list("abcdefghijklmnopqrstuvwxyz ")
model = keras.Sequential([  keras.layers.Flatten(input_shape=(40, 40)),
                            keras.layers.Dense(128, activation='relu'),
                            keras.layers.Dense(27, activation='softmax') ])
model.compile(  optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'] )
print("\n\nTraining ...\n")
model.fit(train_images, train_labels, epochs=epochs)
test_loss, test_acc = model.evaluate(test_images, test_labels)
eff = round(test_acc*100, 1)
print(f"\nTesting ......    \nEfficacité sur les images test: {eff} %")
predictions = model.predict(test_images)
print("\nTest sur 1 image")
img = test_images[11]
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
res = np.argmax(predictions_single[0])
print("Image: {} Prédiction {}".format(test_labels[11], res))
print("\nDétails:")
n = 0
dct = {}
for item in predictions_single[0]:
    dct[n] = item
    n += 1
sort_dct = sorted(dct.items(), key=lambda x: x[1], reverse=True)
for i in range(10):
    print("        ", sort_dct[i][0], "  ", sort_dct[i][1])
