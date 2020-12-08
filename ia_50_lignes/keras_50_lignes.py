import os
import numpy as np
from tensorflow import keras

epochs = 2
fichier = np.load('mnist.npz')
x_train = fichier['x_train'] / 255
y_train = fichier['y_train']
x_test = fichier['x_test'] / 255
y_test = fichier['y_test']

class_names = list("abcdefghijklmnopqrstuvwxyz ")
model = keras.Sequential([  keras.layers.Flatten(input_shape=(28, 28)),
                            keras.layers.Dense(128, activation='relu'),
                            keras.layers.Dense(27, activation='softmax') ])
model.compile(  optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'] )
print("\n\nTraining ...\n")
model.fit(x_train, y_train, epochs=epochs)
test_loss, test_acc = model.evaluate(x_test, y_test)
eff = round(test_acc*100, 1)
print(f"\nTesting ......    \nEfficacité sur les images test: {eff} %")
predictions = model.predict(x_test)
print("\nTest sur 1 image")
img = x_test[11]
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
rep = y_test[11], np.argmax(predictions_single[0])
print("Image: {} Prédiction {}".format(rep[0], rep[1]))
print("\nDétails:")
n = 0
dct = {}
for item in predictions_single[0]:
    dct[n] = item
    n += 1
sort_dct = sorted(dct.items(), key=lambda x: x[1], reverse=True)
for i in range(10):
    print("        ", sort_dct[i][0], "  ", sort_dct[i][1])
