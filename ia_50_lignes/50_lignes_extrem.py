import numpy as np
from tensorflow import keras
fichier = np.load('mnist.npz')
x_train = fichier['x_train'] / 255
y_train = fichier['y_train']
x_test = fichier['x_test'] / 255
y_test = fichier['y_test']
model = keras.Sequential([  keras.layers.Flatten(input_shape=(28, 28)),
                            keras.layers.Dense(128, activation='relu'),
                            keras.layers.Dense(27, activation='softmax') ])
model.compile(  optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'] )
print("\n\nTraining ...\n")
model.fit(x_train, y_train, epochs=2)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTesting\nEfficacité sur les images test: {round(test_acc*100, 1)} %")
