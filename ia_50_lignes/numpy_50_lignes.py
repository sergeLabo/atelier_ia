import numpy as np, cv2
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_prime(z): return z * (1 - z)
def relu(x): return np.maximum(0, x)
def relu_prime(z): return np.asarray(z > 0, dtype=np.float32)
layers = [784, 100, 100, 10]
activations, learningrate = [relu, relu, sigmoid], 0.1
f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0
s = np.arange(len(x_train))
np.random.shuffle(s)
x_train, y_train = x_train[s], y_train[s]
Y = np.eye(10, 10)
activations_prime = [globals()[f.__name__ + '_prime'] for f in activations]
A = {}
W = [np.random.randn(layers[k+1],
     layers[k])/np.sqrt(layers[k]) for k in range(len(layers)-1)]
print("Training...")
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
for i, (a, d) in enumerate(zip(x_train, y_train)):
    if i % 500 == 0:
        print(i)
        cv2.imshow("img", a.reshape(28,28))
        cv2.waitKey(5)
    a = np.array(a, ndmin=2).T
    A[0] = a
    for k in range(len(layers)-1):
        z = np.dot(W[k], a)
        a = activations[k](z)
        A[k+1] = a
    da = a - Y[:,[d]]
    for k in range(len(layers)-2, -1, -1):   # la fameuse "backpropagation"
        dz = da * activations_prime[k](A[k+1])
        dW = np.dot(dz, A[k].T)
        da = np.dot(W[k].T, dz)
        W[k] -= learningrate * dW
print("Testing...")
S = 0
for a, d in zip(x_test, y_test):
    for k in range(len(layers)-1): a = activations[k](np.dot(W[k], a))
    if np.argmax(a) == d: S += 1
print("Accuracy: %.1f %%" % (100.0 * S / len(x_test)))
np.save('weights', W)
