import numpy as np

"""
weights.shape = (3,)

weights = array((100, 784), (100, 100), (10, 100))
"""

weights = np.load('weights.npy', allow_pickle=True)

print("Type de weights =", type(weights))

print("Shape de weights", weights.shape)
for i in range(3):
    print(weights[i].shape)


f = np.load('mnist.npz')
x_test = f['x_train']
a = x_test[11].reshape(784, ) / 255
# layers = [784, 100, 100, 10]
def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
activations = [relu, relu, sigmoid]

for k in range(3):
    print("Calcul n°", k)
    print(weights[k].shape)
    print(a.shape)
    a = activations[k](np.dot(weights[k], a))
    print(a.shape)

"""

    img = (784,)

    weights[0] = (100, 784)
    weights[1] = (100, 100)
    weights[2] = (10, 100)

    Calcul n° 0
    (100, 784) * (784,) = (100,)

    Calcul n° 1
    (100, 100) * (100,) = (100,)

    Calcul n° 2
    (10, 100) * (100,) = (10,)

Fichier de poids: 100*784 + 100*100 + 10*100 = 89400 nombres
"""
print(100*784 + 100*100 + 10*100)
print("Shape de a =", a.shape)  # = 10
for i in range(10):
    print(a[i])

"""
1.0423239490087567e-07
4.798617746074501e-06
0.17542041856036467
9.989075069900288e-07
8.711458929363381e-07
0.03521676583354137
5.2446921813831384e-09
1.5081336941068186e-07
9.82593342451335e-06
1.302714178121318e-08
"""
