import numpy as np
import cv2

fichier = np.load('mnist.npz')

x_train = fichier['x_train']
y_train = fichier['y_train']

x_test = fichier['x_test']
y_test = fichier['y_test']

cv2.namedWindow("IMG")
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    img = x_train[i]
    img = img.reshape(28, 28)
    cv2.imwrite('./img_input/28_input_' + str(i) + '.png', img)
    img = cv2.resize(img, (400, 400))
    cv2.imwrite('./img_input/400_input_' + str(i) + '.png', img)
    img = cv2.resize(img, (900, 900))
    cv2.imshow("IMG", img)
    print("Image d'un:", y_train[i])
    cv2.waitKey(2000)
