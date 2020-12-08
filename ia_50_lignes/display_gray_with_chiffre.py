import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os

fichier = np.load('mnist.npz')

x_train = fichier['x_train']
y_train = fichier['y_train']

x_test = fichier['x_test']
y_test = fichier['y_test']

cv2.namedWindow("IMG")

img = x_train[10]
img = img.reshape(28, 28)

img_gray = cv2.resize(img, (840, 840), interpolation=cv2.INTER_NEAREST)
img_rbg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

loop = 1
while loop:
    for i in range(28):
        for j in range(28):
            cv2.putText(
                        img_rbg,
                        str(img[j][i]),
                        (i*30 + 10, j*30 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                        )

    cv2.imshow("IMG", img_rbg)
    # #cv2.imwrite('pixel.png', img_rbg)

    k = cv2.waitKey(1)

    # Space pour morceau suivant et attente
    if k == 32:
        loop = 0

loop = 1
while loop:
    n = random.randint(0, 60000)
    img = x_train[n]
    img = img.reshape(28, 28)

    img_gray = cv2.resize(img, (840, 840), interpolation=cv2.INTER_NEAREST)
    img_rbg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    cv2.putText(
                img_rbg,
                str(y_train[n]),
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                10,
                (0, 255, 0),
                4,
                cv2.LINE_AA
                )

    cv2.imshow("IMG", img_rbg)
    # #cv2.imwrite('pixel.png', img_rbg)

    k = cv2.waitKey(1000)

    # Space pour morceau suivant et attente
    if k == 32:
        loop = 0

# ## Echap pour finir le script python
# #if k == 27:
cv2.destroyAllWindows()
os._exit(0)
