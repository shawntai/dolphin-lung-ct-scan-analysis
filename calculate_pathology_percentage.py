# -*- coding: utf-8 -*-
"""COLAB_FYP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14xEtmoJLEunxyM-l3j5bIgzhomv6CaHy
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from google.colab import drive
drive.mount('/content/drive')

lungs = plt.imread(r'/content/drive/MyDrive/lungs_pred_0 (99.52, 0.931).png')
patho = plt.imread(r'/content/drive/MyDrive/patho_pred_0 (99.38, 0.9).png')

rgb_weights = [0.2989, 0.5870, 0.1140]
lungs = np.dot(lungs[...,:3], rgb_weights)
patho = np.dot(patho[...,:3], rgb_weights)

n_lungs = 0
n_patho = 0

for i in range(lungs.shape[0]):
	for j in range(lungs.shape[1]):
		if lungs[i][j] > 0.5:
			n_lungs += 1
			if patho[i][j] > 0.5:
				n_patho += 1

f'{n_patho} / {n_lungs} = {n_patho/n_lungs}'

