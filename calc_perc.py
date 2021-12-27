import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


saved_preds_path = 'saved_images/lung/individual_images'
preds_by_dolphin_id = os.listdir(saved_preds_path)

print('Pathology percentage:')

for p in preds_by_dolphin_id:
    lungs_dir = f'saved_images/lung/individual_images/{p}'
    lung_filenames = os.listdir(lungs_dir)

    pathology_dir = f'saved_images/pathology/individual_images/{p}'
    pathology_filenames = os.listdir(lungs_dir)

    if len(lung_filenames) != len(pathology_filenames):
        print('ERROR: NOT THE SAME NUMBER OF LUNG AND PATHOLOGY IMAGES!!!')

    num_images = len(lung_filenames)
    n_lungs_2 = 0
    n_patho_2 = 0
    for i in range(num_images):
        lung_path = os.path.join(lungs_dir, lung_filenames[i])
        pathology_path = os.path.join(pathology_dir, pathology_filenames[i])
        rgb_weights = [0.2989, 0.5870, 0.1140]
        lung = (np.dot(plt.imread(lung_path)[..., :3], rgb_weights) > 0.5).astype(int)
        pathology = (np.dot(plt.imread(pathology_path)[..., :3], rgb_weights) > 0.5).astype(int)
        n_lungs_2 += np.count_nonzero(lung == 1)
        n_patho_2 += np.count_nonzero(lung + pathology == 2)

    print(f'{p}: {n_patho_2} / {n_lungs_2} = {n_patho_2/n_lungs_2}')