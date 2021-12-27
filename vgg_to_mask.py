import os
import json
import numpy as np
import cv2

with open(r'C:\Users\tai10\Downloads\via_project_28Jun2021_11h3m_json.json') as f:
    data = json.load(f)

for itr in data:
    regions = data[itr]['regions']
    img = np.zeros((1024, 1024))
    for region in regions:
        if region and 'shape_attributes' in region and 'cx' in region['shape_attributes'] and 'cy' in region['shape_attributes']:
            if region['shape_attributes']['name'] == 'ellipse':
                img = cv2.ellipse(img,
                                  (region['shape_attributes']['cx'], region['shape_attributes']['cy']),
                                  (round(region['shape_attributes']['rx']), round(region['shape_attributes']['ry'])),
                                  round(region['shape_attributes']['theta'] / 3.14159265358979 * 360),
                                  0,
                                  360,
                                  255,
                                  -1)
            elif 'r' in region['shape_attributes']:
                img = cv2.circle(img, (region['shape_attributes']['cx'], region['shape_attributes']['cy']),
                                 int(region['shape_attributes']['r']), (255, 255, 255), -1)
            else:
                img = cv2.circle(img, (region['shape_attributes']['cx'], region['shape_attributes']['cy']), 5,
                                 (255, 255, 255), -1)
        elif region and 'shape_attributes' in region and 'all_points_x' in region['shape_attributes'] and 'all_points_y' in region['shape_attributes']:
            points = []
            for i in range(len(region['shape_attributes']['all_points_x'])):
                points.append(
                    [region['shape_attributes']['all_points_x'][i], region['shape_attributes']['all_points_y'][i]])
            img = cv2.fillPoly(img, np.array([points], dtype='int32'), 255)
    if len(regions) == 0:
        print(f'Skipped {data[itr]["filename"]}')
    cv2.imwrite(r'C:\Users\tai10\Desktop\test vgg circle and point\patho\\' + data[itr]["filename"][:-4] + '_mask.png', img)
