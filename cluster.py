import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

image_list = list()
folder_path = r'C:\Users\User\Desktop\project2\train\new_correct'
path = r'C:\Users\User\Desktop\project2\cluster'
my_list = os.listdir(folder_path)
j = 0
for imPath in tqdm(my_list):
    image = cv2.imread(f'{folder_path}/{imPath}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_list.append(image)
    j += 1
    if j == 30000: break
train = np.array(image_list[0].reshape(-1, 400)).astype(np.float32)
for i in tqdm(range(1, 29999)):
    train = np.concatenate((train, image_list[i].reshape(-1, 400)), axis=0).astype(np.float32)

k_means = KMeans(n_clusters=18).fit(train)
label = k_means.labels_


j = 0
for imPath in tqdm(my_list):
    if not os.path.exists(path + '\\' + str(label[j])):
        os.makedirs(path + '\\' + str(label[j]))
    cv2.imwrite(path + '\\' + str(label[j]) + '\\' + str(j) + '.jpg', image_list[j])
    j += 1
    if j == 29998: break
