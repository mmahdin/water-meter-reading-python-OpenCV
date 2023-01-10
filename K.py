import cv2
import numpy as np
import os


class KNN():

    def train(self, img):
        # read all image
        image_list = list()
        folder_path = r'C:\Users\User\Desktop\project2\train\correct'
        my_list = os.listdir(folder_path)

        for imPath in my_list:
            image = cv2.imread(f'{folder_path}/{imPath}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_list.append(image)

        folder_path = r'C:\Users\User\Desktop\project2\train\incorrect'
        my_list = os.listdir(folder_path)

        for imPath in my_list:
            image = cv2.imread(f'{folder_path}/{imPath}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_list.append(image)

        # creat train array
        train = np.array(image_list[0].reshape(-1, 400)).astype(np.float32)
        for i in range(1, 284):
            train = np.concatenate((train, image_list[i].reshape(-1, 400)), axis=0).astype(np.float32)

        # creat test array
        test = img
        testA = np.array(test.reshape(-1, 400)).astype(np.float32)

        # creat label for answer
        k1 = np.array([1])
        k2 = np.array([0])
        label = np.concatenate(
            (k1.repeat(135)[:, np.newaxis].astype(np.float32), k2.repeat(149)[:, np.newaxis].astype(np.float32)), axis=0)

        # implement
        knn = cv2.ml.KNearest_create()
        knn.train(train, cv2.ml.ROW_SAMPLE, label)
        ret, result, neighbours, dist = knn.findNearest(testA, 3)

        return result
