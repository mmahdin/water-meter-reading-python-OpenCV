import cv2
import numpy as np
import os


class KNN():

    def train(self, img):
        # read all image
        image_list = list()
        for i in range(10):
            folder_path = f'C:\\Users\\User\\Desktop\\project2\\TrainEnglish\\{i}'
            my_list = os.listdir(folder_path)
            count = 0
            for imPath in my_list:
                if count == 10: break
                image = cv2.imread(f'{folder_path}/{imPath}')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_list.append(image)
                count += 1

        # creat train array
        train = np.array(image_list[0].reshape(-1, 400)).astype(np.float32)
        for i in range(1, 100):
            train = np.concatenate((train, image_list[i].reshape(-1, 400)), axis=0).astype(np.float32)

        # creat test array
        test = img
        testA = np.array(test.reshape(-1, 400)).astype(np.float32)

        # creat label for answer
        k = np.arange(10)
        label = k.repeat(10)[:, np.newaxis].astype(np.float32)

        # implement
        knn = cv2.ml.KNearest_create()
        knn.train(train, cv2.ml.ROW_SAMPLE, label)
        ret, result, neighbours, dist = knn.findNearest(testA, k=5)

        return result
