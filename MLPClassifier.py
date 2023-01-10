import cv2
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import svm



class KNN():

    def train(self, img):
        # read all image
        image_list = list()
        folder_path = r'C:\Users\User\Desktop\project2\train\true'
        my_list = os.listdir(folder_path)

        for imPath in my_list:
            image = cv2.imread(f'{folder_path}/{imPath}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_list.append(image)

        folder_path = r'C:\Users\User\Desktop\project2\train\false'
        my_list = os.listdir(folder_path)

        for imPath in my_list:
            image = cv2.imread(f'{folder_path}/{imPath}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_list.append(image)

        # creat train array
        train = np.array(image_list[0].reshape(-1, 400)).astype(np.float32)
        for i in range(1, 1326):
            train = np.concatenate((train, image_list[i].reshape(-1, 400)), axis=0).astype(np.float32)

        # creat test array
        test = img
        testA = np.array(test.reshape(-1, 400)).astype(np.float32)

        # creat label for answer
        k1 = np.array([1])
        k2 = np.array([0])
        label = np.concatenate(
            (k1.repeat(702)[:, np.newaxis].astype(np.float32), k2.repeat(624)[:, np.newaxis].astype(np.float32)))

        # implement
        x_train, x_test, y_train, y_test = train_test_split(train, label.ravel(),
                                                            test_size=0.05,
                                                            random_state=1230,
                                                            stratify=label.ravel())

        clf = MLPClassifier(hidden_layer_sizes=(500, 500), activation='tanh', alpha=0.001)
        clf.fit(x_train, y_train)

        y_pred_1 = clf.predict(x_test)
        y_pred_2 = clf.predict(train)

        print('accuracy test = %.2f' % accuracy_score(y_test, y_pred_1))
        print('accuracy train = %.2f' % accuracy_score(label.ravel(), y_pred_2))
        cm = confusion_matrix(y_test, y_pred_1)
        print(cm)

        correct = np.where(y_pred_1 == y_test)[0]
        incorrect = np.where(y_pred_1 != y_test)[0]
        res = clf.predict(testA)

        return res


img = cv2.imread(r'C:\Users\User\Desktop\project2\train\true\4719.jpg', cv2.IMREAD_GRAYSCALE)
print(KNN().train(img))
# cv2.imshow('img', img)
cv2.waitKey(0)
