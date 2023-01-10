from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import cv2

img = cv2.imread(r'C:\Users\User\Desktop\project2\train\new_correct\32447.jpg', cv2.IMREAD_GRAYSCALE)

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

test = cv2.resize(img.copy(), (20, 20))
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
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

cv = StratifiedKFold(shuffle=True)
gamma_range = np.logspace(-3, 5, 9)
c_range = np.logspace(-3, 5, 9)
param_grid = dict(gamma=gamma_range, C=c_range)

grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=cv)

grid.fit(train, label.ravel())

print(grid.best_params_, grid.best_score_)

# clf = svm.SVC(gamma=0.01, kernel='poly', probability=True)
# clf.fit(x_train_scaled, y_train)
#
# acc = clf.score(x_test_scaled, y_test)
#
# y_pred = clf.predict(x_test_scaled)
#
# accuracy = accuracy_score(y_test, y_pred)
#
# conf_mat = confusion_matrix(y_test, y_pred)
#
# print('\nSVM Trained Classifier Accuracy: ', acc)
# print('\nPredicted Values: ', y_pred)
# print('\nAccuracy of Classifier on Validation Images: ', accuracy)
# print('\nConfusion Matrix: \n', conf_mat)
#
# testA_scaled = scaler.transform(testA)
# ans = clf.predict(testA_scaled)
# print(ans)
