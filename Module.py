import cv2
import numpy
import numpy as np
import math
import Main
import imutils
from PIL import Image
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

##############################################
MAX_DIAG_SIZE_MULTIPLE_AWAY_1 = 2
MAX_ANGLE_BETWEEN_CHARS_1 = 50
MAX_CHANGE_IN_AREA_1 = 0.3  # -> 1.3
MAX_CHANGE_IN_WIDTH_1 = 0.1  # -> 0.09
MAX_CHANGE_IN_HEIGHT_1 = 0.1  # -> 0.03

MAX_DIAG_SIZE_MULTIPLE_AWAY_2 = 4
MAX_ANGLE_BETWEEN_CHARS_2 = 50
MAX_CHANGE_IN_AREA_2 = 0.4  # -> 1.3
MAX_CHANGE_IN_WIDTH_2 = 0.2  # -> 0.09
MAX_CHANGE_IN_HEIGHT_2 = 0.2

MIN_NUMBER_OF_MATCHING_CHARS_1_2 = 4

MIN_NUMBER_OF_MATCHING_CHARS_3 = 3

MAX_WIDTH_HEIGHT_RATIO_FOR_NUMBER = 3
MIN_WIDTH_HEIGHT_RATIO_FOR_NUMBER = 1

MIN_ERROR_VERIFICATION = 0.95
MAX_ERROR_VERIFICATION = 1.03

MIN_PARTITION_ERROR = 0.1
MAX_PARTITION_ERROR = 1
INCREASE_IN_REPETITION = 0.1
ERROR_OF_HEIGHT = 1.3
ERROR_OF_WIDTH = 1.1

MAX_AREA_ERROR = 1.05
MIN_AREA_ERROR = 0.05
MAX_WIDTH_ERROR = 1.2
MIN_WIDTH_ERROR = 0.2
MAX_HEIGHT_ERROR = 1.2
MIN_HEIGHT_ERROR = 0.2
MIN_DISTANCE_ERROR = 0.95
MAX_DISTANCE_ERROR = 1.05
Y_ERROR = 0.99

##############################################
knn = cv2.ml.KNearest_create()
clf = MLPClassifier(hidden_layer_sizes=(90,), activation='tanh', alpha=0.0001, shuffle=False)
clf_2 = svm.SVC(gamma=0.001, kernel='poly', probability=True)
KNN = KNeighborsClassifier(n_neighbors=15)


#############################################


class Thresh:

    def __init__(self, img):
        try:
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            self.img = img

    def main(self):
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgTopHat = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, structuringElement)
        imgGrayscalePlusTopHat = cv2.add(self.img, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
        imgBlurred = cv2.medianBlur(imgGrayscalePlusTopHatMinusBlackHat, 5, 0)
        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19,
                                          5)

        # ret, imgThresh = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY_INV)
        # img = cv2.subtract(imgThresh, self.img)

        return imgThresh


class AdvancedThresh:

    # get image from PIL
    def div(self, img):
        # col = Image.open(r'C:\Users\User\Desktop\project2\image\7.jpg')
        gray = img.convert('L')
        gray.save('l.jpg')
        bw = gray.point(lambda x: 0 if x < 90 else 255, '1')
        bw.save("cp19.png")
        image = cv2.imread('cp19.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        div = cv2.divide(gray, blur, scale=192)

        return div


class Contour:

    def get_contours(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            try:
                contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            finally:
                contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def delete_under_mean(self, contours):
        areaList = list()
        for i in contours:
            area = cv2.contourArea(i)
            areaList.append(area)

        mean = np.mean(areaList)

        new_contours_list = []
        counter = 0
        for i in contours:
            if areaList[counter] > mean / 3:
                new_contours_list.append(i)
            counter += 1

        return new_contours_list

    def zScore(self, contours, threshold=3):
        counter = 0
        areaList = list()
        for i in contours:
            area = cv2.contourArea(i)
            areaList.append(area)
            counter += 1

        mean = np.mean(areaList)
        std = np.std(areaList)
        outlier = []
        new_contours = []
        counter = 0
        for i in areaList:
            z = (i - mean) / std
            if z > threshold:
                outlier.append(i)
            else:
                new_contours.append(contours[counter])
            counter += 1

        return new_contours, mean, std, outlier, areaList

    def delete_zeros(self, contours):
        areaList = list()
        for i in contours:
            area = cv2.contourArea(i)
            areaList.append(area)

        mean = np.mean(areaList)

        new_contours_list = []
        counter = 0
        for i in contours:
            if areaList[counter] != 0.0:
                new_contours_list.append(i)
            counter += 1

        return new_contours_list


class ImageDetails:

    def __init__(self, contours, img):
        self.contour = contours

        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)

        self.img = cv2.getRectSubPix(img, (self.intBoundingRectWidth, self.intBoundingRectHeight), (
            (2 * self.intBoundingRectX + self.intBoundingRectWidth) / 2,
            (2 * self.intBoundingRectY + self.intBoundingRectHeight) / 2))


class ProccesseImage:

    def Shred_image(self, contours, img):
        listOfImage = []
        for i in contours:
            instance = ImageDetails(i, img)
            listOfImage.append(instance)

        return listOfImage


class OpenCv:
    # image should be gray scale and threshed
    def approximating_contours(self, img):
        img = img.copy()
        contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for c in contours:
            accuracy = 0.03 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, accuracy, True)
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            cv2.imshow('Approx polyDP', img)

        return img

    def identifying_shapes(self, img):
        img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 127, 255, 1)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                shape_name = "Triangle"
                cv2.drawContours(img, [cnt], 0, (0, 255, 0), -1)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # cv2.boundingRect return the left width and height in pixels, starting from the top
                    # left corner, for square it would be roughly same
                    if abs(w - h) <= 3:
                        shape_name = "square"
                        # find contour center to place text at center
                        cv2.drawContours(img, [cnt], 0, (0, 125, 255), -1)
                        cv2.putText(img, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                else:
                    shape_name = "Reactangle"
                    # find contour center to place text at center
                    cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.putText(img, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            elif len(approx) == 10:
                shape_name = 'star'
                cv2.drawContours(img, [cnt], 0, (255, 255, 0), -1)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            elif len(approx) >= 15:
                shape_name = 'circle'
                cv2.drawContours(img, [cnt], 0, (0, 255, 255), -1)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(img, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        return img

    def line_detection(self, img):
        img = img.copy()
        h, w = img.shape
        edges = cv2.Canny(img, 100, 170, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 151)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for i in range(0, len(lines)):
                for rho, theta in lines[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    try:
                        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 15)
                    except:
                        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            return img
        except:
            return img

    def line_detection_p(self, img):
        edges = cv2.Canny(img, 100, 170, apertureSize=3)

        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, 20, 20)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for i in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[i]:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        except:
            return img
        return img

    def detect_plate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        try:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)
            return new_image
        except:
            return None

    def corner_detection(self, img, draw=False):
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            img_gray = img
        corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)
        corners_1_4 = sorted(corners, key=lambda point: math.sqrt(point.ravel()[0] ** 2 + point.ravel()[1] ** 2))
        corners_3 = sorted(corners, key=lambda point: point.ravel()[1])
        corners_2 = (corners_1_4[-1].ravel()[0], corners_1_4[0].ravel()[1])

        corners = (corners_1_4[0].ravel(), corners_2, corners_3[-1].ravel(), corners_1_4[-1].ravel())
        if draw:
            for i in corners:
                x, y = i[0], i[1]
                cv2.circle(img, (x, y), 6, (0, 255, 0), thickness=-1)

        return corners, img

    def crop_image_by_corner(self, img, corners):
        img = cv2.getRectSubPix(img, (corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]), (
            corners[0][0] + (corners[1][0] - corners[0][0]) / 2, corners[0][1] + (corners[2][1] - corners[0][1]) / 2))

        return img

    def rotate_photo(self, corners, img):
        first_point = corners[0]
        second_point = corners[3]
        c_x = (first_point[0] + second_point[0]) / 2
        c_y = (first_point[1] + second_point[1]) / 2
        hypotenuse = math.sqrt((second_point[0] - first_point[0]) ** 2 + (second_point[1] - first_point[1]) ** 2)
        angle_in_deg = math.asin((second_point[1] - first_point[1]) / hypotenuse) * (180.0 / math.pi)
        height, width, _ = img.shape
        print(angle_in_deg)
        rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), 20, 1.0)
        img_rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

        return img_rotated


class DetectChar:

    def is_char(self, imgThresh):
        pass

    def possible_char(self, list_of_image):
        new_list_of_image = []
        for image in list_of_image:
            if MIN_WIDTH_HEIGHT_RATIO_FOR_NUMBER < (
                    image.intBoundingRectHeight / image.intBoundingRectWidth) < MAX_WIDTH_HEIGHT_RATIO_FOR_NUMBER:
                new_list_of_image.append(image)
        return new_list_of_image

    def find_list_of_lists_of_matching_chars_1(self, list_of_possible_chars):

        list_of_lists_of_matching_chars = []

        for possible_char in list_of_possible_chars:
            list_of_matching_chars = self.find_list_of_matching_chars_1(possible_char, list_of_possible_chars)

            list_of_matching_chars.append(possible_char)

            #####################################
            # cv2.imshow('main', possible_char.img)
            # cv2.waitKey(0)
            # for i in list_of_matching_chars:
            #     cv2.imshow(str(i),i.img)
            #     cv2.resizeWindow(str(i),200,200)
            #     cv2.waitKey(0)
            ####################################

            if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS_1_2:
                continue

            list_of_lists_of_matching_chars.append(list_of_matching_chars)

            # list_of_possible_chars_with_current_matches_removed = list(
            #     set(list_of_possible_chars) - set(list_of_matching_chars))
            #
            # recursive_list_of_lists_of_matching_chars = self.find_list_of_lists_of_matching_chars(
            #     list_of_possible_chars_with_current_matches_removed)
            #
            # for recursiveListOfMatchingChars in recursive_list_of_lists_of_matching_chars:
            #     list_of_lists_of_matching_chars.append(recursiveListOfMatchingChars)
            # break

        return list_of_lists_of_matching_chars

    def find_list_of_matching_chars_1(self, possible_char, list_of_chars):

        list_of_matching_chars = []

        for possible_matching_char in list_of_chars:
            if possible_matching_char == possible_char:
                continue

            distance_between_chars = self.distance_between_chars(possible_char, possible_matching_char)

            angle_between_chars = self.angle_between_chars(possible_char, possible_matching_char)

            change_in_area = float(
                abs(possible_matching_char.intBoundingRectArea - possible_char.intBoundingRectArea)) / float(
                possible_char.intBoundingRectArea)

            change_in_width = float(
                abs(possible_matching_char.intBoundingRectWidth - possible_char.intBoundingRectWidth)) / float(
                possible_char.intBoundingRectWidth)
            change_in_height = float(
                abs(possible_matching_char.intBoundingRectHeight - possible_char.intBoundingRectHeight)) / float(
                possible_char.intBoundingRectHeight)

            # check if chars match
            if (distance_between_chars < (possible_char.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY_1) and
                    angle_between_chars < MAX_ANGLE_BETWEEN_CHARS_1 and
                    change_in_area < MAX_CHANGE_IN_AREA_1 and
                    change_in_width < MAX_CHANGE_IN_WIDTH_1 and
                    change_in_height < MAX_CHANGE_IN_HEIGHT_1):
                list_of_matching_chars.append(
                    possible_matching_char)

        return list_of_matching_chars

    def find_list_of_lists_of_matching_chars_2(self, list_of_possible_chars):

        list_of_lists_of_matching_chars = []

        for possible_char in list_of_possible_chars:
            list_of_matching_chars = self.find_list_of_matching_chars_2(possible_char, list_of_possible_chars)

            list_of_matching_chars.append(possible_char)

            #####################################
            # cv2.imshow('main', possible_char.img)
            # cv2.waitKey(0)
            # for i in list_of_matching_chars:
            #     cv2.imshow(str(i),i.img)
            #     cv2.resizeWindow(str(i),200,200)
            #     cv2.waitKey(0)
            ####################################

            if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS_1_2:
                continue

            list_of_lists_of_matching_chars.append(list_of_matching_chars)

            # list_of_possible_chars_with_current_matches_removed = list(
            #     set(list_of_possible_chars) - set(list_of_matching_chars))
            #
            # recursive_list_of_lists_of_matching_chars = self.find_list_of_lists_of_matching_chars(
            #     list_of_possible_chars_with_current_matches_removed)
            #
            # for recursiveListOfMatchingChars in recursive_list_of_lists_of_matching_chars:
            #     list_of_lists_of_matching_chars.append(recursiveListOfMatchingChars)
            # break

        return list_of_lists_of_matching_chars

    def find_list_of_lists_of_matching_chars_3(self, list_of_possible_chars):

        list_of_lists_of_matching_chars = []

        for possible_char in list_of_possible_chars:
            list_of_matching_chars = self.find_list_of_matching_chars_2(possible_char, list_of_possible_chars)

            list_of_matching_chars.append(possible_char)

            #####################################
            # cv2.imshow('main', possible_char.img)
            # cv2.waitKey(0)
            # for i in list_of_matching_chars:
            #     cv2.imshow(str(i),i.img)
            #     cv2.resizeWindow(str(i),200,200)
            #     cv2.waitKey(0)
            ####################################

            if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS_3:
                continue

            list_of_lists_of_matching_chars.append(list_of_matching_chars)

            # list_of_possible_chars_with_current_matches_removed = list(
            #     set(list_of_possible_chars) - set(list_of_matching_chars))
            #
            # recursive_list_of_lists_of_matching_chars = self.find_list_of_lists_of_matching_chars(
            #     list_of_possible_chars_with_current_matches_removed)
            #
            # for recursiveListOfMatchingChars in recursive_list_of_lists_of_matching_chars:
            #     list_of_lists_of_matching_chars.append(recursiveListOfMatchingChars)
            # break

        return list_of_lists_of_matching_chars

    def find_list_of_matching_chars_2(self, possible_char, list_of_chars):

        list_of_matching_chars = []

        for possible_matching_char in list_of_chars:
            if possible_matching_char == possible_char:
                continue

            distance_between_chars = self.distance_between_chars(possible_char, possible_matching_char)

            angle_between_chars = self.angle_between_chars(possible_char, possible_matching_char)

            change_in_area = float(
                abs(possible_matching_char.intBoundingRectArea - possible_char.intBoundingRectArea)) / float(
                possible_char.intBoundingRectArea)

            change_in_width = float(
                abs(possible_matching_char.intBoundingRectWidth - possible_char.intBoundingRectWidth)) / float(
                possible_char.intBoundingRectWidth)
            change_in_height = float(
                abs(possible_matching_char.intBoundingRectHeight - possible_char.intBoundingRectHeight)) / float(
                possible_char.intBoundingRectHeight)

            # check if chars match
            if (distance_between_chars < (possible_char.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY_2) and
                    angle_between_chars < MAX_ANGLE_BETWEEN_CHARS_2 and
                    change_in_area < MAX_CHANGE_IN_AREA_2 and
                    change_in_width < MAX_CHANGE_IN_WIDTH_2 and
                    change_in_height < MAX_CHANGE_IN_HEIGHT_2):
                list_of_matching_chars.append(
                    possible_matching_char)

        return list_of_matching_chars

    def distance_between_chars(self, first_char, second_char):
        int_x = abs(first_char.intCenterX - second_char.intCenterX)
        int_y = abs(first_char.intCenterY - second_char.intCenterY)

        return math.sqrt((int_x ** 2) + (int_y ** 2))

    def angle_between_chars(self, first_char, second_char):
        adj = float(abs(first_char.intCenterX - second_char.intCenterX))
        opp = float(abs(first_char.intCenterY - second_char.intCenterY))

        if adj != 0.0:
            angle_in_rad = math.atan(opp / adj)
        else:
            angle_in_rad = 1.5708

        angle_in_deg = angle_in_rad * (180.0 / math.pi)

        return angle_in_deg


class DetectPlate:

    def partitioning_1(self, org_image, image, error):

        x = image.intBoundingRectX
        y = image.intBoundingRectY
        w = int(image.intBoundingRectWidth * ERROR_OF_WIDTH)
        h = int(image.intBoundingRectHeight * ERROR_OF_HEIGHT)
        cx = image.intCenterX
        cy = image.intCenterY

        list_of_image = []
        while x - (w + error * w) > -10:
            x = x - (w + error * w)
            cx = cx - (w + error * w)
            img = cv2.getRectSubPix(org_image, (w, h), (cx, cy))
            list_of_image.append(img)

        list_of_image.reverse()
        cx = image.intCenterX
        x = image.intBoundingRectX

        img = cv2.getRectSubPix(org_image, (w, h), (cx, cy))
        list_of_image.append(img)

        while x + 2 * w + error * w <= org_image.shape[1] + 10:
            x = x + (w + error * w)
            cx = cx + (w + error * w)
            img2 = cv2.getRectSubPix(org_image, (w, h), (cx, cy))
            list_of_image.append(img2)

        return list_of_image

    def partitioning_2(self, org_image, image1, possible_chars, distance):
        x1 = image1.intBoundingRectX
        y1 = image1.intBoundingRectY
        w1 = int(image1.intBoundingRectWidth * ERROR_OF_WIDTH)
        h1 = int(image1.intBoundingRectHeight * ERROR_OF_HEIGHT)
        cx1 = image1.intCenterX
        cy1 = image1.intCenterY
        area1 = image1.intBoundingRectArea

        new_x = x1
        list_of_image = []
        while new_x >= 0:
            for image2 in possible_chars:
                x2 = image2.intBoundingRectX
                y2 = image2.intBoundingRectY
                w2 = int(image2.intBoundingRectWidth * ERROR_OF_WIDTH)
                h2 = int(image2.intBoundingRectHeight * ERROR_OF_HEIGHT)
                cx2 = image2.intCenterX
                cy2 = image2.intCenterY
                area2 = image2.intBoundingRectArea

                if area1 * MIN_AREA_ERROR < area2 < area1 * MAX_AREA_ERROR and \
                        w1 * MIN_WIDTH_ERROR < w2 < w1 * MAX_WIDTH_ERROR and \
                        h1 * MIN_HEIGHT_ERROR < h2 < h1 * MAX_HEIGHT_ERROR and \
                        y1 * Y_ERROR < y2 < y1 + (2 * h1) / 3 and \
                        new_x - distance * MAX_DISTANCE_ERROR < x2 < new_x - distance * MIN_DISTANCE_ERROR:
                    list_of_image.append(image2)

            new_x -= distance

        new_x = x1
        while new_x <= org_image.shape[1]:
            for image2 in possible_chars:
                x2 = image2.intBoundingRectX
                y2 = image2.intBoundingRectY
                w2 = int(image2.intBoundingRectWidth * ERROR_OF_WIDTH)
                h2 = int(image2.intBoundingRectHeight * ERROR_OF_HEIGHT)
                cx2 = image2.intCenterX
                cy2 = image2.intCenterY
                area2 = image2.intBoundingRectArea

                if area1 * MIN_AREA_ERROR < area2 < area1 * MAX_AREA_ERROR and \
                        w1 * MIN_WIDTH_ERROR < w2 < w1 * MAX_WIDTH_ERROR and \
                        h1 * MIN_HEIGHT_ERROR < h2 < h1 * MAX_HEIGHT_ERROR and \
                        y1 * Y_ERROR < y2 < y1 + (2 * h1) / 3 and \
                        new_x + distance * MIN_DISTANCE_ERROR < x2 < new_x + distance * MAX_DISTANCE_ERROR:
                    list_of_image.append(image2)

            new_x += distance

    def main(self, org_image, image):
        list_of_list_of_image = []
        error = MIN_PARTITION_ERROR
        while error <= MAX_PARTITION_ERROR:
            list_of_list_of_image.append(self.partitioning_1(org_image, image, error))
            error += INCREASE_IN_REPETITION

        return list_of_list_of_image


class Verification:
    percent = 0.999
    counter = 50

    def is_linear(self, list_of_numbers):
        key_value = {}
        boolean_list = []
        # get key value
        for i, img_of_number in enumerate(list_of_numbers):
            key_value[i] = img_of_number.intCenterX

        # is linear or no
        a = numpy.abs(key_value[0] - key_value[1])
        x0 = key_value[0]
        for i in range(len(list_of_numbers)):
            if MIN_ERROR_VERIFICATION * key_value[i] <= a * i + x0 <= MAX_ERROR_VERIFICATION * key_value[i]:
                boolean_list.append(True)
            else:
                boolean_list.append(False)
        return boolean_list

    def number_of_numbers(self, list_of_image):
        counter = 0

        for image in list_of_image:
            # if knn.findNearest(np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32), k=5)[1]:
            # if clf_2.predict_proba(np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][
            #     10] < 1 - self.percent:
            if clf_2.predict_proba(np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[
                0][1] > self.percent:
                counter += 1
        if counter > 8:
            return 0
        else:
            return counter

    def number_of_number_2(self, list_of_list_of_image):
        list_of_count = []

        for list_ in list_of_list_of_image:
            list_of_count.append(self.number_of_numbers(list_))

        return sorted(list_of_count, reverse=True)

    def sort(self, list_of_list_of_image):
        list_of_count = self.number_of_number_2(list_of_list_of_image)
        if (list_of_count[0] == list_of_count[1] or list_of_count[1] == list_of_count[2]) and self.counter != 0 and \
                list_of_count[0] < 4:
            self.counter -= 1
            self.percent -= 0.01
            list_of_list_of_image = self.sort(list_of_list_of_image)
            return list_of_list_of_image
        else:
            list_of_list_of_image = sorted(list_of_list_of_image, key=self.number_of_numbers, reverse=True)
            return list_of_list_of_image

    def delete_false(self, list_of_image):
        new_list_of_image = []
        for image in list_of_image:
            # if knn.findNearest(np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32), k=3)[
            #     1]:
            # if clf_2.predict_proba(np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][
            #     10] < 0.9:
            #     print(
            #         clf_2.predict_proba(
            #             np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][10])
            #     new_list_of_image.append(image)
            # else:
            #     cv2.imshow(str(image), image)
            #     print(
            #         clf_2.predict_proba(
            #             np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][10])
            if clf_2.predict_proba(np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][
                1] > 0.1:
                print(
                    clf_2.predict_proba(
                        np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][1])
                new_list_of_image.append(image)
            else:
                cv2.imshow(str(image), image)
                print(
                    clf_2.predict_proba(
                        np.array(cv2.resize(image.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][1])

        return new_list_of_image


def train():
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

    # creat label for answer
    k1 = np.array([1])
    k2 = np.array([0])
    label = np.concatenate(
        (k1.repeat(702)[:, np.newaxis].astype(np.float32), k2.repeat(624)[:, np.newaxis].astype(np.float32)))

    knn.train(train, cv2.ml.ROW_SAMPLE, label)


def train2():
    # read all image
    image_list = list()
    folder_path = r'C:\Users\User\Desktop\project2\train\true_2'
    my_list = os.listdir(folder_path)

    for imPath in my_list:
        image = cv2.imread(f'{folder_path}/{imPath}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(image)

    folder_path = r'C:\Users\User\Desktop\project2\train\false_2'
    my_list = os.listdir(folder_path)

    for imPath in my_list:
        image = cv2.imread(f'{folder_path}/{imPath}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(image)

    # creat train array
    train = np.array(image_list[0].reshape(-1, 400)).astype(np.float32)
    for i in range(1, 1644):
        train = np.concatenate((train, image_list[i].reshape(-1, 400)), axis=0).astype(np.float32)

    # creat label for answer
    k1 = np.array([1])
    k2 = np.array([0])
    label = np.concatenate(
        (k1.repeat(832)[:, np.newaxis].astype(np.float32), k2.repeat(812)[:, np.newaxis].astype(np.float32)))

    x_train, x_test, y_train, y_test = train_test_split(train, label.ravel(),
                                                        test_size=10,
                                                        random_state=1230,
                                                        stratify=label.ravel())
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print('accuracy of train = %.2f' % accuracy_score(y_test, y_pred))


def train3():
    # image_list = list()
    # my_list = list()
    # folder_path = r'C:\Users\User\Desktop\project2\train'

     

    # for i in tqdm(range(11)):
    #     my_list.append(os.listdir(folder_path + '\\' + str(i)))
    
    # i = 0
    # for list_ in my_list:
    #     for imPath in tqdm(list_):
    #         image = cv2.imread(f'{folder_path}\\{i}\\{imPath}')
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         image_list.append(image)
    #     i += 1
    
    
    # image_list = list()
    # folder_path = r'C:\Users\User\Desktop\project2\train\true_2'
    # my_list = os.listdir(folder_path)

    # for imPath in tqdm(my_list):
    #     image = cv2.imread(f'{folder_path}/{imPath}')
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     image_list.append(image)

    # folder_path = r'C:\Users\User\Desktop\project2\train\10'
    # my_list = os.listdir(folder_path)

    # for imPath in tqdm(my_list):
    #     image = cv2.imread(f'{folder_path}/{imPath}')
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     image_list.append(image)

    # creat train array

    # train = np.array(image_list[0].reshape(-1, 400)).astype(np.float32)
    # for i in tqdm(range(1, 1821)):
    #     train = np.concatenate((train, image_list[i].reshape(-1, 400)), axis=0).astype(np.float32)
    train = cv2.imread(r'C:\Users\User\Desktop\project2\train\traindata.png').astype(np.float32)
    train = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

    # creat label for answer
    k1 = np.arange(10)
    k2 = np.array([10])
    
    label = np.concatenate(
        (k1.repeat(100)[:, np.newaxis].astype(np.float32), k2.repeat(821)[:, np.newaxis].astype(np.float32)))

    # k1 = np.array([1])
    # k2 = np.array([0])
    # label = np.concatenate(
    #     (k1.repeat(975)[:, np.newaxis].astype(np.float32), k2.repeat(821)[:, np.newaxis].astype(np.float32)))

    # implement
    x_train, x_test, y_train, y_test = train_test_split(train, label.ravel(),
                                                        test_size=0.25,
                                                        random_state=1230,
                                                        stratify=label.ravel())

    clf_2.fit(x_train, y_train)

    acc = accuracy_score(label.ravel(), clf_2.predict(train))

    y_pred = clf_2.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print('\nSVM Trained Classifier Accuracy: ', acc)
    print('\nAccuracy of Classifier on Validation Images: ', accuracy)


def train4():
    image_list = list()
    folder_path = r'C:\Users\User\Desktop\project2\train\true'
    my_list = os.listdir(folder_path)

    print('load data and train ...')

    for imPath in tqdm(my_list):
        image = cv2.imread(f'{folder_path}/{imPath}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(image)

    folder_path = r'C:\Users\User\Desktop\project2\train\false'
    my_list = os.listdir(folder_path)

    for imPath in tqdm(my_list):
        image = cv2.imread(f'{folder_path}/{imPath}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(image)

    # creat train array
    train = np.array(image_list[0].reshape(-1, 400)).astype(np.float32)
    for i in tqdm(range(1, 1642)):
        train = np.concatenate((train, image_list[i].reshape(-1, 400)), axis=0).astype(np.float32)

    # creat label for answer
    k1 = np.array([1])
    k2 = np.array([0])
    label = np.concatenate(
        (k1.repeat(830)[:, np.newaxis].astype(np.float32), k2.repeat(812)[:, np.newaxis].astype(np.float32)))

    # implement
    x_train, x_test, y_train, y_test = train_test_split(train, label.ravel(),
                                                        test_size=0.25,
                                                        random_state=1230,
                                                        stratify=label.ravel())

    KNN.fit(x_train, y_train)

    acc = accuracy_score(label.ravel(), KNN.predict(train))

    y_pred = KNN.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print('\nSVM Trained Classifier Accuracy: ', acc)
    print('\nAccuracy of Classifier on Validation Images: ', accuracy)
