import cv2
import numpy as np
import Module as m
import math

#######################################################
path = r'C:\Users\User\Desktop\project2\train\new_correct'


#######################################################


# def rotate_image(img, thr=True):
#     # __________________________________ template _______________________________________
#     temp = cv2.imread(r'C:\Users\User\Desktop\project2\image\sa.jpg')
#     gray_t = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
#     blur = cv2.medianBlur(gray_t, 3)
#     _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     template_contour = contours[0]
#     template_img = m.ProccesseImage().Shred_image([template_contour], thresh)

#     # ____________________________ original image __________________________________________
#     if thr:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         blur = cv2.medianBlur(gray, 5)
#         thresh = m.Thresh(blur).main()
#     else:
#         thresh = img
#     contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#     contours = m.Contour().delete_zeros(contours)

#     # get M in original image
#     closest_contour = []
#     box = None
#     for c in contours:
#         match = cv2.matchShapes(template_contour, c, 1, 0.0)
#         if match < 0.3:
#             rect = cv2.minAreaRect(c)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
#             cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
#             closest_contour.append(m.ProccesseImage().Shred_image([c], thresh))
#     closest_contour = sorted(closest_contour, key=lambda image: image[0].intBoundingRectArea, reverse=True)
#     try:
#         contours, _ = cv2.findContours(closest_contour[0][0].img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#         angle_target = math.atan(-1 * (box[0][1] - box[1][1]) / (box[0][0] - box[1][0])) * (180.0 / math.pi)
#         # rotate Image
#         rotation_matrix = cv2.getRotationMatrix2D((int(box[1][0]), int(box[1][1])), 90 - angle_target, 1.0)
#         img_rotated = cv2.warpAffine(img, rotation_matrix, (thresh.shape[1], thresh.shape[0]))
#         return img_rotated
#     except:
#         print('not find match in image')
#         return img


def first_attempt(img, thr=True):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img
    if thr:
        blur = cv2.medianBlur(gray, 5)
        thresh = m.Thresh(blur).main()
    else:
        thresh = img

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # delete contours that their area is 0
    contours = m.Contour().delete_zeros(contours)
    # shred image and calculate their property
    list_of_image = m.ProccesseImage().Shred_image(contours, thresh)
    # get contours that look like number
    new_list_of_image = m.DetectChar().possible_char(list_of_image)
    # get list of number that have some property
    char_in_plate = m.DetectChar().find_list_of_lists_of_matching_chars_1(new_list_of_image)

    if len(char_in_plate):
        # sort list of image based on x
        for i in range(len(char_in_plate)):
            char_in_plate[i] = sorted(char_in_plate[i], key=lambda image: image.intBoundingRectX)
        # if x's of list have no linear relationship should be delete from char_in_plate
        for i in char_in_plate:
            is_linear = m.Verification().is_linear(i)
            if not all(is_linear):
                char_in_plate.remove(i)

        return char_in_plate
    else:
        return None


def second_attempt(img, thr=True):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img
    if thr:
        blur = cv2.medianBlur(gray, 5)
        thresh = m.Thresh(blur).main()
    else:
        thresh = img

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # delete contours that their area is 0
    contours = m.Contour().delete_zeros(contours)
    # shred image and calculate their property
    list_of_image = m.ProccesseImage().Shred_image(contours, thresh)
    # get contours that look like number
    new_list_of_image = m.DetectChar().possible_char(list_of_image)
    # get list of number that have some property
    char_in_plate = m.DetectChar().find_list_of_lists_of_matching_chars_2(new_list_of_image)

    if len(char_in_plate):
        # sort list of image based on x
        for i in range(len(char_in_plate)):
            char_in_plate[i] = sorted(char_in_plate[i], key=lambda image: image.intBoundingRectX)
        # if x's of list have no linear relationship should be delete from char_in_plate
        for i in char_in_plate:
            is_linear = m.Verification().is_linear(i)
            if not all(is_linear):
                char_in_plate.remove(i)

        return char_in_plate
    else:
        return None


def third_attempt(img, thr=True):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img
    if thr:
        blur = cv2.medianBlur(gray, 5)
        thresh = m.Thresh(blur).main()
    else:
        thresh = img

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # delete contours that their area is 0
    contours = m.Contour().delete_zeros(contours)
    # shred image and calculate their property
    list_of_image = m.ProccesseImage().Shred_image(contours, thresh)
    # get contours that look like number
    new_list_of_image = m.DetectChar().possible_char(list_of_image)
    # get list of number that have some property
    char_in_plate = m.DetectChar().find_list_of_lists_of_matching_chars_3(new_list_of_image)

    if len(char_in_plate):
        # sort list of image based on x
        for i in range(len(char_in_plate)):
            char_in_plate[i] = sorted(char_in_plate[i], key=lambda image: image.intBoundingRectX)
        # if x's of list have no linear relationship should be delete from char_in_plate
        for i in char_in_plate:
            is_linear = m.Verification().is_linear(i)
            if not all(is_linear):
                char_in_plate.remove(i)

        return char_in_plate
    else:
        return None


def forth_attempt(img):
    corners, img = m.OpenCv().corner_detection(img, draw=True)
    for i in corners:
        cv2.circle(img, (i[0], i[1]), 5, (0, 255, 255), -1)

    img = m.OpenCv().crop_image_by_corner(img, corners)

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    img = m.OpenCv().line_detection(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = m.Contour().delete_zeros(contours)
    contours = m.Contour().delete_under_mean(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    list_of_image = m.ProccesseImage().Shred_image(contours, img)

    list_of_possible_image = m.DetectChar().find_list_of_lists_of_matching_chars_2(list_of_image)

    new_list_of_possible_image = []
    for image_list in list_of_possible_image:
        new_list_of_possible_image.append(sorted(image_list, key=lambda image: image.intBoundingRectX))

    new_list_of_possible_image = sorted(new_list_of_possible_image, key=lambda list_: len(list_), reverse=True)

    return new_list_of_possible_image


def fifth_attempt(img, thr=True):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = img
    if thr:
        blur = cv2.medianBlur(gray, 5)
        thresh = m.Thresh(blur).main()
    else:
        thresh = img

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # delete contours that their area is 0
    contours = m.Contour().delete_zeros(contours)
    # shred image and calculate their property
    list_of_image = m.ProccesseImage().Shred_image(contours, thresh)
    # get contours that look like number
    new_list_of_image = m.DetectChar().possible_char(list_of_image)
    # find a number
    for image in new_list_of_image:
        if m.clf_2.predict_proba(
                    np.array(cv2.resize(image.img.copy(), (20, 20)).reshape(-1, 400)).astype(np.float32))[0][
                    1] > 0.98:
            if img.shape[0] * 1 / 2 < image.intBoundingRectHeight and \
                    1 < image.intBoundingRectHeight / image.intBoundingRectWidth < 3:
                return image


def delete_red(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
    mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))

    mask = cv2.bitwise_or(mask1, mask2)
    cropped = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    try:
        [x, y, w, h] = cv2.boundingRect(contours[0])
        cropped = cv2.getRectSubPix(img, (x, thresh.shape[0]), (x / 2, thresh.shape[0] / 2))
        return cropped
    except:
        return img


def delete_black(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 255, 30])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    blur = cv2.medianBlur(mask, 3)
    dilate = cv2.dilate(blur, (2, 2), iterations=40)
    blur = cv2.medianBlur(dilate, 15)
    contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    try:
        [x, y, w, h] = cv2.boundingRect(contours[0])
        if x > img.shape[1] / 2 and h / w < 2.5 and img.shape[0] * 1 / 3 < h:
            cropped = cv2.getRectSubPix(img, (x, img.shape[0]), (x / 2, img.shape[0] / 2))
            return cropped
        else:
            return img
    except:
        return img


def main():
    m.train3()
    img = cv2.imread(r'E:\\python code\\project21\\New folder\\project\\image/1.jpg')

    img = cv2.resize(img, (290, 95))

    cropped = delete_red(img)
    cropped = delete_black(cropped)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    thresh = m.Thresh(blur).main()

    image = None

    char_in_plate = first_attempt(gray)
    if not char_in_plate:
        char_in_plate = second_attempt(gray)
    if not char_in_plate:
        char_in_plate = third_attempt(gray)
    if not char_in_plate:
        char_in_plate = forth_attempt(gray)
    if not char_in_plate:
        image = fifth_attempt(gray)

    if char_in_plate:
        try:
            list_of_list_of_image = m.DetectPlate().main(thresh, char_in_plate[0][1])
            list_of_list_of_image = m.Verification().sort(list_of_list_of_image)
            deleted = m.Verification().delete_false(list_of_list_of_image[0])
            j = 0
            # number = ''
            for i in deleted:
                cv2.imshow(str(j), i)
                cv2.resizeWindow(str(j), 200, 200)
                # number.join(str(m.clf_2.predict(np.array(cv2.resize(i, (20, 20)).reshape(-1, 400)).astype(np.float32))))
                j += 1
            # print(number)

        except:
            print('nothing')
    else:
        try:
            list_of_list_of_image = m.DetectPlate().main(thresh, image)
            list_of_list_of_image = m.Verification().sort(list_of_list_of_image)
            deleted = m.Verification().delete_false(list_of_list_of_image[0])
            j = 0
            # number = ''
            for i in deleted:
                cv2.imshow(str(j), i)
                cv2.resizeWindow(str(j), 200, 200)
                # number.join(m.clf_2.predict(np.array(cv2.resize(i, (20, 20)).reshape(-1, 400)).astype(np.float32)))
                j += 1

        except:
            print('nothing')
    cv2.imshow('img',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
