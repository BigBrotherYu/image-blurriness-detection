import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import time


# Generate interests patches
ls = []
for row in range(6):
    for columns in range(5):
        ls.append([150+row*300, 200+columns*200])

del ls[0]

# constance
PATH = r'C:\Users\Administrator\AppData\Local\Programs\Python\Python35\Lib\site-packages\cv2\data' # path to cv2 location
REFERENCE_WINDOWS = ls
WINDOWS_HEIGHT = 50
WINDOWS_WIDTH = 100


class Cell_images:

    def __init__(self, image):
        assert image.shape[0] > 0
        self.image = image
        self.secs = []
        self.gray = []

    def human_face(self):

        gray = self.gray.copy()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_cascade.load(PATH + '\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces):
            return True
        else:
            return False

    def homogeneity(self, gate=20):

        '''

        :param gate: 拉普拉斯积分布的阈值
        :return:
        '''
        image = self.image.copy()
        s = cv2.Laplacian(image[:, 367:, :], cv2.CV_64F).var()
        if s < gate:
            s2 = cv2.Laplacian(image[200:, :, :], cv2.CV_64F).var()
            if s2 < gate:
                return True
            else:
                return False
        else:
            return False

    def section(self):

        secs = []
        copy = self.image.copy()
        for (w, h) in REFERENCE_WINDOWS:
            sec = copy[h:h+WINDOWS_HEIGHT, w:w+WINDOWS_WIDTH]
            secs.append(sec)
            # cv2.rectangle(copy, (w, h), (w+WINDOWS_WIDTH, h+WINDOWS_HEIGHT), color=[255, 0, 0], thickness=5)

        # plt.imshow(copy)
        # plt.show()

        return secs

    def extreme_contrast(self, contrast_gate=200):

        if len(self.secs):
            secs = self.secs
        else:
            secs = self.section()

        sec_score = []
        for sec in secs:
            sechsv = cv2.cvtColor(sec, cv2.COLOR_BGR2HSV)
            # print(sec_score)
            brightness = sechsv[:, :, 2].flatten()
            sec_score.append(np.mean(brightness))

        contrast = max(sec_score) - min(sec_score)
        # print(contrast, "gao", max(sec_score), min(sec_score))

        if contrast > contrast_gate:
            return True
        else:
            return False

    def edge_ambiguity(self, canny_gate=150, number_of_areas=2):

        '''
        判断边缘清晰情况
        :param canny_gate: canny 的阈值
        :param number_of_areas: 分析点的阈值
        :return:
        '''

        if len(self.secs):
            pass
        else:
            self.secs = self.section()

        secs = self.secs
        sec_score = 0
        for sec in secs:

            if sec.any():
                pass
            else:
                continue
            sec = cv2.cvtColor(sec, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(sec, 55, canny_gate)
            # plt.imshow(edges, cmap='gray')
            # plt.show()
            if edges.any():
                sec_score = sec_score + 1
            else:
                pass

        if sec_score > number_of_areas:
            # print('clear')
            return False
        else:
            return True

    def low_brightness(self, brightness_gate=50):

        '''
        :param brightness_gate: 黑度阀值， 越高删除的越多
        :return:
        '''

        image = self.image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        all_value = hsv[:, :, 2].flatten()
        brightness_score = np.mean(all_value)
        if brightness_score < brightness_gate:
            return True
        else:
            return False

    def detectdirt(self):
        if self.low_brightness():
            return 1
        else:
            if self.extreme_contrast():
                return 2
            else:
                if self.homogeneity():
                    return 3
                else:
                    if self.edge_ambiguity():
                        return 4
                    else:
                        return 0
                        # if self.human_face():
                        #     return 5
                        # else:
                        #     return 0

    def aHash(self, hash_size=(16, 16)):
        image = self.image.copy()
        small = cv2.cvtColor(cv2.resize(image, hash_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        average = np.mean(small)
        imhash = []
        for i in range(hash_size[0]):
            for j in range(hash_size[1]):
                if small[i, j] > average:
                    imhash.append(1)
                else:
                    imhash.append(0)
        return imhash
        