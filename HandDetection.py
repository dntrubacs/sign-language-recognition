'''
Created by Daniel-Iosif Trubacs for the UOS AI SOCIETY on 2 January 2022.
A modules used to simplify the Hand Detection algorithm. To use just import the

'''

#importing the necessary libraries
import mediapipe as mp
import cv2 as cv


# Setting the hand detection algorithm up (using mediapipe)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False)
mpDraw = mp.solutions.drawing_utils


class HandDetection:
    def __init__(self,img):
        self.img = img

        # dimensions of the image
        self.height = int(self.img.shape[0])
        self.width = int(self.img.shape[1])

        # scaling up the image found
        self.alpha = int(0.05*self.height)
        self.beta = int(0.05*self.width)

    def highlight(self,show_hand,show_landmarks):

        #finding the landmarks in the image
        results = hands.process(self.img)

        # going through all the landmarks found
        parameters = results.multi_hand_landmarks
        if parameters:
                # the coordinates of all the landmarks found, used to find the rectangle containg the hand
                x_pos = []
                y_pos = []
                for handLms in parameters:
                    # appending all the coordinates of each landmark
                    for lm in handLms.landmark:
                        x_pos.append(lm.x)
                        y_pos.append(lm.y)
                    # showing the landmarks found in the image
                    if show_landmarks:
                        mpDraw.draw_landmarks(self.img, handLms, mpHands.HAND_CONNECTIONS)

                # the points defining the rectangle that contains the hand
                P_max = [int(self.width * min(x_pos)), int(self.height * min(y_pos))]
                P_min = [int(self.width * max(x_pos)), int(self.height * max(y_pos))]

                # a adjustesd (larger) rectangle for the hand
                P_max_0 = [int(self.width * min(x_pos)) - self.alpha, int(self.height * min(y_pos)) - self.beta]
                P_min_0 = [int(self.width * max(x_pos)) + self.alpha, int(self.height * max(y_pos)) + self.beta]

                # highlighting the region containing the hand
                if show_hand:
                 cv.rectangle(self.img, P_min_0, P_max_0, (0, 0, 255), 2)


        return img




# trial
img = cv.imread('hand0.jpeg')
img = cv.resize(img,[800,800])

Hand  = HandDetection(img)
img = Hand.highlight(show_hand=True, show_landmarks=True)



cv.imshow("cool",img)
cv.waitKey()

