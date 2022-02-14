'''
Created by Daniel-Iosif Trubacs for the UoS AI society on 2 January 2022. Aiming to use a trained Conv-Net model and a hand
detection algorithm from mediapipe to for sign recognition. To be used with the SRmodel, ModelEvaluation and HandDetection modules.
'''



# importing the necessary libraries
import mediapipe as mp
import cv2 as cv
from tensorflow import keras

#loading the trained model
sr_model = keras.models.load_model('sign_recongiton_model')

#checking if the model has been loaded succesfully
if sr_model:
    print("The model has been loaded  succesfully ")

# getting the live feed from a web site (created using IP web cam on adroid)
capture = cv.VideoCapture('http://10.14.132.177:8080/video')
#checking if the web site is created
if capture:
    print("Live feed set")

# Setting the hand detection algortihm up (using mediapipe)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False)
mpDraw = mp.solutions.drawing_utils

# set to True to start the detection
start = True

# Reading the live feed
while start:
    # getting the original frame and storring the objects found
    A, frame = capture.read()

    # the dimensions of the frame (needed for some image manipulation later)
    height = int(frame.shape[0])
    width = int(frame.shape[1])

    # changing to RGB for hand detection algorithm (required by media pipe)
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(img)

    # going thorugh all the landmarks found
    parameters = results.multi_hand_landmarks
    if parameters:
            # the coordinates of all the landmarks found
            x_pos = []
            y_pos = []
            for handLms in parameters:
                for lm in handLms.landmark:
                    x_pos.append(lm.x)
                    y_pos.append(lm.y)
                #uncomment to highlight the landmarks found
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # the points defining the rectangle that cointains the hand
            P_max = [int(width * min(x_pos)), int(height * min(y_pos))]
            P_min = [int(width * max(x_pos)), int(height * max(y_pos))]

            alpha = int(0.05 * width)
            beta = int(0.05 * height)

            # a better rectangle for the hand
            P_max_0 = [int(width * min(x_pos)) - alpha, int(height * min(y_pos)) - beta]
            P_min_0 = [int(width * max(x_pos)) + alpha, int(height * max(y_pos)) + beta]


            #uncommnet to highlight the hand in the iamge
            cv.rectangle(frame, P_min, P_max, (0, 255, 0), 2)
            cv.rectangle(frame, P_min_0, P_max_0, (0, 0, 255), 2)
            cv.putText(frame, 'P_max', P_max, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'P_min', P_min, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'P_max', P_max_0, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame, 'P_min', P_min_0, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)



        # preparing the image to feed into the neural net
        # expected shape is (1,28,28,1)

        #croping the image to get only the hand
        img = frame[P_min_0[0]:P_max_0[0],P_min_0[1]:P_max_0[1]]

        # changing the image to gray
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # bluring the image
        img = cv.blur(img, (3, 3))

        # resizing the image to (28,28)
        img = cv.resize(img, (28,28), interpolation=cv.INTER_AREA)

        cv.imshow("Live feed", frame)



        #cv.imshow("Live Feed", img)

        if cv.waitKey(1) == ord('q'):
            break



capture.release()
cv.destroyAllWindows()