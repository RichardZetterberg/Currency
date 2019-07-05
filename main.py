import cv2
import numpy as np
import locale
import math

#defining UTF-8 "rus"
locale.setlocale(locale.LC_ALL, 'ru_RU')

orb = cv2.ORB_create()
global frame
training_set = ['files/1.jpg', 'files/2.jpg', 'files/5.jpg', 'files/10.jpg', 'files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/200.jpg', 'files/500.jpg']

print("starting video capture...")
# start the video capture
cap = cv2.VideoCapture(0)

# kNN creation
knn = cv2.ml.KNearest_create()

def recognize(descriptor, valueName):
    frame = 0
    count = 0
    valueСenter = (0, 0)
    for h, des in enumerate(descriptor):
        #do reshaping of the each element of the array
        des = np.array(des, np.float32).reshape(1, len(des))
        retval, results, neigh_resp, dists = knn.findNearest(des, 1)
        res, distance = int(results[0][0]), dists[0][0]

        # Draw matched key points on original image
        x, y = kp[res].pt
        center = (int(x), int(y))

        if distance < 0.1:
            # draw matched keypoints in red color
            valueCenter = center
            color = (0, 0, 255)
            count += 1
        else:
            # draw unmatched in blue color
            # print distance
            color = (255, 0, 0)

        # Draw matched key points on original image
        cv2.circle(frame, center, 2, color, -1)

    # if 50% of the poins matches, write in the bill
    if float(count) / len(descriptor) >= 0.5:
        cv2.putText(frame, ">>" + valueCenter + "<<", valueCenter, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

#going to the next string
print ("\n")

#creating data set
temp_01 = cv2.imread(training_set[0]);
temp_02 = cv2.imread(training_set[1]);
temp_05 = cv2.imread(training_set[2]);
temp_10 = cv2.imread(training_set[3]);
temp_20 = cv2.imread(training_set[4]);
temp_50 = cv2.imread(training_set[5]);
temp_100 = cv2.imread(training_set[6]);
temp_200 = cv2.imread(training_set[7]);
temp_500 = cv2.imread(training_set[8]);

# get keypoints and descriptors for each template
key_01, desc_01 = orb.detectAndCompute(temp_01, None)
key_02, desc_02 = orb.detectAndCompute(temp_02, None)
key_05, desc_05 = orb.detectAndCompute(temp_05, None)
key_10, desc_10 = orb.detectAndCompute(temp_10, None)
key_20, desc_20 = orb.detectAndCompute(temp_20, None)
key_50, desc_50 = orb.detectAndCompute(temp_50, None)
key_100, desc_100 = orb.detectAndCompute(temp_100, None)
key_200, desc_200 = orb.detectAndCompute(temp_200, None)
key_500, desc_500 = orb.detectAndCompute(temp_500, None)


while(True):
    #capturing frames from the video stream
    ret, frame = cap.read()

    #converting video stream to grey
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the keypoints with ORB with a grey frame
    (kp, des) = orb.detectAndCompute(greyFrame, None)

    # compute the descriptors with ORB
    #kp, des = orb.compute(greyFrame, kp)

    # Setting up samples and responses for kNN. Len() - number of bytes in string
    samples = np.array(des)
    #Return evenly spaced values within a given interval
    responses = np.arange(len(kp), dtype=np.float32)

    # kNN training
    knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # regognize value acording to the descriptor
    recognize(desc_01, "Одна гривня")
    recognize(desc_02, "Дві гривні")
    recognize(desc_05, "П'ять гривень")
    recognize(desc_10, "Десять гривень")
    recognize(desc_20, "Двадцять гривень")
    recognize(desc_50, "П'ятдесят гривень")
    recognize(desc_100, "Сто гривень")
    recognize(desc_200, "Двісті гривень")
    recognize(desc_500, "П'ятсот гривень")

    # print a new line
    print("\n")

    #displaying the result
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
        cv2.imread('frame.png',frame)
        break
#After all - releasing
cap.release()
cv2.destroyAllWindows()