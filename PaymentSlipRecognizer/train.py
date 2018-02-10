import sys
import numpy as np
import cv2


print 'Start traning!'

img_slip_train = cv2.imread('/Users/nikolaspiric/PycharmProjects/ElektronskeUplatnice/upl1.png')  #load from disk
img_slip_train = cv2.cvtColor(img_slip_train, cv2.COLOR_BGR2RGB)    #convert to RGB
#print img_slip_train.shape     #size of image

img_slip_train_gray = cv2.cvtColor(img_slip_train, cv2.COLOR_RGB2GRAY)   #convert to gray
#cv2.imshow('image_gray', img_slip_train_gray)     #print image for test
#cv2.waitKey(0)

img_slip_train_thres = cv2.adaptiveThreshold(img_slip_train_gray, 255, 1, 1, 11, 2)   #adaptivThreshold
#cv2.imshow('image_threshold', img_slip_thres)     #print image for test
#cv2.waitKey(0)

img, contours, hierarchy = cv2.findContours(img_slip_train_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     #find contours
#img = img_slip_train.copy()        #print image with countors for test
#cv2.drawContours(img, contours, - 1, (255, 0, 0), 1)
#cv2.imshow('image_countours', img)
#cv2.waitKey(0)


find_countours = np.empty((0, 100))
number_countours = []
keys = [i for i in range(48, 58)]


for contour in contours:
    if cv2.contourArea(contour) > 8 and cv2.contourArea(contour) < 55:
        [x, y, w, h] = cv2.boundingRect(contour)
        if y > 25 and x > 250:
            if h > 10:
                cv2.rectangle(img_slip_train, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi = img_slip_train_thres[y:y+h, x:x+w]        #cut
                roismall = cv2.resize(roi, (10, 10))            #cut
                cv2.imshow('train_slip', img_slip_train)
                key = cv2.waitKey(0)

                if key == 27:
                    sys.exit()
                elif key in keys:
                    number_countours.append(int(chr(key)))
                    #cv2.imwrite('number' + chr(key) + '.png', roismall)            #save cut image
                    find_countour = roismall.reshape((1, 100))
                    find_countours = np.append(find_countours, find_countour, 0)

number_countours = np.array(number_countours, np.float32)
number_countours = number_countours.reshape(number_countours.size, 1)

print 'Traning is done!'

#print len(number_countours)

#making train data
np.savetxt('number_contour_train.data', find_countours)
np.savetxt('number_countours.data', number_countours)



#learn from
#https://gist.github.com/yosemitebandit/5295069
#https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python