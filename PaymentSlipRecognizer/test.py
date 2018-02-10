import sys
import numpy as np
import  cv2
import operator


class BankSlip:

    def __init__(self, account, amount):
        self.account = account
        self.amount = amount


bank_acc1 = BankSlip('333-2212567678010-17', 100)
bank_acc2 = BankSlip('314-8981205454901-67', 200)
bank_acc3 = BankSlip('871-1234567891234-77', 300)

all_acc = []
all_acc.append(bank_acc1)
all_acc.append(bank_acc2)
all_acc.append(bank_acc3)


find_countours = np.loadtxt('number_contour_train.data', np.float32)
number_countours = np.loadtxt('number_countours.data', np.float32)
number_countours = number_countours.reshape(number_countours.size, 1)

knn = cv2.ml.KNearest_create()
knn.train(find_countours, cv2.ml.ROW_SAMPLE, number_countours)

for i in range(1, 4):

    print ""
    print "Racun broj: " + str(i)

    img_slip = cv2.imread('/Users/nikolaspiric/PycharmProjects/ObradaUplatnica/upl' + str(i) + '.png')
    img_slip = cv2.cvtColor(img_slip, cv2.COLOR_BGR2RGB)
    img_slip_gray = cv2.cvtColor(img_slip, cv2.COLOR_RGB2GRAY)
    img_slip_thres = cv2.adaptiveThreshold(img_slip_gray, 255, 1, 1, 11, 2)
    img, contours, hierarchy = cv2.findContours(img_slip_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    price = {}
    bank = {}
    model = {}

    for contour in contours:
        if cv2.contourArea(contour) > 8 and cv2.contourArea(contour) < 55:
            [x, y, w, h] = cv2.boundingRect(contour)
            if y > 25 and x > 250:
                if h > 10:
                    cv2.rectangle(img_slip, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    roi = img_slip_thres[y:y+h, x:x+w]  # cut
                    roismall = cv2.resize(roi, (10, 10))  # cut
                    roismall = roismall.reshape((1, 100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = knn.findNearest(roismall, k=1)
                    result = int((results[0][0]))
                    str_result = str(result)

                    if y < 100:
                        price[x] = result

                    if y > 100 and y < 150:
                        bank[x] = result

                    if y > 150 and y < 200:
                        model[x] = result

    cv2.imshow('slip', img_slip)

    #sort on x
    sort_price = sorted(price.items(), key=operator.itemgetter(0))
    sort_bank = sorted(bank.items(), key=operator.itemgetter(0))
    sort_model = sorted(model.items(), key=operator.itemgetter(0))

    #recognize real price
    price_x = sort_price[0][0]
    price_str = ""
    for j in sort_price:
        if(j[0] - price_x) > 20:
            price_str = price_str + "."
        price_str =price_str + str(j[1])
        price_x = j[0]

    print '    Iznos: ' + price_str


    #recognice real bank number
    bank_x = sort_bank[0][0]
    bank_str = ""
    for j in sort_bank:
        if(j[0] - bank_x) > 20:
            bank_str = bank_str + "-"
        bank_str =bank_str + str(j[1])
        bank_x = j[0]

    print "    Racun: " + bank_str

    for bank_acc in all_acc:
        if bank_acc.account == bank_str:
            bank_acc.amount = bank_acc.amount + float(price_str)
            print "    Novi iznos: " + str(bank_acc.amount)


    # recognice real model number
    model_x = sort_bank[0][0]
    model_str = ""
    for j in sort_model:
        if (j[0] - model_x) > 20:
            model_str = model_str + "-"
        model_str = model_str + str(j[1])
        model_x = j[0]

    print "    Model: " + model_str

    cv2.waitKey(0)

print ""
for bank_acc in all_acc:
    print "Racun: " + bank_acc.account + "   Iznos: " + str(bank_acc.amount)