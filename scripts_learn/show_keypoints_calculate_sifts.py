
## Na danym obrazku znajduje 'key points' i je na nim maluje
## Liczy tez deskryptory w tych miejscach (128-wymiarowe punkty)
## -- nie sa tutaj do niczego wykorzystywane (ale sa potrzebne do Bag of Words/Features)
##



import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


## SIFT byl w opencv, potem go nie bylo (prawne rzeczy), teraz jest znowu
## dziala w opencv 4.4.0
##
## Mozna sprawdzic wersje opencv:
## print(cv2.__version__)


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--image', default="datasets_raw/pics/IM.png", required=True, help='input image (default: %(default)s)')
    args = parser.parse_args()
    
    return  args.image
 
image_file = ParseArguments()

# wczytaj plik
img = cv2.imread(image_file)


# img grayscale:
img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# stworz "detektor"
detector = cv2.SIFT_create()

#liczymy 'key points' i deskryptory
kp,des =    detector.detectAndCompute(img_gray,None)

# Uwaga: do rysowania punktow kluczowych wystarczyloby
# kp = detector.detect(gray,None)

# narysuj kp na obrazku 
img_with_kp = cv2.drawKeypoints(img_gray,kp,img)


print("Liczba znalezionych punktow kluczowych: ", len(kp))
print("Rozmiar deskryptorow: ", des.shape)

plt.imshow(img_with_kp)
plt.show()
