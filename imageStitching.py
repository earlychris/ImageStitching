import cv2 as cv
import numpy as np
import glob
#import imutils

imgpath = glob.glob('./*.jpg')
images = []

for image in imgpath:
    img = cv.imread(image)
    images.append(img)
    cv.imshow("Image", img)
    cv.waitKey(0)

imageStitcher = cv.Stitcher_create()

error, stitched_img = imageStitcher.stitch(images)

if not error:

    cv.imwrite("stitchedOutput.png", stitched_img)
    cv.imshow("Stiched Image", stitched_img)
    cv.waitKey(0)