import numpy as np
import cv2 as cv
import imutils
from stitchedpanorama import Stitcher

print("Starting Cameras..")

leftStream = cv.VideoCapture(0)
rightStream = cv.VideoCapture(1)

stitcher = Stitcher()


while True:
    # grab the frames from their respective video streams
    succL, left = leftStream.read()
    succR, right = rightStream.read()
    # resize the frames
    left = cv.resize(left, (left.shape[0], 400))
    right = cv.resize(right, (right.shape[0], 400))
    # stitch the frames together to form the panorama
    # IMPORTANT: you might have to change this line of code
    # depending on how your cameras are oriented; frames
    # should be supplied in left-to-right order
    result = stitcher.stitch([left, right])
    # no homograpy could be computed
    if result is None:
        print("[INFO] homography could not be computed")
        break

    # draw the matches
    # detect keypoints and extract
    (kpsA, featuresA) = stitcher.detectAndDescribe(left)
    (kpsB, featuresB) = stitcher.detectAndDescribe(right)
    # match features between the two images
    Match = stitcher.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio = 0.75, reprojThresh = 4.0)
    # if the match is None, then there aren't enough matched keypoints to create a panorama
    if Match is None:
        print("There are not enough Matches")
    matches, status = Match[0], Match[2]
    drawMatches = stitcher.drawMatches(left,right,kpsA, kpsB, matches, status)
    # show the output images
    cv.imshow("Result", result)
    cv.imshow("Matches", drawMatches)
    #cv.imshow("Left Frame", left)
    #cv.imshow("Right Frame", right)
    key = cv.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

#cleanup
print("[INFO] cleaning up...")
cv.destroyAllWindows()
