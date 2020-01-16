import cv2 # opencv-python
import numpy as np
from skimage.filters import threshold_local # scikit-image
import imutils


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right poi
    # nt will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped





if __name__ == '__main__':


    image = cv2.imread('../temp_images/stas.jpg')
    original_image = image.copy()

    image = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
    cv2.imshow("bounds up",image)
    cv2.waitKey(0)

    # resize using ratio (old height to the new height)
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    cv2.imshow("yuv",image_yuv)
    cv2.waitKey(0)

    # grap only the Y component
    image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
    image_y[:, :] = image_yuv[:, :, 0]

    image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)
    cv2.imshow("gauss", image_blurred)
    cv2.waitKey(0)

    # find edges in the image
    edges = cv2.Canny(image_blurred, 50, 200, apertureSize=3)

    # find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw all contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    # !! Attention !! Do not draw contours on the image at this point
    # I have drawn all the contours just to show below image
    cv2.imshow("im+cnt",image)
    cv2.waitKey(0)
    # to collect all the detected polygons
    polygons = []

    # loop over the contours
    for cnt in contours:
        # find the convex hull
        hull = cv2.convexHull(cnt)

        # compute the approx polygon and put it into polygons
        polygons.append(cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), False))

    # sort polygons in desc order of contour area
    sortedPoly = sorted(polygons, key=cv2.contourArea, reverse=True)

    # draw points of the intersection of only the largest polyogon with red color
    cv2.drawContours(image, sortedPoly[0], -1, (0, 0, 255), 5)
    cv2.imshow("poly",image)
    cv2.waitKey(0)

    # get the contours of the largest polygon in the image
    simplified_cnt = sortedPoly[0]
    cv2.destroyAllWindows()
    # check if the polygon has four point
    #if len(simplified_cnt) >= 4:
    # trasform the prospective of original image
    cropped_image = four_point_transform(original_image, simplified_cnt.reshape(4, 2) * ratio)

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray_image, 11, offset=10, method="gaussian")
    binarized_image = (gray_image > T).astype("uint8") * 255

    # Show images
    cv2.imshow("Original", original_image)
    cv2.imshow("Scanned", binarized_image)
    cv2.imshow("Cropped", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









