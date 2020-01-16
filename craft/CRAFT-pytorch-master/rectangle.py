
import numpy as np
import matplotlib.pyplot as plt
import cv2

def resize(img, height=800):
    """ Resize image to given height """
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

if __name__ == '__main__':
    bw = False
    if not bw: path_to_image = "../temp_images/extract.jpg"
    else: path_to_image = "../temp_images/revert.jpg"


    im = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
    cv2.imshow("1",im)
    cv2.waitKey(0)

#

    rgb_planes = cv2.split(im)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    #cv2.imshow('shadows_out.png', result)
    #cv2.waitKey(0)
    cv2.imshow('shadows_out_norm.png', result_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#

    # add border
    #row, col = im.shape[:2]
    #bottom = im[row - 30:row, 0:col]
    #mean = cv2.mean(bottom)[0]

    bordersize = 30

    image = cv2.copyMakeBorder(
        result_norm,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )


    cv2.imshow("add borders",image)
    cv2.waitKey(0)


    img = cv2.cvtColor(resize(image,400), cv2.COLOR_BGR2GRAY)
    cv2.imshow("cvtcolot",img)
    cv2.waitKey(0)
    #bileteral

    img = cv2.GaussianBlur(img,(3,3),0)
    # cv2.imshow("G",img)
    # cv2.waitKey(0)

    lap = cv2.Laplacian(img, cv2.CV_64F)
    cv2.imshow("Lapl", lap)
    cv2.waitKey(0)



    #img = cv2.bilateralFilter(img,30, 10,20)
    #cv2.imshow("bilat", img)
    #cv2.waitKey(0)

    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    #cv2.imshow("ath",img)
    #cv2.waitKey(0)



    #img = cv2.GaussianBlur(img,(3,3),10)
    #cv2.imshow("G",img)
    #cv2.waitKey(0)

    #img = cv2.Laplacian(img,cv2.CV_64F)
    #cv2.imshow("Lapl", img)
    #cv2.waitKey(0)


    #img = cv2.medianBlur(img, 18)
    #cv2.imshow("6",img)
    #cv2.waitKey(0)

    #edges = cv2.Canny(img, 200, 250)
    #cv2.imshow("Can",edges)
    #cv2.waitKey(0)


    cv2.destroyAllWindows()







    #detect 4 points

    
























