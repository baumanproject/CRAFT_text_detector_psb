# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import pytesseract

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            one_line=[]
            isNew = True
            row_num=0
            prev_y_min = 0
            prev_y_max = 100000
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)
                ################################BEGIN OCR
                poly = poly.reshape(-1, 2)

                x_l = [i[0] for i in poly]
                y_l = [i[1] for i in poly]
                x_min,x_max,y_min,y_max = min(x_l)-2,max(x_l)+2,min(y_l)-2,max(y_l)+2

                if row_num != 0:
                    check_left = one_line[0][-1][1][1][0]   #[first_line][last_element][poly_coords][2_point_RT][x]
                    if x_min >= check_left:
                       # continue
                        pass

                ROI = img[y_min:y_max,x_min:x_max]
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ROI = cv2.resize(ROI, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
                ROI = cv2.blur(ROI, (2, 2))
                ROI = cv2.bilateralFilter(src=ROI, d=170, sigmaColor=30, sigmaSpace=20) #поиграть с фильтром
                #ROI = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



                #if poly[2][1]-poly[0][1] > poly[2][0]-poly[0][0]:
                 #   M = cv2.getRotationMatrix2D((ROI.shape[0]/2,ROI.shape[1]/2),90,1)
                  #  ROI = cv2.warpAffine(ROI,M,(ROI.shape[0],ROI.shape[1]))    #ЗДЕСЬ ЕБАЛА С ПОВОРОТОМ ХОТЯ ОН ПРАВИЛЬНЫЙ
                config = ('-l rus --oem 1 --psm 8')
                rec_str = pytesseract.image_to_string(ROI,config = config)

                if(isNew):
                    one_line.append([])
                    one_line[row_num].append([rec_str,poly])
                    isNew = False
                else:
                    if ((prev_y_min <= y_min <= prev_y_max) or
                        (prev_y_min <= y_max <= prev_y_max) or
                        (y_min <= prev_y_min and y_max >= prev_y_max)):
                        one_line[row_num].append([rec_str,poly])
                    else:
                        one_line.append([])
                        row_num-=-1
                        one_line[row_num].append([rec_str,poly])
                prev_y_min,prev_y_max = y_min,y_max


                #cv2.imshow("roi",ROI)
                #cv2.waitKey(0)

                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

                #JSON CODE HERE

                #


                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        for i in range(row_num+1):
            for j in range(len(one_line[i])):
                print(one_line[i][j][0]+" | ")
            print("\n")

        # Save result image
        cv2.imwrite(res_img_file, img)

