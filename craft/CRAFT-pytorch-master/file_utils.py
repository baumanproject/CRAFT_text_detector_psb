# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import pytesseract
import json

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py


def rotate_doc(path):
    image = cv2.imread(path)
    #cv2.imshow("Before", image)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(tresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    Matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_doc = cv2.warpAffine(image, Matrix, (width, height),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    #cv2.imshow("TEMP", rotated_doc)
    #cv2.waitKey(0)

    ######## Пост обртаботка

    gray = cv2.cvtColor(rotated_doc, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(tresh > 0))
    ret_val = cv2.minAreaRect(coords)

    # это для первой страницы паспорта
    if np.abs((ret_val[0][1] - ret_val[1][1])) < 20:
        (height, width) = rotated_doc.shape[:2]
        center = (width // 2, height // 2)
        Matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated_doc = cv2.warpAffine(rotated_doc, Matrix, (width, height),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

    #cv2.imshow("temp_90", rotated_doc)
    #cv2.waitKey(0)

    #cv2.imshow("temp_after", rotated_doc)
    #cv2.waitKey(0)

    #if "Дата" not in text or "СВИДЕТЕЛЬСТВО" not in text:
     #   (height, width) = rotated_doc.shape[:2]
      #  center = (width // 2, height // 2)
       # Matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
        #rotated_doc = cv2.warpAffine(rotated_doc, Matrix, (width, height),
         #                            flags=cv2.INTER_CUBIC,
          #                           borderMode=cv2.BORDER_REPLICATE)

    # cv2.imshow("input",image)
    #cv2.imshow("after", rotated_doc)
    #cv2.waitKey(0)
    return rotated_doc



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

def ocr_doc(img, poly,row_num,isNew,prev_y_min,prev_y_max):
    pass

def saveResult(stringson, img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = rotate_doc(img_file)
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
            doc_type = "СНИЛС"
            counter = 1

            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

################################## BEGIN OCR

                poly = poly.reshape(-1, 2)

                x_l = [i[0] for i in poly]
                y_l = [i[1] for i in poly]
                x_min,x_max,y_min,y_max = min(x_l)-3,max(x_l)+3,min(y_l)-3,max(y_l)+3

                if row_num != 0:
                    check_left = one_line[0][-1][1][1][0]   #[first_line][last_element][poly_coords][2nd_point_RT][x]
                    if x_min >= check_left:
                        pass

                config = ('-l rus --oem 1 --psm 8')
                ROI = img[y_min:y_max,x_min:x_max]
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ROI = cv2.blur(ROI, (2, 2))
                ROI = cv2.bilateralFilter(src=ROI, d=175, sigmaColor=30, sigmaSpace=20)
                ROI = cv2.resize(ROI, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)

                # Rotation for passport number

                if doc_type != "СНИЛС":
                    if y_max-y_min > x_max-x_min :
                        if counter == 4:
                            continue
                        else:
                            ROI = cv2.transpose(ROI)
                            ROI = cv2.flip(ROI, flipCode=0)
                            string = pytesseract.image_to_string(ROI,config = config)
                            if string.isnumeric():
                                print("turn")
                                print(string)
                                rec_str = string
                                counter-=-1
                                #добавить в номер паспорта сразу (json)
                            else:
                                continue



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

                #cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

                # END OCR

                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)


        if doc_type == "СНИЛС":
            formal_words = ["российская","федерация","свидетельство","обязательного", "пенсионного","государственного",
                            "страхования","страховое","ф.и.о.","фио","дата", "место", "рождения","пол","регистрации"]
            all_words = []
            for i in one_line:
                for j in i:
                    flag = True
                    for k in formal_words:
                        if k in j[0].lower():
                            flag = False
                        if j[0].lower() == 'и':
                            flag = False
                    if flag:
                        all_words.append(j[0].replace("‘",""))

            print(all_words)

            stringson["client_mail_id"][doc_type]["Серия и номер"] = all_words[0]+" "+all_words[1]
            stringson["client_mail_id"][doc_type]["Фамилия"] = all_words[2]
            stringson["client_mail_id"][doc_type]["Имя"] = all_words[3]
            stringson["client_mail_id"][doc_type]["Отчество"] = all_words[4]

            lst = all_words[5:9]
            stringson["client_mail_id"][doc_type]["Дата рождения"] = [0,0,0,0]
            while len(lst) != 0:
                for k in lst:
                    if len(k) == 1 or len(k) == 2:
                        stringson["client_mail_id"][doc_type]["Дата рождения"][0] = k
                        lst.remove(k)
                        break
                    if k == "года":
                        stringson["client_mail_id"][doc_type]["Дата рождения"][-1] = k
                        lst.remove(k)
                        break
                    if k.isnumeric() and len(k) ==4 :
                        stringson["client_mail_id"][doc_type]["Дата рождения"][-2] = k
                        lst.remove(k)
                        break
                    else:
                        stringson["client_mail_id"][doc_type]["Дата рождения"][1] = k
                        lst.remove(k)
                        break
            stringson["client_mail_id"][doc_type]["Дата рождения"] = " ".join(stringson["client_mail_id"][doc_type]["Дата рождения"])

            fl =True
            cnt = 9
            stringson["client_mail_id"][doc_type]["Место рождения"] = ""
            while fl:
                if all_words[cnt] != "мужской":
                    if all_words[cnt] != "женский":
                        stringson["client_mail_id"][doc_type]["Место рождения"] += all_words[cnt]+" "
                        cnt-=-1
                        continue
                fl = False

            lst = all_words[-4:]
            stringson["client_mail_id"][doc_type]["Дата регистрации"] = [0, 0, 0, 0]
            while len(lst) != 0:
                for k in lst:
                    if len(k) == 1 or len(k) == 2:
                        stringson["client_mail_id"][doc_type]["Дата регистрации"][0] = k
                        lst.remove(k)
                        break
                    if k == "года":
                        stringson["client_mail_id"][doc_type]["Дата регистрации"][-1] = k
                        lst.remove(k)
                        break
                    if k.isnumeric() and len(k) == 4:
                        stringson["client_mail_id"][doc_type]["Дата регистрации"][-2] = k
                        lst.remove(k)
                        break
                    else:
                        stringson["client_mail_id"][doc_type]["Дата регистрации"][1] = k
                        lst.remove(k)
                        break
            stringson["client_mail_id"][doc_type]["Дата регистрации"] = " ".join(
                stringson["client_mail_id"][doc_type]["Дата регистрации"])


            stringson["client_mail_id"][doc_type]["Пол"] = all_words[-5]

        with open("../MVP/output2.json","w") as outfile:
            json.dump(stringson,outfile,ensure_ascii=False,indent = 4)

        # Print recognised text
        for i in range(row_num+1):
            for j in range(len(one_line[i])):
                print(one_line[i][j][0]+" | "+str(one_line[i][j][1][3][1]-one_line[i][j][1][0][1]))
            print("\n")

        # Save result image
        cv2.imwrite(res_img_file, img)

