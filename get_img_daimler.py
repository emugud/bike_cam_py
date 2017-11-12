import numpy as np
import cv2
import json
import os
import glob

path_labels = r'/home/eg/Downloads/labelData/train/tsinghuaDaimlerDataset'
path_img = r'/home/eg/Downloads/leftImg8bit/train/tsinghuaDaimlerDataset'
path_out = r'/home/eg/Dropbox/bike_cam/cycl_train/td'

os.chdir(path_out)
os.chdir(path_img)
os.chdir(path_labels)

ls_json = glob.glob(path_labels + '/*.json')
ls_json = ls_json
nr_json = len(ls_json)


n = 0
counter = 0
for j in ls_json:
    counter = counter + 1
    # we open the json file
    os.chdir(path_labels)
    json_file = open(j)
    json_str = json_file.read()
    json_data = json.loads(json_str)

    # we read which image the json corresponds to
    img_name = json_data['imagename']

    # we open the image
    os.chdir(path_img)
    try:
        img = cv2.imread(img_name)
    except:
        print('failed to open image ' + img_name)
        continue
    # the bounding boxes for each cyclist are saved as children dicts
    for cyc in json_data['children']:
        # maybe not all are cyclists, them we want to skip
        if not cyc['identity'] == 'cyclist':
            print("found" + cyc['identity'] + ', skipping...')
            continue

        # in case the dict has the entry powerassist it has to be delted
        if 'powerassist' in cyc:
            continue

        # we open the image ant extract the roi
        n = n + 1
        output_name = img_name[23:-4] + '_' + str(n) + '.png'
        roi = img[cyc['minrow']:cyc['maxrow'],cyc['mincol']:cyc['maxcol']]


        # and show what was found
        try:
            cv2.imshow('image', roi)
            cv2.waitKey(50)
            # save the image
            os.chdir(path_out)
            cv2.imwrite(output_name, roi)
        except:
            print('failed to fetch roi in image ' + img_name)
            continue
    json_file.close()
    print("json nr " + str(counter) + ' of ' + str(nr_json))


cv2.destroyAllWindows()