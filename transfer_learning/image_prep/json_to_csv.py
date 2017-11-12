
import os
import glob
import pandas as pd
import json
import cv2


def json_to_csv(path,path_img):

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    json_df = pd.DataFrame(columns=column_name)
    ls_json = glob.glob(path + '/*.json')
    nr_json = len(ls_json)

    print('Found '  + str(nr_json) + ' json files in ' + path)
    i = 0

    for json_file in ls_json:
        if not (i % 1000):
            print(str(i)+'/'+str(nr_json))
        i = i + 1

        json_file = open(json_file)
        json_str = json_file.read()
        json_data = json.loads(json_str)

        # we read which image the json corresponds to
        img_name = json_data['imagename']

        # the bounding boxes for each cyclist are saved as children dicts
        for cyc in json_data['children']:

            # maybe not all are cyclists, them we want to skip
            if not cyc['identity'] == 'cyclist':
                #print("found" + cyc['identity'] + ', skipping...')
                continue

            # in case the dict has the entry powerassist it has to be delted
            if 'powerassist' in cyc:
                continue

            # we see if the image can be accessed
            try:
                img = cv2.imread(os.path.join(path_img, img_name))
            except:
                print('failed to open image ' + img_name)
                continue

            # we open the image ant extract the roi
            tmp_df = pd.DataFrame({'filename': img_name,
                      'width': img.shape[1],
                      'height': img.shape[0],
                      'class': 'cyclist',
                      'xmin': cyc['mincol'],
                      'ymin': cyc['minrow'],
                      'xmax': cyc['maxcol'],
                      'ymax': cyc['maxrow']}, columns=column_name, index=[0])

            json_df = json_df.append(tmp_df,ignore_index=True)

        json_file.close()

    return json_df

if __name__ == '__main__':
    path_labels = r'/mnt/427149F311EAC541/td_raw/labelData/train/tsinghuaDaimlerDataset'
    path_img =  r'/mnt/427149F311EAC541/td_raw/leftImg8bit/train/tsinghuaDaimlerDataset'
    json_df = json_to_csv(path_labels, path_img)
    print(json_df.head())
    print(json_df.shape)