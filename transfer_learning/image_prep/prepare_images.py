# This script copies everything into a single folder (labels, images) and calls
# a script to generate the tfrecords in the subfolder train and test

import sys
sys.path.append('../../')
from transfer_learning.image_prep.dir_to_tfrecords import main as dir_to_tfr
import os
import shutil
from  transfer_learning.image_prep.histogram_equalization import main as hist_equalizer



def main(list_path_labels, list_path_img, path_out, clean, hist_eq, n_test):
    for p in list_path_labels:
        if not os.path.exists(p):
            raise ValueError(p + ' not found')

    for p in list_path_img:
        if not os.path.exists(p):
            raise ValueError(p + ' not found')

    if os.path.exists(path_out):
        raise ValueError(path_out + ' already exists. Delete it before running this code')

    if not os.path.exists(path_out):
        os.makedirs(path_out)


    for path_img, path_labels in zip(list_path_img,list_path_labels):
        print('starting to copy from ' + path_img + ' to ' + path_out)
        for img_name in os.listdir(path_img):
            if not os.path.isdir(os.path.join(path_img, img_name)):
                shutil.copy2(os.path.join(path_img, img_name), path_out)
        print('Copied from ' + path_img + ' to ' + path_out)

        # in case the labels and img folder are the same we dont have to copy anything
        if path_img != path_labels:
            for label_name in os.listdir(path_labels):
                if not os.path.isdir(os.path.join(path_labels, label_name)):
                    shutil.copy2(os.path.join(path_labels, label_name), path_out)
            print('Copied from ' + path_labels + ' to ' + path_out)

    print('finished with copy')

    if hist_eq:
        print('Performing histogram Equalization inplace in ' + path_out)
        hist_equalizer(path_out, inplace=True, show=False)

    dir_to_tfr(path_out, clean=clean, n_test=n_test)

    # Clean the pipeline folder from the images. content is in train and test
    # define endings for pictures
    im_types = ('png', 'jpg', 'jpeg', 'bmp')
    if clean:
        for fname in os.listdir(path_out):
            if fname.endswith(im_types):
                os.remove(os.path.join(path_out, fname))

if __name__ == '__main__':
    list_path_labels = [r'/mnt/427149F311EAC541/td_raw/labelData/train/tsinghuaDaimlerDataset',
                        r'/mnt/427149F311EAC541/MEGA/bike_cam/data/cycl_train_label/yt']
    list_path_img = [r'/mnt/427149F311EAC541/td_raw/leftImg8bit/train/tsinghuaDaimlerDataset',
                     r'/mnt/427149F311EAC541/MEGA/bike_cam/data/cycl_train_label/yt']
    path_out = r'/mnt/427149F311EAC541/pipeline'
    clean = True
    hist_eq = True
    n_test = 100
    main(list_path_labels, list_path_img, path_out, clean, hist_eq, n_test )








