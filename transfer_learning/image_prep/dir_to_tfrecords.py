import sys
sys.path.append('../../')

from transfer_learning.image_prep import xml_to_csv
from transfer_learning.image_prep import json_to_csv
import os
from sklearn.cross_validation import train_test_split
import numpy as np
import shutil

import io
import pandas as pd
import tensorflow as tf
import transfer_learning.image_prep.generate_tfrecord as gen_tfr
from  transfer_learning.image_prep.split_images import main as split_img
from  transfer_learning.image_prep.filter_tiny import main as filter_tiny

# This script looks for all xml and json files in the dir_pics and makes two sub folders train and test
# where it  writes a file data.tfrecords containing the labels and pictures as a binary file


def main(dir_pics, clean=True, n_test=10):
    # define endings for pictures
    im_types = ('png', 'jpg', 'jpeg', 'bmp')

    # test if the dir exists
    if not os.path.exists(dir_pics):
        raise ValueError(dir_pics + ' not found')

    # we see if the train and test directory exist and make them if necessary
    train_folder = os.path.join(dir_pics,'train')
    test_folder = os.path.join(dir_pics,'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)


    # read xml files
    files = os.listdir(dir_pics)
    xml = [x for x in files if x.endswith(".xml")]
    pics = [x for x in files if x.endswith(im_types)]
    n_total_pics = len(pics)

    # we filter and keep only the pictures that have a xml file
    filter_xml = [x[:-4] for x in xml ]
    pics = [x for x in pics if x[:-4] in filter_xml]
    n_filtered_pics = len(pics)

    print('Found xml labels for ' + str(n_filtered_pics) + ' out of ' + str(n_total_pics) + ' files.')

    # we make a pandas df from the xml files
    xml_df = xml_to_csv.xml_to_csv(dir_pics)
    print('Successfully converted xml to csv')

    # here comes the code to read json files
    xml_df = xml_df.append(json_to_csv.json_to_csv(dir_pics, dir_pics),ignore_index=True)
    print('Successfully converted json to csv')

    # we make a dict of the classes
    classes = xml_df.loc[:,['class']].drop_duplicates()

    # we have to use 1 as the first label in tf
    classes.index = classes.index + 1

    #we extract a pd series
    classes = classes.iloc[:,0]
    classes = classes.to_dict()

    # we have to change the key, value from 1: cyclist to cyclist: 1
    classes = {y:x for x, y in classes.items()}
    print('found the following classes: ')
    print(classes)

    str_out = ''
    for k, v in classes.items():
        str_out = str_out + 'item { \n'
        str_out = str_out + '\t' + 'id: ' + str(v) + '\n'
        str_out = str_out + '\t' + 'name: \'' + str(k) + '\'\n'
        str_out = str_out + '}\n'
    with open(os.path.join(dir_pics, 'label_map.pbtxt'), "w") as text_file:
        text_file.write(str_out)

    # here comes the code to filter out labels and select or modify pictures pictures
    # for example: adjust contrast, split the the pictures because they are too large

    xml_df = split_img(xml_df, dir_pics)

    # we want to keep an entry if
    # a) it is larger than 30px (in the scaled version)
    # or
    # b) it is larger than 15px and another cyclist on this pic is larger than 30

    xml_df = filter_tiny(xml_df, img_size=512, y_small = 100, y_tiny = 50)


    xml_df.to_csv(os.path.join(dir_pics, 'xml_df_after_mod.csv'))

    # the next step is to randomly split the data in train and test
    # the split has to be performed on the image level and not on bounding box level
    images = xml_df.filename.unique()
    train, test = train_test_split(images, test_size=n_test)
    train = pd.DataFrame({'filename': train})
    test = pd.DataFrame({'filename': test})
    train = train.merge(xml_df)
    test = test.merge(xml_df)



    # we write the csv files containing a list of the files and labels to the train and test dir
    train.to_csv(os.path.join(train_folder, 'train_labels.csv'), index=None)
    test.to_csv(os.path.join(test_folder, 'test_labels.csv'), index=None)


    print("starting to copy files to training folder")
    # we copy the files
    for i in train.filename:
        shutil.copy2(os.path.join(dir_pics,i), train_folder)

    print("starting to copy files to test folder")
    # we copy the files
    for i in test.filename:
        shutil.copy2(os.path.join(dir_pics,i), test_folder)

    print('files copied to train and test dirs in ' + dir_pics)

    # we now can run the code that actually generates the tfrecords
    for p, df in zip([train_folder, test_folder], [train, test]):
        print('Starting to generate tf records in ' + p)
        writer = tf.python_io.TFRecordWriter(os.path.join(p, 'data.record'))
        grouped = gen_tfr.split(df, 'filename')

        for group in grouped:
            tf_example = gen_tfr.create_tf_example(group, p, classes)
            writer.write(tf_example.SerializeToString())

        writer.close()

        # we clean the folder from the copied images
        if clean:
            for fname in os.listdir(p):
                if fname.endswith(im_types):
                    os.remove(os.path.join(p, fname))

        print('Successfully created the TFRecords: {}'.format(p))

if __name__ == '__main__':
    path_out = r'/mnt/427149F311EAC541/pipeline'
    clean = True
    n_test = 100
    main(path_out, clean=clean, n_test=n_test)
