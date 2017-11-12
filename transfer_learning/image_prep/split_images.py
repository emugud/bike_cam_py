import numpy as np
from PIL import Image
import pandas as pd
import os

def merge_intevals(l):
    """ takes a list of lists and merges overlapping lists"""
    l.sort(key=lambda x: x[0])
    n = len(l)
    i = 0
    while i < (n-1):
        # no overalapping
        if l[i + 1][0] > l[i][1]:
            i = i + 1
            continue
        # overlapping, end of next interval is large
        if l[i + 1][1] > l[i][1]:
            l[i][1] = l[i + 1][1]
            del l[i + 1]
            n = len(l)
            continue
        # overlapping, end of next interval is smaller
        if l[i + 1][1] <= l[i][1]:
            del l[i + 1]
            n = len(l)
            continue
    return l


def main(df, dir_l):
    """splits images listed in the column filename of df in the middle, if there is no object
    otherwise it finds the closest possible point to split without cutting an object in half"""

    for i in df.filename.unique():
        #print('splitting image ' + i )
        iw = df.loc[df.filename == i, 'width'].unique()[0]
        ih = df.loc[df.filename == i, 'height'].unique()[0]
        mid = iw // 2
        xmin = df.loc[df.filename == i, 'xmin']
        xmax = df.loc[df.filename == i, 'xmax']
        # we check if any of the bounding boxes boarder crosses the mid line
        if any(np.logical_and(xmin < mid, xmax > mid)):

            # and if yes we merge overlapping intervals and find the closest splitting line
            l = zip(xmin, xmax)
            l = [[x[0], x[1]] for x in l]
            l = merge_intevals(l)
            # we identify the endpoints of the interval that covers the midpoint
            l = [x for x in l if (x[0] < mid and x[1] > mid)][0]
            # and assign the closer endpoint to mid
            if (l[1] - mid) > (mid - l[0]):
                mid = l[0]
            else:
                mid = l[1]
            print("object in midpoint, new point is " + str(mid))
        # we read the image
        img = Image.open(os.path.join(dir_l, i))
        # and divide
        img_l = img.crop((0, 0, mid, ih))
        img_r = img.crop((mid, 0, iw, ih))

        # we construct the filenames
        l_filename = i[:-4] + '_l' + i[-4:]
        r_filename = i[:-4] + '_r' + i[-4:]

        # and save the images
        img_l.save(os.path.join(dir_l, l_filename))
        img_r.save(os.path.join(dir_l, r_filename))

        # we have to update the df:
        # for objects in the left image: filename = l_filename, width = mid
        # object is in the left iff xmax < mid
        df.loc[(df.filename == i) & (df.xmax <= mid), 'width'] = mid
        df.loc[(df.filename == i) & (df.xmax <= mid), 'filename'] = l_filename

        # for objects in the right
        df.loc[(df.filename == i) & (df.xmin >= mid), 'filename'] = r_filename
        df.loc[(df.filename == r_filename) & (df.xmin >= mid), 'width'] = mid
        df.loc[(df.filename == r_filename) & (df.xmin >= mid), 'xmax'] = df.loc[(df.filename == r_filename) & (
        df.xmin >= mid), 'xmax'] - mid
        df.loc[(df.filename == r_filename) & (df.xmin >= mid), 'xmin'] = df.loc[(df.filename == r_filename) & (
        df.xmin >= mid), 'xmin'] - mid

    return df