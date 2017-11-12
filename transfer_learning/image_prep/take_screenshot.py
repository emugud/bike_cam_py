import pyscreenshot as ImageGrab
import os


folder_out = r'/mnt/427149F311EAC541/MEGA/bike_cam/data/cycl_train_label/yt/'
im_types = ('png', 'jpg', 'jpeg', 'bmp')
n_files = len([x for x in os.listdir(folder_out) if x.endswith(im_types)])

# part of the screen
im = ImageGrab.grab()
bbox=(270, 220 , 1380, 870)
im = im.crop(bbox)

name_out = 'img_' + str(n_files+1) + '.png'

# to file
im.save( os.path.join(folder_out,name_out) )

