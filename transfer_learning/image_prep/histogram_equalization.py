import cv2
import numpy as np
import os

# perform an histogram equalization on all images of the given folder
# tif inplace = True images are overwritten

def main(input_folder, inplace = False, show = False):

    im_types = ('png', 'jpg', 'jpeg', 'bmp')

    if not os.path.exists(input_folder):
        raise ValueError('Dir %s not found' % input_folder)

    if inplace:
        output_folder = input_folder
    else:
        output_folder = os.path.join(input_folder, 'hist_equalized')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    total_count = len(list(os.listdir(input_folder)))
    print('found ' + str(total_count) + ' files in ' + input_folder +'. Performing histogramm equalization.')
    i = 0

    for image_file in os.listdir(input_folder):

        image_filename = os.fsdecode(image_file)

        if image_filename.endswith(im_types):

            img = cv2.imread(os.path.join(input_folder, image_filename))

            # we see if the image could be loaded by accessing it shape
            try:
                img_h = img.shape[0]
            except:
                print(' - - - - - ' + image_filename + ' could not be loaded, skipping')
                continue

            if img_h < 150:
                print(image_filename + ' too small, skipping')
                continue

            if show:
                cv2.imshow('Color input image', img)


            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            if show:
                cv2.imshow('Histogram equalized', img_output)

            if inplace:
                img_out_name = image_filename
            else:
                img_out_name = image_filename[:-4] + '_he' + image_filename[-4:]

            img_out_name = os.path.join(output_folder, img_out_name)
            cv2.imwrite(img_out_name,img_output )

            if show:
                if cv2.waitKey(15) & 0xFF == ord('q'):
                    break

cv2.destroyAllWindows()

if __name__ == '__main__':
    main(r'/mnt/427149F311EAC541/try',inplace=True)