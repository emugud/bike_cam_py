import sys
sys.path.append('../')

import os

#from tf_models.models.research.object_detection import train
#from tf_models.models.research.object_detection import eval

import subprocess

cmd_train = ''

PATH_TO_TF = '/home/eg/Dropbox/bike_cam/bike_cam_py/tf_models/models/research/object_detection/'
PATH_TO_MODEL_DIRECTORY = '/home/eg/Dropbox/bike_cam/bike_cam_py/transfer_learning/train_ssd/'
PATH_TO_PROCESS_DIR = r'/mnt/427149F311EAC541/MEGA/bike_cam/tf_learning_data/'

PATH_TO_TRAIN_DIR = os.path.join(PATH_TO_PROCESS_DIR,'train')
PATH_TO_PIPELINE_CONFIG = os.path.join(PATH_TO_MODEL_DIRECTORY,'config_files/ssd_mobilenet_mod.config')
PATH_TO_EVAL_DIR = os.path.join(PATH_TO_PROCESS_DIR,'eval')


d_param = {'PATH_TO_MODEL_DIRECTORY': PATH_TO_MODEL_DIRECTORY,
         'PATH_TO_TRAIN_DIR': PATH_TO_TRAIN_DIR,
         'PATH_TO_PIPELINE_CONFIG': PATH_TO_PIPELINE_CONFIG}


cmd_train = 'python ' + os.path.join(PATH_TO_TF,'train.py') + \
            ' --logtostderr --pipeline_config_path=%s'%PATH_TO_PIPELINE_CONFIG + \
            ' --train_dir=%s'%PATH_TO_TRAIN_DIR
print(cmd_train)

cmd_eval = 'python ' + os.path.join(PATH_TO_TF,'eval.py') + \
            ' --logtostderr  --pipeline_config_path=%s'%PATH_TO_PIPELINE_CONFIG + \
            ' --checkpoint_dir=%s'%PATH_TO_TRAIN_DIR + \
            ' --eval_dir=%s'%PATH_TO_EVAL_DIR

print(cmd_eval)

cmd_tb = 'tensorboard --logdir=%s'%PATH_TO_MODEL_DIRECTORY

print(cmd_tb)

cmd_export = 'python ' + os.path.join(PATH_TO_TF,'export_inference_graph.py') +\
    ' --input_type=image_tensor ' + \
    ' --pipeline_config_path=' + PATH_TO_PIPELINE_CONFIG + \
    ' --trained_checkpoint_prefix=' + os.path.join(PATH_TO_TRAIN_DIR,'model.ckpt-100000') + \
    ' --output_directory=' + PATH_TO_PROCESS_DIR
print(cmd_export)

