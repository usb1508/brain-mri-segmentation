import os, glob

import pandas as pd
import torch
from model import DenseTV
#import torchgeometry as tgm
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import pydicom
import dicom2nifti
import pydicom
from inference import inference_nii
import nibabel
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--block_config", default=((4, 4, 4),(4,4,4)), type=tuple)
parser.add_argument("--num_classes", default=3, type=int)
parser.add_argument("--path_model",default='CODE/model_DenseTV.pth', help='the model path')
parser.add_argument("--dir_data_src",default='CODE/Data', help='dir root of data')
parser.add_argument("--dir_data_dst",default='CODE/Data', help='dir root of data destination(output)')
opt = parser.parse_args()
def main():
    # set up model at this fold
    model = DenseTV(num_classes=3, block_config=opt.block_config)
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load(opt.path_model)['model']
    model.load_state_dict(state)
    model = torch.nn.DataParallel(model).cuda()
    orientation = np.array([[1, 0, 0, 0, 0, -1], [1,0,0,0,1,0], [0, 1, 0, 0, 0, -1]]) # the second orientation represents axial
    # read data
    ls_subj = [x for x in os.listdir(opt.dir_data_src) if '.' not in x]
    # ls_subj = ls_subj[53:]
    # ls_subj = ['1079461E']
    # ls_subj = ['HCRNYWLT1893', '1082530E', '1083032SnA', '1083512']
    for subj in ls_subj:
        path_subj = opt.dir_data_src + '/' + subj
        path_subj_save = opt.dir_data_dst + '/' + subj
        checkdirctexist(path_subj_save)
        ls_v = [x for x in os.listdir(path_subj) if '.' not in x]
        for visit in ls_v:
            path_v = os.path.join(path_subj, visit)
            path_v_save = os.path.join(path_subj_save, visit)
            checkdirctexist(path_v_save)
            print('Analyzing:', path_v)
            # if there are slices from 3 planes, remove the sagittal and coronal ones
            if '3_PLANES' in os.listdir(path_v)[0]:
                for path_dcm in glob.glob(path_v +'/*/*.dcm'):
                    dicom = pydicom.read_file(path_dcm)
                    is_axial = np.where(np.mean(abs(orientation - np.round(dicom.ImageOrientationPatient)), 1) == 0)[0] == 1
                    if not is_axial:
                        os.remove(path_dcm)
            # if no NIFTI file is found in the directory, create it by converting DICOM series to NIFTI file		   
            if len([x for x in os.listdir(path_v) if 'nii' in x]) != -1: 
                dicom2nifti.convert_directory(path_v, path_v)
                # find metadata
                for path_dcm in glob.glob(path_v +'/*/*.dcm'):
                    dicom = pydicom.read_file(path_dcm)
                    date = dicom.StudyDate
                    break
            # find the file name of .nii file
            path_nii = [x for x in os.listdir(path_v) if 'nii.gz' in x and 'seg' not in x and 'hemi' not in x][0]
            nii_orig = nibabel.load(path_v + '/' + path_nii)
            # standardize the orientation
            try:
                nii_orig = nibabel.as_closest_canonical(nii_orig)
            except:
                print(path_nii)
                os.remove(path_v + '/' + path_nii)
            print('Saving:', path_v_save + '/' + path_nii)
            nibabel.save(nii_orig, path_v_save + '/' + path_nii)
            # segment the MRI images using the model
            nii_seg = inference_nii(model, nii_orig)
            print('Saving:', path_v_save + '/' + visit + '_seg.nii.gz')
            nibabel.save(nii_seg, path_v_save + '/' + visit + '_seg.nii.gz')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
