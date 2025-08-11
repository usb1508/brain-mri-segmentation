import os, glob
import pandas as pd
import torch
from model import Dense
#import torchgeometry as tgm
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import pydicom
import dicom2nifti
import pydicom
from utils import *
import nibabel
from inference import inference_nii, inference_nii_hemi

parser = argparse.ArgumentParser()
parser.add_argument("--block_config", default=((2,2,2),(2,2,2)), type=tuple)
parser.add_argument("--num_classes", default=2, type=int)
# parser.add_argument("--path_model",default='model_hemi_mask.pth', help='the model path')
# parser.add_argument("--dir_data",default='./data/', help='dir root of data')
parser.add_argument("--path_model",default='CODE/model_hemi_mask.pth', help='the model path')
parser.add_argument("--dir_data",default='CODE/Data', help='dir root of data')
parser.add_argument("--dir_data_dst",default='CODE/Data', help='dir root of data destination(output)')

opt = parser.parse_args()
def main():
	# set up model at this fold
	model = Dense(num_classes=opt.num_classes, block_config=opt.block_config)
	state = torch.load(opt.path_model)['model']
	model = torch.nn.DataParallel(model).cuda()
	model.load_state_dict(state)
	
	# read data
	ls_subj = [x for x in os.listdir(opt.dir_data) if '.' not in x]
	for subj in ls_subj:
		print(subj)
		path_subj = opt.dir_data + '/' +subj
		path_save = opt.dir_data_dst + '/' + subj
		checkdirctexist(path_save)
		for visit in [x for x in os.listdir(path_subj) if 'csv' not in x and '.' not in x]:
			path_v = os.path.join(path_subj, visit)
			path_save_visit = os.path.join(path_save, visit)
			checkdirctexist(path_save_visit) 
			if( '.' not in path_v):
				if len([x for x in os.listdir(path_save_visit) if 'hemi_mask' in x ]) != 1:
					path_nii = [x for x in os.listdir(path_v) if 'nii' in x and 'seg' not in x and 'hemi_mask' not in x][0]
					nii_orig = nibabel.load(path_v + '/' + path_nii)
					nii_orig1 = nibabel.as_closest_canonical(nii_orig)
					if np.sum(nii_orig.affine != nii_orig1.affine) != 0:
						print('still wrong', path_v)
						nibabel.save(nii_orig1, path_save_visit + '/' + path_nii)
					
					nii_seg = inference_nii_hemi(model, nii_orig1)
					print(f"Saving {path_save_visit}")
					nibabel.save(nii_seg, path_save_visit + '/' + 'hemi_mask.nii.gz')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()