import numpy as np
import torch
import cv2

import nibabel as nib
import os, glob

def inference_nii(model, nii):
	model.eval()
	data = nii.get_fdata()
	# get number of images
	num_img = data.shape[2]
	seg_results = []
	for j in range(num_img):
		if j < 2 or j > num_img - 2:
			seg_results.append(np.zeros(data[:, :, j].shape))
			continue
		# the head is towards left
		img = np.rot90(data[:, :, j], k=1) # rotate 3 times counter clock-wise
		H, W = img.shape
		vmax = np.percentile(img[img> 0], 99.7)
		img = img / vmax
		img = img / (np.max(img) + 0.001)
		img = cv2.resize(img, (448,448))
		img = np.concatenate([img[None, None, :, :], img[None, None, :, :], img[None, None, :, :]], 1)
		mean = np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
		std = np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
		img = (img - mean) / std
		# expand the image to 1x1xHxW and convert it to cuda tensor
		data_ts = torch.tensor(img, dtype=torch.float32).cuda()

		# input to model
		with torch.no_grad():
			out, _, _, _ = model(data_ts)
		out = out.cpu().numpy()
		pred = np.argmax(out[0], 0).astype(np.float32) # shape: H x W, values of each pixel: 0 - 5
		pred = cv2.resize(pred, (W, H))
		seg_results.append(np.rot90(pred, k=-1))
	seg_results = np.array(seg_results)
	seg_results = np.round(np.rollaxis(seg_results, 0, 3))

	seg_result_nii = nib.Nifti1Image(seg_results, nii.affine, nii.header)
	return seg_result_nii



def inference_nii_hemi(model, nii):
	model.eval()
	data = nii.get_fdata()
	# get number of images
	num_img = data.shape[2]
	seg_results = []
	#print("===> The shape of the data input: {}".format(data.shape))
	for j in range(num_img):
		if j < 2 or j > num_img - 2:
			seg_results.append(np.zeros(data[:, :, j].shape))
			continue
		# the head is towards left
		
		img = np.rot90(data[:, :, j], k=1) # rotate 3 times counter clock-wise
		H, W = img.shape
		vmax = np.percentile(img[img> 0], 99.7)
		img = img / vmax
		img = img / (np.max(img) + 0.001)
		img = cv2.resize(img, (448,448))
		img = np.concatenate([img[None, None, :, :], img[None, None, :, :], img[None, None, :, :]], 1)
		mean = np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
		std = np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
		img = (img - mean) / std
		# expand the image to 1x1xHxW and convert it to cuda tensor
		data_ts = torch.tensor(img, dtype=torch.float32).cuda()

		# input to model
		with torch.no_grad():
			out = model(data_ts)

		out = out.cpu().numpy()
		pred = np.argmax(out[0], 0).astype(np.float32) # shape: H x W, values of each pixel: 0 - 5
		pred = cv2.resize(pred, (W, H))
		seg_results.append(np.rot90(pred, k=-1))
	seg_results = np.array(seg_results)
	seg_results = np.round(np.rollaxis(seg_results, 0, 3))

	
	seg_result_nii = nib.Nifti1Image(seg_results, nii.affine, nii.header)
	return seg_result_nii


def check_orientation(ct_image, ct_arr):
	"""
	Check the NIfTI orientation, and flip to  'RPS' if needed.
	:param ct_image: NIfTI file
	:param ct_arr: array file
	:return: array after flipping
	"""
	x, y, z = nib.aff2axcodes(ct_image.affine)
	if x != 'R':
		ct_arr = nib.orientations.flip_axis(ct_arr, axis=0)
	if y != 'P':
		ct_arr = nib.orientations.flip_axis(ct_arr, axis=1)
	if z != 'S':
		ct_arr = nib.orientations.flip_axis(ct_arr, axis=2)
	return ct_arr
	
	
