import numpy as np
import math, os
import pdb
import torch
import shutil
import SimpleITK as sitk

def checkdirctexist(dirct):
	if not os.path.exists(dirct):
		os.makedirs(dirct)


def save_exp(root_dir, exp_name, codes):
	exp_dir = os.path.join(root_dir, 'experiments', exp_name)

	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)

	if not os.path.exists(os.path.join(exp_dir, 'results')):
		os.makedirs(os.path.join(exp_dir, 'results'))
	if not os.path.exists(os.path.join(exp_dir, 'codes')):
		os.makedirs(os.path.join(exp_dir, 'codes'))
	if not os.path.exists(os.path.join(exp_dir, 'loss')):
		os.makedirs(os.path.join(exp_dir, 'loss'))
	if not os.path.exists(os.path.join(exp_dir, 'loss')):
		os.makedirs(os.path.join(exp_dir, 'loss'))

	shutil.copy(os.path.join(root_dir, codes['main']), os.path.join(exp_dir, 'codes', codes['main']))
	shutil.copy(os.path.join(root_dir, codes['model']), os.path.join(exp_dir, 'codes', codes['model']))
	shutil.copy(os.path.join(root_dir, codes['dataset']), os.path.join(exp_dir, 'codes', codes['dataset']))
	return exp_dir


def PSNR_self(pred, gt, shave_border=0):
	height, width = pred.shape[:2]
	pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
	gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
	imdff = (pred -gt)
	pred_np = pred.cpu().images.numpy()
	gt_np = pred.cpu().images.numpy()
	rmse = math.sqrt(np.mean(imdff.cpu().images.numpy() ** 2))
	if rmse == 0:
		return 100
	return 20.0 * math.log10(1/rmse)


def adjust_learning_rate(epoch, opt):
	lr = opt.lr * (opt.lr_reduce ** (epoch // opt.step))
	return lr


def save_checkpoint(model, epoch, optimizer, opt, save_path):
	checkdirctexist(save_path)
	model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
	state = {"epoch": epoch, "model": model.state_dict(), "optimizer":optimizer.state_dict()}
	# check path status
	if not os.path.exists("model/"):
		os.makedirs("model/")
	# save model_files
	torch.save(state, model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))
	# copy python file to checkpoints folder


def normalize(data):
	N, _, H, W = data.shape
	img = torch.cat([data, data, data], 1)
	mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).cuda()
	std = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).cuda()
	img = (img - mean) / std
	return img


def convert_to_cbf(map, num):
	RGB =[[245, 121, 58], [169, 90, 161], [133, 192, 249], [15, 32, 128]]
	new_map = np.zeros([map.shape[0], map.shape[1], 3])
	map = np.round(map).astype(int)
	for c in range(num):
		new_map[map==c+1] = RGB[c]
	return new_map.astype(np.uint8)



def bias_field_correction(image):
	corrector = sitk.N4BiasFieldCorrectionImageFilter()
	inputImage = sitk.GetImageFromArray(image)
	#image = inputImage
	maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

	numberFittingLevels = 3
	corrector.SetMaximumNumberOfIterations(
		[10] * numberFittingLevels
	)
	correct_image = corrector.Execute(inputImage, maskImage)

	correct_image = sitk.GetArrayFromImage(correct_image)
	return correct_image