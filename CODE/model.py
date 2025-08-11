import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


def _bn_function_factory(norm, relu, conv):
	def bn_function(*inputs):
		concated_features = torch.cat(inputs, 1)
		bottleneck_output = conv(relu(norm(concated_features)))
		return bottleneck_output

	return bn_function


class _DenseLayer(nn.Sequential):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
		super(_DenseLayer, self).__init__()
		self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
		self.add_module('relu1', nn.ReLU(inplace=True)),
		self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
										   growth_rate, kernel_size=1, stride=1,
										   bias=False)),
		self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
		self.add_module('relu2', nn.ReLU(inplace=True)),
		self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
										   kernel_size=3, stride=1, padding=1,
										   bias=False)),
		self.drop_rate = drop_rate
		self.memory_efficient = memory_efficient

	def forward(self, *prev_features):
		bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
		bottleneck_output = bn_function(*prev_features)
		new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate,
									 training=self.training)
		return new_features


class _DenseBlock(nn.Module):
	def __init__(self, num_layers, num_input_features, num_output_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(
				num_input_features + i * growth_rate,
				growth_rate=growth_rate,
				bn_size=bn_size,
				drop_rate=drop_rate,
				memory_efficient=memory_efficient,
			)
			self.add_module('denselayer%d' % (i + 1), layer)
		#self.add_module('se', _SE(num_input_features + num_layers* growth_rate, num_input_features + num_layers* growth_rate//16))
		self.add_module('trans', nn.Conv2d(num_input_features + num_layers* growth_rate, num_output_features, kernel_size=(1,1), stride=(1,1)))
	def forward(self, init_features):
		features = [init_features]
		for name, layer in self.named_children():
			if name != 'trans':
				new_features = layer(*features)
				features.append(new_features)
		#se = self.se(torch.cat(features, 1))
		return self.trans(torch.cat(features, 1))


class TransitionBlockEncoder(nn.Module):
	def __init__(self, in_planes, out_planes, dropRate=0.0):
		super(TransitionBlockEncoder, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=(1,1), stride=(1,1),
										padding=(0,0), bias=False)
		self.pool1 = nn.MaxPool2d(2,stride=2)
		self.droprate = dropRate

	def forward(self, x):
		out = self.conv1(self.relu(self.bn1(x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
		return self.pool1(out)


class TransitionBlockDecoder(nn.Module):
	def __init__(self, in_planes, out_planes, cubic = False, dropRate=0.0):
		super(TransitionBlockDecoder, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
										padding=0, bias=False)
		self.droprate = dropRate
		self.cubic = cubic

	def forward(self, x):
		out = self.conv1(self.relu(self.bn1(x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
		if self.cubic:
			return F.upsample_bilinear(out, scale_factor=2)
		else:
			return F.upsample_nearest(out, scale_factor=2)


class Encoder(nn.Module):
	def __init__(self, pretrain = True, inter_planes=128, block_config=(4, 4), growth_rate =32):
		super(Encoder, self).__init__()
		############# Encoder 0 - 256 ##############
		self.conv0 = models.densenet121(pretrained = pretrain).features.conv0
		self.pool0 = models.densenet121().features.pool0
		self.dense1 = _DenseBlock(num_layers=block_config[0], num_input_features=64, num_output_features=inter_planes[0],
								  growth_rate=growth_rate)
		self.trans1 = TransitionBlockEncoder(inter_planes[0], inter_planes[1])
		self.dense2 = _DenseBlock(num_layers=block_config[0], num_input_features=inter_planes[1], num_output_features=inter_planes[1],
								  growth_rate=growth_rate)
		self.trans2 = TransitionBlockEncoder(inter_planes[1], inter_planes[2])
		############# Encoder 3 - 32 ##########################
		self.dense3 = _DenseBlock(num_layers=block_config[1], num_input_features=inter_planes[2], num_output_features=inter_planes[2],
								  growth_rate=growth_rate)
		#self.trans3 = TransitionBlockEncoder(inter_planes[2], inter_planes[3])
		# self.dense4 = _DenseBlock(num_layers=block_config[2], num_input_features=inter_planes[3], num_output_features=inter_planes[3],
		#						   growth_rate=growth_rate)
		############# Decoder 0 -32 ##############################
	def forward(self, x):
		out0 = self.dense1(self.pool0(self.conv0(x)))
		out1 = self.dense2(self.trans1(out0))
		out2 = self.dense3(self.trans2(out1))
		return out0, out1, out2



class DecoderTV(nn.Module):
	def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
		super(DecoderTV, self).__init__()
		############# Decoder 0 - 256 ##############
		self.TransDecoder0 = TransitionBlockDecoder(in_planes, 512)
		############# Decoder 1 - 128 ########################
		num_feat = 512
		self.DenseDecoder0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_feat + inter_planes[0],
										 growth_rate=growth_rate, num_output_features=(num_feat + block_config[0] * growth_rate)//2)
		num_feat = (num_feat + block_config[0] * growth_rate)//2
		self.TransDecoder1 = TransitionBlockDecoder(num_feat, num_feat)
		############# Decoder 2 - 64  ########################
		self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_feat + inter_planes[1],
										 growth_rate=growth_rate, num_output_features=(num_feat + block_config[1] * growth_rate)//2)
		num_feat = (num_feat + block_config[1] * growth_rate)//2
		self.TransDecoder2= TransitionBlockDecoder(num_feat, num_feat)
		############# Decoder 3 - 32 ##########################
		self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_feat,
										 growth_rate=growth_rate, num_output_features=(num_feat + block_config[2] * growth_rate)//2)
		num_feat = (num_feat + block_config[2] * growth_rate)//2
		############# Final  ##############################
		self.TransDecoder3 = TransitionBlockDecoder(num_feat, out_planes, cubic=True)
	def forward(self, x0, x1, x2):
		'''
		:param x0: 256 x 128 x 128
		:param x1: 512 x 64 x 64
		:param x2: 512 x 32 x 32
		:return:
		'''
		out30 = self.TransDecoder0(x2)
		out31 = torch.cat([x1, out30], 1)
		out40 = self.TransDecoder1(self.DenseDecoder0(out31))
		out41 = torch.cat([x0, out40], 1)
		out50 = self.TransDecoder2(self.DenseDecoder1(out41))
		out = self.TransDecoder3(self.DenseDecoder2(out50))
		return out, out30, out40, out50





class Decoder(nn.Module):
	def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
		super(Decoder, self).__init__()
		############# Decoder 0 - 256 ##############
		self.TransDecoder0 = TransitionBlockDecoder(in_planes, 512)
		############# Decoder 1 - 128 ########################
		num_feat = 512
		self.DenseDecoder0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_feat + inter_planes[0],
										 growth_rate=growth_rate, num_output_features=(num_feat + block_config[0] * growth_rate)//2)
		num_feat = (num_feat + block_config[0] * growth_rate)//2
		self.TransDecoder1 = TransitionBlockDecoder(num_feat, num_feat)
		############# Decoder 2 - 64  ########################
		self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_feat + inter_planes[1],
										 growth_rate=growth_rate, num_output_features=(num_feat + block_config[1] * growth_rate)//2)
		num_feat = (num_feat + block_config[1] * growth_rate)//2
		self.TransDecoder2= TransitionBlockDecoder(num_feat, num_feat)
		############# Decoder 3 - 32 ##########################
		self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_feat,
										 growth_rate=growth_rate, num_output_features=(num_feat + block_config[2] * growth_rate)//2)
		num_feat = (num_feat + block_config[2] * growth_rate)//2
		############# Final  ##############################
		self.TransDecoder3 = TransitionBlockDecoder(num_feat, out_planes, cubic=True)
	def forward(self, x0, x1, x2):
		'''
		:param x0: 256 x 128 x 128
		:param x1: 512 x 64 x 64
		:param x2: 512 x 32 x 32
		:return:
		'''
		out3 = self.TransDecoder0(x2)
		out3 = torch.cat([x1, out3], 1)
		out4 = self.TransDecoder1(self.DenseDecoder0(out3))
		out4 = torch.cat([x0, out4], 1)
		out5 = self.TransDecoder2(self.DenseDecoder1(out4))
		out = self.TransDecoder3(self.DenseDecoder2(out5))
		return out


class Dense(nn.Module):
	def __init__(self, num_classes, pretrain = True, block_config=((4,4), (12,16,4)), growth_rate =32):
		super(Dense, self).__init__()
		############# First downsampling  ############## 512

		self.encoder = Encoder(pretrain=pretrain, inter_planes=[128, 256, 1024], block_config=block_config[0], growth_rate=growth_rate)
		self.decoder = Decoder(in_planes=1024, inter_planes=[256, 128], out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
		self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
	def forward(self, x):
		x0, x1, x2 = self.encoder(x)
		out = self.decoder(x0, x1, x2)
		out = self.conv_out(out)
		return out



class DenseTV(nn.Module):
	def __init__(self, num_classes, pretrain = True, block_config=((4,4), (4,4,4)), growth_rate =32):
		super(DenseTV, self).__init__()
		############# First downsampling  ############## 512

		self.encoder = Encoder(pretrain=pretrain, inter_planes=[128, 256, 512], block_config=block_config[0], growth_rate=growth_rate)
		self.decoder = DecoderTV(in_planes=512, inter_planes=[256, 128], out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
		self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
	def forward(self, x):
		x0, x1, x2 = self.encoder(x)
		out, out3, out4, out5 = self.decoder(x0, x1, x2)
		out = self.conv_out(out)
		return out, out3, out4, out5



class _SoftMaskBlock(nn.Module):
	def __init__(self, num_layers, num_input_features, drop_rate, memory_efficient=False):
		super(_SoftMaskBlock, self).__init__()
		self.down = nn.Sequential(OrderedDict([
			('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
			('res0', _ResidualLayer(num_input_features, num_input_features))]))
		self.down_up = nn.Sequential(OrderedDict([
			('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
		]))
		for i in range(num_layers):
			self.down_up.add_module('res{}'.format(i), _ResidualLayer(num_input_features, num_input_features))
		self.down_up.add_module('upsample0', nn.UpsamplingBilinear2d(scale_factor=2))
		self.skip = _ResidualLayer(num_input_features, num_input_features)
		self.up = nn.Sequential(OrderedDict([
			('res{}'.format(num_layers + 2), _ResidualLayer(num_input_features, num_input_features)),
			('upsample1', nn.UpsamplingBilinear2d(scale_factor=2)),
			('conv0', nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1,
								padding=0, bias=False)),
			('conv1', nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1,
								padding=0, bias=False)),
			('sig', nn.Sigmoid())
		]))

	def forward(self, init_features):
		out_down = self.down(init_features)
		out_downup = self.down_up(out_down)
		out_skip = self.skip(out_downup)
		out_up = self.up(out_downup + out_skip)
		return out_up



class _ResidualLayer(nn.Module):  #@save
	def __init__(self, input_channels, output_channels,
				 use_1x1conv=False, strides=1):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, output_channels,
							   kernel_size=3, padding=1, stride=strides)
		self.conv2 = nn.Conv2d(output_channels, output_channels,
							   kernel_size=3, padding=1)
		if use_1x1conv:
			self.conv3 = nn.Conv2d(input_channels, output_channels,
								   kernel_size=1, stride=strides)
		else:
			self.conv3 = None
		self.bn1 = nn.BatchNorm2d(output_channels)
		self.bn2 = nn.BatchNorm2d(output_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, X):
		Y = F.relu(self.bn1(self.conv1(X)))
		Y = self.bn2(self.conv2(Y))
		if self.conv3:
			X = self.conv3(X)
		Y += X
		return F.relu(Y)

