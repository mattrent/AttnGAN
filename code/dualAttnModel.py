import torch
import torch.nn as nn
import torch.nn.parallel
from GlobalAttention import GlobalAttentionGeneral as TextAttentionModule
from GlobalAttention import conv1x1

class ChannelAttentionModule(nn.Module):
	def __init__(self, input_size):
		super(ChannelAttentionModule, self).__init__()
		#stride may be == 1 (NEEDS TESTING)
		self.maxPool = nn.MaxPool2d(kernel_size = 3, stride = 2)
		self.avgPool = nn.AvgPool2d(kernel_size = 3, stride = 2)
		#same layers in MLP for maxPool and avgPool?
		self.hidden_size = (input_size // 2 - 1) // 2 - 1
		self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
		self.linear_2 = nn.Linear(self.hidden_size * 2, input_size)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, spatial_features):
		maxP = self.maxPool(spatial_features)
		avgP = self.avgPool(spatial_features)
		mlp_1 = self.linear_2(
			self.relu(
				self.linear_1(maxP)
			)
		)
		mlp_2 = self.linear_2(
			self.relu(
				self.linear_1(avgP)
			)
		)
		mlp_sigm = self.sigmoid(mlp_1 + mlp_2)
		out = mlp_sigm * spatial_features
		return out


class SpatialAttentionModule(nn.Module):
	# https://arxiv.org/pdf/1805.08318v2.pdf, page 4 (Self-Attention Generative Adversarial Networks)
	# f,g,h names from the paper above
	def __init__(self, input_size, input_channels):
		super(SpatialAttentionModule, self).__init__()
		self.avgPool = nn.AvgPool2d(kernel_size = 3, stride = 2)
		self.maxPool = nn.MaxPool2d(kernel_size = 3, stride = 2)
		self.hidden_size = (input_size // 2 - 1) // 2 - 1
		self.upsampling = nn.Upsample(size = (input_size, input_size), mode = 'bilinear')
		self.conv_f = conv1x1(input_channels, input_channels)
		self.conv_g = conv1x1(input_channels, input_channels)
		self.conv_h = conv1x1(input_channels, input_channels)
		self.softmax = torch.nn.Softmax()

	def forward(self, spatial_features):
		downsampled = self.maxPool(self.avgPool(spatial_features))
		f = self.conv_f(downsampled)
		g = self.conv_g(downsampled)
		h = self.conv_h(downsampled)
		f_transposed = torch.transpose(f)
		attention_map = self.softmax(f_transposed * g)
		out = self.upsampling(h * attention_map)
		return out


class VisualAttentionModule(nn.Module):
	def __init__(self):
		pass

	#hidden_stage_result is h in the paper
	def forward(self, hidden_stage_result):
		pass


class AttentionEmbeddingModule(nn.Module):
	def __init__(self):
		pass

	def forward(self, visual_attention_result, textual_attention_result):
		pass


class DualAttentionModule(nn.Module):
	def __init__(self):
		pass

	def forward(self, hidden_stage_result, word_features):
		pass
