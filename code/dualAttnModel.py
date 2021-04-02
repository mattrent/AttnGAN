import torch
import torch.nn as nn
import torch.nn.parallel

from GlobalAttention import GlobalAttentionGeneral as TextAttentionModule
from GlobalAttention import conv1x1
from model import GET_IMAGE_G
from model import upBlock

from miscc.config import cfg


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
	def __init__(self, input_size, input_channels):
		super(VisualAttentionModule, self).__init__()
		self.channel_attention_module = ChannelAttentionModule(input_size)
		self.spatial_attention_module = SpatialAttentionModule(input_size, input_channels)

	#hidden_stage_result is h in the paper
	def forward(self, hidden_stage_result):
		out_1 = self.channel_attention_module(hidden_stage_result)
		out_2 = self.spatial_attention_module(out_1)
		return out_2


class DualAttentionModule(nn.Module):
	def __init__(self, hidden_stage_result_size, hidden_state_result_channels, word_features_size):
		self.textual_attention_module = TextAttentionModule(hidden_state_result_channels, word_features_size)
		self.visual_attention_module = VisualAttentionModule(hidden_stage_result_size, hidden_state_result_channels)

	def forward(self, hidden_stage_result, word_features, mask):
		self.textual_attention_module.applyMask(mask)
		results_t = self.textual_attention_module(hidden_stage_result, word_features)
		results_v = self.visual_attention_module(hidden_stage_result)
		out_1 = hidden_stage_result + results_t
		out_2 = hidden_stage_result + results_v
		#concat along which dim?
		out = torch.cat((out_1, out_2), dim = 1)
		return out


class DualAttn_INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM
        nz, ngf = self.in_dim, self.gf_dim

        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
			#changed from GLU to ReLU
            nn.ReLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class DualAttn_NEXT_STAGE_G(nn.Module):
	#image features size, embedding size, condition size
	def __init__(self, hidden_state_result_channels, word_features_size, sentence_features_size):
		pass

	def forward(self):
		pass


class DualAttn_G_NET(nn.Module):
	def __init__(self):
		pass

	def forward(self):
		pass
