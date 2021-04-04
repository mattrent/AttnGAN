import torch
import torch.nn as nn
import torch.nn.parallel

from GlobalAttention import GlobalAttentionGeneral as TextAttentionModule
from GlobalAttention import conv1x1
from model import GET_IMAGE_G
from model import upBlock
from model import ResBlock
from model import CA_NET

from miscc.config import cfg


class ChannelAttentionModule(nn.Module):
	def __init__(self, input_size, input_channels):
		super(ChannelAttentionModule, self).__init__()
		#stride may be == 1 (NEEDS TESTING)
		self.maxPool = nn.MaxPool2d(kernel_size = 3, stride = 4)
		self.avgPool = nn.AvgPool2d(kernel_size = 3, stride = 4)


		#same layers in MLP for maxPool and avgPool?
		self.input_size = input_size
		self.input_channels = input_channels
		self.hidden_size = (input_size // 8)  * (input_size // 8) * input_channels // 8

		#NOTE: prohibitively high memory usage, possibly wrong model?
		#TODO: add convolution before and transposed convolution after (reduce dimension before linear layer and then scale back to original size)

		self.conv = nn.Conv2d(input_channels, input_channels // 8, kernel_size = 1, stride = 2)
		self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
		self.linear_2 = nn.Linear(self.hidden_size * 2, input_size // 2 * input_size // 2 * input_channels // 8)
		self.conv_t = nn.ConvTranspose2d(input_channels // 8, input_channels, kernel_size = 1, stride = 2, output_padding = (1, 1))

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		# print("INPUT_SIZE ->", input_size)
		# print("INPUT_CHANNELS ->", input_channels)
		# print("HIDDEN SIZE ->", self.hidden_size)

	def forward(self, spatial_features):
		maxP = self.maxPool(spatial_features)
		avgP = self.avgPool(spatial_features)
		# print("maxPooling -> ", maxP.shape)

		compressed_maxP = self.conv(maxP)
		compressed_avgP = self.conv(avgP)
		# print("compressed_maxP ->", compressed_maxP.shape)
		# print("hidden size ->", self.hidden_size)

		mlp_1 = self.linear_2(
			self.relu(
				self.linear_1(torch.reshape(compressed_maxP, (-1, self.hidden_size)))
			)
		)
		mlp_2 = self.linear_2(
			self.relu(
				self.linear_1(torch.reshape(compressed_avgP, (-1, self.hidden_size)))
			)
		)

		#mlp_sigm = self.sigmoid(mlp_1 + mlp_2)
		#out = torch.reshape(mlp_sigm, spatial_features.shape) * spatial_features

		#print("MLP_SIGM ->", mlp_sigm.shape)
		#print("SPATIAL_FEATURES->", spatial_features.shape)
		mlp_out = mlp_1 + mlp_2
		# print("MLP_OUT ->", mlp_out.shape)
		mlp_out_reshaped = torch.reshape(mlp_out, (-1, self.input_channels // 8, self.input_size // 2, self.input_size // 2))
		mlp_sigm_reshaped = self.sigmoid(self.conv_t(mlp_out_reshaped))
		out = mlp_sigm_reshaped * spatial_features

		return out


class SpatialAttentionModule(nn.Module):
	# https://arxiv.org/pdf/1805.08318v2.pdf, page 4 (Self-Attention Generative Adversarial Networks)
	# f,g,h names from the paper above
	def __init__(self, input_size, input_channels):
		super(SpatialAttentionModule, self).__init__()
		self.avgPool = nn.AvgPool2d(kernel_size = 3, stride = 2)
		self.maxPool = nn.MaxPool2d(kernel_size = 3, stride = 2)
		self.hidden_size = (input_size // 2 - 1)
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
		f_transposed = torch.transpose(f, 2, 3)
		# print("G SHAPE->", g.shape)
		# print("F_TRANSPOSED SHAPE->", f_transposed.shape)
		attention_map = self.softmax(f_transposed * g)
		out = self.upsampling(h * attention_map)
		return out


class VisualAttentionModule(nn.Module):
	def __init__(self, input_size, input_channels):
		super(VisualAttentionModule, self).__init__()
		self.channel_attention_module = ChannelAttentionModule(input_size, input_channels)
		self.spatial_attention_module = SpatialAttentionModule(input_size, input_channels)

	#hidden_stage_result is h in the paper
	def forward(self, hidden_stage_result):
		out_1 = self.channel_attention_module(hidden_stage_result)
		out_2 = self.spatial_attention_module(out_1)
		return out_2


class DualAttentionModule(nn.Module):
	def __init__(self, hidden_stage_result_size, hidden_stage_result_channels, word_features_size):
		super(DualAttentionModule, self).__init__()
		self.textual_attention_module = TextAttentionModule(hidden_stage_result_channels, word_features_size)
		self.visual_attention_module = VisualAttentionModule(hidden_stage_result_size, hidden_stage_result_channels)

	def forward(self, hidden_stage_result, word_features, mask):
		self.textual_attention_module.applyMask(mask)
		results_t, _ = self.textual_attention_module(hidden_stage_result, word_features)
		results_v = self.visual_attention_module(hidden_stage_result)
		out_1 = hidden_stage_result + results_t
		out_2 = hidden_stage_result + results_v
		#concat along which dim?
		out = torch.cat((out_1, out_2), dim = 1)
		return out


class DualAttn_INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(DualAttn_INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM
        nz, ngf = self.in_dim, self.gf_dim

        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
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
	def __init__(self, hidden_state_result_size, hidden_state_result_channels, word_features_size, sentence_features_size):
		super(DualAttn_NEXT_STAGE_G, self).__init__()
		self.dual_attention_module = DualAttentionModule(hidden_state_result_size, hidden_state_result_channels, word_features_size)
		#TODO: inverted residual instead of residual
		self.residual = self._make_layer(ResBlock, hidden_state_result_channels * 2)
		self.upsample = upBlock(hidden_state_result_channels * 2, hidden_state_result_channels)

	def _make_layer(self, block, channel_num):
		layers = []
		for i in range(cfg.GAN.R_NUM):
			layers.append(block(channel_num))
		return nn.Sequential(*layers)

	def forward(self, hidden_stage_result, word_features, mask):
		out_att = self.dual_attention_module(hidden_stage_result, word_features, mask)
		out_res = self.residual(out_att)
		out = self.upsample(out_res)
		return out


class DualAttn_G_NET(nn.Module):
    def __init__(self):
        super(DualAttn_G_NET, self).__init__()
        self.hidden_state_result_channels = cfg.GAN.GF_DIM
        self.word_features_size = cfg.TEXT.EMBEDDING_DIM
        self.sentence_features_size = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = DualAttn_INIT_STAGE_G(self.hidden_state_result_channels * 16, self.sentence_features_size)
            self.img_net1 = GET_IMAGE_G(self.hidden_state_result_channels)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = DualAttn_NEXT_STAGE_G(64, self.hidden_state_result_channels, self.word_features_size, self.sentence_features_size)
            self.img_net2 = GET_IMAGE_G(self.hidden_state_result_channels)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = DualAttn_NEXT_STAGE_G(128, self.hidden_state_result_channels, self.word_features_size, self.sentence_features_size)
            self.img_net3 = GET_IMAGE_G(self.hidden_state_result_channels)

    def forward(self, z_code, sentence_features, word_features, mask):
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sentence_features)
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            # print("H_CODE1 -> ", h_code1.shape)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, word_features, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, word_features, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs, [], mu, logvar
