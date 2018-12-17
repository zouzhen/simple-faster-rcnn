from keras import backend as K
import math

class Config:
	'''
	创建一个信息类，然后所有的常用常数都在init中定义就可以
	'''

	def __init__(self):

		self.verbose = True

		self.network = 'resnet50'

		# 设置数据的参数
		self.use_horizontal_flips = False
		self.use_vertical_flips = False
		self.rot_90 = False

		# 锚点框的大小
		self.anchor_box_scales = [128, 256, 512]

		# 锚点框的比例
		self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

		# 调整图像最小边的大小。 
		self.im_size = 600

		# 图像通道均值相减 
		self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_scaling_factor = 1.0

		# 一次ROIs的次数
		self.num_rois = 4

		# RPN的步进（这取决于网络配置） 
		self.rpn_stride = 16

		# 是否需要平衡类别
		self.balanced_classes = False

		# 缩放比例
		self.std_scaling = 4.0
		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

		# 重叠的RPN的阈值
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7

		# 分类器区域重叠的阈值
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		# 类映射的占位符，由解析器自动生成 
		self.class_mapping = None

		#location of pretrained weights for the base network 
		# weight files can be found at:
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

		self.model_path = 'model_frcnn.vgg.hdf5'
