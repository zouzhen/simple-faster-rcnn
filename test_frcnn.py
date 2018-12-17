from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default='/home/duoda/WorkSpace/VSCode/DeepLearning/keras-frcnn-master/demo')
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

if not options.test_path:   # 如果文件名字没有给出
	parser.error('Error: path to test data must be specified. Pass --path to command line')

# 从option类里面得到训练类的文件路径
config_output_filename = options.config_filename

# 采用只读，加载对象
with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

# 引入特征提取的网络
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img_size(img, C):
	""" 1. 首先从配置文件夹中得到最小边的大小
		2. 得到图片的高度和宽度
		3. 根据高度和宽度谁大谁小，确定规整后图片的高宽
		4. 将图片缩放到指定的大小，用的是立方插值。返回的缩放后的图片img和相应的缩放的比例。 """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" 1. 将图片的BGR变成RGB，因为网上训练好的VGG图片都是以此训练的
		2. 将图片数据类型转换为np.float32，并减去每一个通道的均值，理由同上
		3. 图片的像素值除一个缩放因子，此处为1
		4. 将图片的深度变到第一个位置
		5. 给图片增加一个维度 """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" 1. 将图片缩放到规定的大小
		2. 对图片每一个通道的像素值做规整 """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

# 测试需要的基本参数的设定
class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

# 读取每一个类对应的数字（如：'person':0）
# 如果class_mapping中没有背景类的话，添加背景类且其对应的数字为len(class_mapping)
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

#预选框的个数，一次识别多少语预选框
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# 定义公共的层 (这里的resnet, 可以是VGG, Inception, 等等)
shared_layers = nn.nn_base(img_input, trainable=True)

# 定义RPN网络, 建立在公共层之上
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

# 构建classifier输出，参数分别是：特征层输出，预选框，探测框的数目，多少个类，是否可训练
classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

# 构建rpn网络
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.02

visualise = True

# 利用enumerate同时获得索引和值
for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	
	# 开始计时，得到一张图片的路径，读取文件
	st = time.time()
	filepath = os.path.join(img_path,img_name)
	img = cv2.imread(filepath)

	# 将图片规整到制定的大小
	X, ratio = format_img(img, C)

	# 如果用的是tensorflow内核，需要将图片的深度变换到最后一位
	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# 从RPN网络中得到特征区域和输出进行区域预测
	# 1. Y1:anchor包含物体的概率
	# 2. Y2:每一个anchor对应的回归梯度
	# 3. F:卷积后的特征图，接下来会有用
	[Y1, Y2, F] = model_rpn.predict(X)
	
	# 根据rpn预测的结果，得到预选框。
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

	# 转换坐标形式，从(x1,y1,x2,y2) 到 (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# 将空间金字塔池应用于拟议区域 
	bboxes = {}
	probs = {}

	# 遍历所有的预选框，需要注意的是每一次遍历预选框的个数为C.num_rois
	# 1. 每一次遍历C.num_rois个预选框，那么总共需要R.shape[0]//C.num_rois + 1【注：//是取整（如：print(10//3) 输出是3）】
	# 2. 取出从C.num_rois个预选框，并增加一个维度【注：当不满一个C.num_rois，其自动只取到最后一个】
	# 3. 当预选框被取空的时候，停止循环
	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		# 当最后一次去不足C.num_rois个预选框时，补第一个框使其达到C.num_rois个。
		# 1. 得到现在ROIs的shape
		# 2. 得到目标ROIs的shape
		# 3. 创建一个元素都为0的目标ROIs
		# 4. 将目标ROIs前面用现在的ROIs填充
		# 5. 剩下的用第一个框填充
		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		# 进行类别预测和边框回归
		# 1. P_cls：该边框属于某一类别的概率
		# 2. P_regr：每一个类别对应的边框回归梯度
		# 3. F:rpn网络得到的卷积后的特征图
		# 4. ROIS:处理得到的区域预选框
		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		# 遍历每一个预选宽（P_cls.shape[1]：预选框的个数）
		for ii in range(P_cls.shape[1]):
			
			# 如果该预选框的最大概率小于设定的阈值（即预测的肯定程度大于一定的值，我们才认为这次的类别的概率预测是有效的）
			# 或者最大的概率出现在背景上，则认为这个预选框是无效的，进行下一次预测。
			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			# 不属于上面的两种情况，取最大的概率处为此边框的类别得到其名称。
			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			# 创建两个list，用于存放不同类别对应的边框和概率
			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			# 1. 得到该预选框的信息
			# 2. 得到类别对应的编号
			# 3. 根据类别编号得到该类的边框回归梯度
			# 4. 对回归梯度进行规整化【？为什么要这么做】
			# 5. 对预测的边框进行修正
			(x, y, w, h) = ROIs[0, ii, :]
			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			
			# 向相应的类里面添加信息.【?为什么要乘 C.rpn_stride，边框的预测都是在特征图上进行的要将其映射到规整后的原图上】
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	# 遍历bboxes里的类，取出某一类的bbox，合并一些重合度较高的选框（剩下来的这些框就是最终识别的结果了）
	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

		# 得到新的预选框，然后将其映射到原图上（训练和测试的图片都是经过规整化的），画框
		# 注：cv2.rectangle五个参数，图片名称、两个对角点坐标、颜色数组、线宽
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			# 向预选框上加上类别和文字
			# 1. 得到框对应的类和相应的概率text(【注：符串格式化(format)】)
			# 2. all_dets只是后来做显示用
			# 3. 得到文本的相关信息（注：getTextSize的用法）
			# 4. 确定文本框的左上点，文本框的边框大小，文本框的底色，加上文字
			# 5. 需要说明的是：cv2画框和添加文本是直接在图片上进行的是改变图片的
			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	# 1. 打印出测试一张图片需要的时间(【注：time.time()即得到当前的时间】)
	# 2. 打印所有的识别信息
	# 3. 显示与保存图片图片，需要特别注意的是cv2.imshow的第一个参数是显示框名称，第二个才是图片(【注：cv2.waitKey(0)暂停程序，显示该图片知道关闭显示窗口】)
	print('Elapsed time = {}'.format(time.time() - st))
	print(all_dets)
	# cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
