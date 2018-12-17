from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


def union(au, bu, area_intersection):
	'''
	并的区域就好求的多了：两个区域的总面积减去交面积就可以了
	'''
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	'''
	得到交区域的左上和右下角坐标。如果右下坐标大于左上坐标则求交面积
	'''
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	'''
	if语句是要求右下角的点大于左上角的点，属于逻辑检查
	intersection：计算两个面积的交
	union：计算两个面积的并
	最后返回交并并比【分母加上1e-6，是为了防止分母为0】
	'''

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
	'''
	得到最短边，另一个边按比列缩放。
	'''
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	'''
	创建类需要有__init__函数，以指定类的基本信息self.classes:去除那些类别数为0的类，
	比如背景。还有就是字典有key()这个关键词,也有items(),更重要的是for循环与if的合用，
	这个是从一个list里仅取出我们需要的数据的重要方法itertools.cycle()、next()的组合
	也是十分重要的。它可以无限地反复地从数组中取值,第一个是将一个list做成一个迭代器对象，
	第二个是取出这个值。
	'''
	def __init__(self, class_count):
		# 忽略了样本数为零的那些类
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):
		'''
		当输入一张图片时，决定是否要跳过该图片。该图片中包含需要的类返回False，否则返回True
		【注：cls_name = bbox['class']这是如何用键来取出值】
		'''
		# 默认不包含需要的类
		class_in_img = False

		# 遍历图片的预选框，看是否包含需要的类
		for bbox in img_data['bboxes']:
			
			# 拿到预选框所对应的分类
			cls_name = bbox['class']

			# 如果预选框框出来的类别就是当前图片所属的类别
			if cls_name == self.curr_class:

				# 标记为包含需要的类
				class_in_img = True

				# 迭代到下一个图片的类别信息
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
	'''
	函数输入：
	训练信息类(包含一张图片的路径，bbox的坐标和对应的分类)
	图片信息
	图片宽度
	图片高度（重新计算bboxes要用）
	规整化后图片宽度
	规整化后图片高度
	计算特征图大小函数

	函数输出：
	是否包含类
	相应的回归梯度
	'''
	# 初始化预选框的相关信息
	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	

	# 计算特征图的尺寸
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height)

	# anchor_ratios的个数
	n_anchratios = len(anchor_ratios)
	
	# 初始化框的输出对象
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

	# 图像数据中包含的框的个数
	num_bboxes = len(img_data['bboxes'])

	# 对框的状态随机初始化
	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

	# 将最短边规整到指定的长度后，相应的边框长度也需要发生变化。
	# 注意gta的存储形式是（x1,x2,y1,y2）而不是（x1,y1,x2,y2）
	gta = np.zeros((num_bboxes, 4))
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# 遍历所有可能的预选框组合
	for anchor_size_idx in range(len(anchor_sizes)):
		
		# 遍历预选框的所有的缩放比例
		for anchor_ratio_idx in range(n_anchratios):

			# 得到一种预选框的宽度和高度（然后来和给定的标注框进行比较）
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
			
			# 遍历一种预选框组合下，由锚点衍生出的所有预选框
			# output_width，output_height：特征图的宽度与高度
			# downscale：将特征图坐标映射到原图的比例
			# if语句是将超出图片的框删去
			# 【注：现在我们确定了一个预选框组合有确定了中心点那就是唯一确定一个框了，接下来就是来确定这个宽的性质了：是否包含物体、如包含物体其回归梯度是多少】
			# 要确定以上两个性质，每一个框都需要遍历图中的所有bboxes
			for ix in range(output_width):					
				# 当前框的x边界坐标	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# 忽略跨越图像边界的框					
				if x1_anc < 0 or x2_anc > resized_width:
					continue
					
				for jy in range(output_height):

					# 当前框的y边界坐标
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# 忽略跨越图像边界的框
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# bbox_type表示锚点应该是一个目标  
					bbox_type = 'neg'

					# 这是对应于当前(x,y)的最优的IOU，同时也是当前的锚点
					# 注意，这与GT bbox的最优IOU不同 
					best_iou_for_loc = 0.0

					for bbox_num in range(num_bboxes):
						
						# 计算该预选框与bbox的交并比
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						
						# 如果现在的交并比curr_iou大于该bbox最好的交并比或者大于给定的阈值则求下列参数，即，对最优参数进行更新，得到预测的GT
						# 这些参数是后来要用的即回归梯度
						# tx:两个框中心的宽的距离与预选框宽的比
						# ty:同tx
						# tw:bbox的宽与预选框宽的比
						# th:同理
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0

							tx = (cx - cxa) / (x2_anc - x1_anc)
							ty = (cy - cya) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						# 如果相交的不是背景，那么进行一系列更新
						# 关于bbox的相关信息更新
						# 预选框的相关更新：如果交并比大于阈值这是pos(代表了这是一个有效的预选框)
						# best_iou_for_loc：其记录的是有最大交并比为多少和其对应的回归梯度
						# num_anchors_for_bbox[bbox_num]：记录的是bbox拥有的pos预选框的个数
						# 如果小于最小阈值是neg，在这两个之间是neutral
						# 需要注意的是：判断一个框为neg需要其与所有的bbox的交并比都小于最小的阈值
						if img_data['bboxes'][bbox_num]['class'] != 'bg':

							# 所有GT box应映射到一个anchor box，所以我们跟踪哪个anchor box是最好的。
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# 如果IOU大于0.7，我们设置为特征锚点 (如果有另一个更好的box并不重要，它只是表示重叠。)
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# 我们更新回归层目标如果当前的（x，y）和锚点的位置有最好的IOU
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# 如果IOU大于0.3小于0.7, 这是模糊的，不列入目标中。
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# NEG和POS之间的灰色地带 设置为中性
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# 当结束对所有的bbox的遍历时，来确定该预选宽的性质。
					# y_is_box_valid：该预选框是否可用（nertual就是不可用的）
					# y_rpn_overlap：该预选框是否包含物体
					# y_rpn_regr:回归梯度
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr

	# 如果有一个bbox没有pos的预选宽和其对应，这找一个与它交并比最高的anchor的设置为pos（从中立的框中查找）
	for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0:
			# 没有框与其存在交并比
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	# 将深度变到第一位，给向量增加一个维度
	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	# np.where:输出满足条件的位置，先行后列。默认返回数组中True的位置
	# np.logical_and:对每一个位置返回一个True或者False。只有当都是True时才返回True
	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])

	# 得到正负预选框的位置
	# 从可用的预选框中选择num_regions 
	# 如果pos的个数大于num_regions / 2，则将多下来的地方置为不可用。如果小于pos不做处理
	# 接下来将pos与neg总是超过num_regions个的neg预选框置为不可用
	# 【注：这个随机选取的过程是很值得学习的。首先：在给定的范围内（样本个数）随机选取指定个数的序号；其次：将这些序号对应的位置处置为0】
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	# y_rpn_cls：是否包含类，其前半段是该anchor是否可用
	# y_rpn_regr：回归梯度，前半段包含是否有效
	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
	'''
	函数输入：
	图片信息
	类别统计信息
	训练信息类
	计算输出特征图大小的函数
	keras用的什么内核
	是否为训练

	函数输出：
	图片
	数据对象：第一个是是否包含对象，第二个是回归梯度
	增强后的图片信息
	'''
	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)

	# 这个类是确定是否要跳过图片，以达到类平衡
	sample_selector = SampleSelector(class_count)

	while True:
		if mode == 'train':
			# 打乱图片数据（多维矩阵中，只对第一维（行）做打乱顺序操作）
			np.random.shuffle(all_img_data)

		# 遍历图片数据
		for img_data in all_img_data:
			try:
				
				# 决定是否要跳过该图片，如果包含需要训练的类别，则不跳过。continue是结束本次循环，开始下一次循环
				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue

				# augment是用来增强图片的，主要是图片的选中操作
				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

				# 显示判断经过增强后的图片逻辑上不存在问题，即图片宽高与实际一致
				# get_new_img_size：faster_rcnn要把图片的最短边规整到600（可以设置成其它）.这个是得到新图片的尺寸
				(width, height) = (img_data_aug['width'], img_data_aug['height'])
				(rows, cols, _) = x_img.shape

				assert cols == width
				assert rows == height

				# 计算得到对图片进行标准缩放后的尺寸
				(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

				# 对图片进行尺寸的变换，把图片的最短边规整到600
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

				# 得到每一张图片的每一个点的两个特性以供RPN网络训练
				# y_rpn_cls：是否包含物体
				# y_rpn_regr：回归梯度为多少
				try:
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
				except:
					continue

				# 将RGB图片变为BGR,因为公开训练好的VGG模型是按照这个训练的
				# 减去均值理由同上
				# 将深度变为第一个维度
				# 给图片增加一个维度
				# 给回归梯度除上一个规整因子
				# 如果用的是tf内核,还是要把深度调到最后一位了

				x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor

				x_img = np.transpose(x_img, (2, 0, 1))
				x_img = np.expand_dims(x_img, axis=0)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				if backend == 'tf':
					x_img = np.transpose(x_img, (0, 2, 3, 1))
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				# yield 是一个类似 return 的关键字，迭代一次遇到yield时就返回yield后面(右边)的值。
				# 重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。
				# 带有 yield 的函数不再是一个普通函数，而是一个生成器generator使用时是用next()调用。
				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

			except Exception as e:
				print(e)
				continue
