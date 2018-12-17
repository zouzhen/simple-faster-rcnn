import numpy as np
import pdb
import math
import time
from . import data_generators
import copy


def calc_iou(R, img_data, C, class_mapping):
	'''
	函数输入：
	预选宽
	图片信息
	训练信息
	类别与映射数字之间的关系

	函数输出：
	筛选后的预选框
	对应的类别
	相应的回归梯度
	交并比
	'''

	# 得到图片的基本信息,并将图片的最短边规整到相应的长度,并将bboxes的长度做相应的变化
	bboxes = img_data['bboxes']
	(width, height) = (img_data['width'], img_data['height'])

	# 获取用于调整大小的图像尺寸
	(resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

	gta = np.zeros((len(bboxes), 4))

	for bbox_num, bbox in enumerate(bboxes):
		# 获取GT box坐标，并据此调整图像大小。
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
		gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	IoUs = [] # 仅仅用来测试

	# 遍历所有的预选框R，它并不需要做规整。由于RPN网络预测的框就是基于最短框被规整后的
	for ix in range(R.shape[0]):
		(x1, y1, x2, y2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1

		# 将每一个预选框与所有的bboxes求交并比，记录最大交并比。用来确定该预选框的类别。
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		# 对最佳的交并比作不同的判断
		# 当最佳交并比小于最小的阈值时，放弃概框。因为，交并比太低就说明是很好判断的背景没必要训练。当大于最小阈值时，则保留相关的边框信息
		# 当在最小和最大之间，就认为是背景。有必要进行训练。
		# 大于最大阈值时认为是物体，计算其边框回归梯度
		if best_iou < C.classifier_min_overlap:
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			x_roi.append([x1, y1, w, h])
			IoUs.append(best_iou)

			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				# hard negative example
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']
				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

				cx = x1 + w / 2.0
				cy = y1 + h / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		# 得到该类别对应的数字
		# 将该数字对应的地方置为1【one-hot】
		# 将该类别加入到y_class_num
		# coords是用来存储边框回归梯度的，labels来决定是否要加入计算loss中
		class_num = class_mapping[cls_name]
		class_label = len(class_mapping) * [0]
		class_label[class_num] = 1
		y_class_num.append(copy.deepcopy(class_label))
		coords = [0] * 4 * (len(class_mapping) - 1)
		labels = [0] * 4 * (len(class_mapping) - 1)

		# 如果不是背景的话，计算相应的回归梯度
		if cls_name != 'bg':
			label_pos = 4 * class_num
			sx, sy, sw, sh = C.classifier_regr_std
			coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
			labels[label_pos:4+label_pos] = [1, 1, 1, 1]
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))
		else:
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))

	# 返回数据
	if len(x_roi) == 0:
		return None, None, None, None

	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs

def apply_regr(x, y, w, h, tx, ty, tw, th):
	try:
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.
		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h
	except OverflowError:
		return x, y, w, h
	except Exception as e:
		print(e)
		return x, y, w, h

def apply_regr_np(X, T):
	# 得到相关的信息，而如何修正预选框就和当初是如何训练的有关系了
	try:
		x = X[0, :, :]
		y = X[1, :, :]
		w = X[2, :, :]
		h = X[3, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tw = T[2, :, :]
		th = T[3, :, :]

		# 计算回归梯度：
		# tx：是实际框的中心点cx与预选宽的中心点cxa的差值，除以预选框的宽度。ty是同理
		# tw:是实际框的宽度的log除预选宽的宽度，th同理
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy

		w1 = np.exp(tw.astype(np.float64)) * w
		h1 = np.exp(th.astype(np.float64)) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)
		return np.stack([x1, y1, w1, h1])
	except Exception as e:
		print(e)

		# 当对预选框进行修正的时候，就是训练的逆过程
		return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	'''
	输入参数的含义：
	框
	每个框对应的概率大小（是否有物体）
	重合度阈值
	选取框的个数

	函数输出
	boxes, probs
	框（x1,y1,x2,y2）的形式
	对应的概率

	如果没有任何框, 返回一个空的列表
	对输入的数据进行确认
	不能为空
	左上角的坐标小于右下角
	数据类型的转换
	'''

	if len(boxes) == 0:
		return []

	# 获取boxes的边界坐标
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# 如果边界框为整数形式，将它们转换为浮点数形式
	# 由于我们将做一系列的分工，所以这是很重要的
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# pick（拾取）用来存放边框序号
	# 计算框的面积
	# 初始化picked索引
	# probs按照概率从小到大排序
	pick = []
	area = (x2 - x1) * (y2 - y1)
	idxs = np.argsort(probs)

	# 按照概率从大到小取出框，且框的重合度不可以高于overlap_thresh。代码的思路是这样的：
	# 每一次取概率最大的框（即idxs最后一个）
	# 删除掉剩下的框中重和度高于overlap_thresh的框
	# 直到取满max_boxes为止
	while len(idxs) > 0:
		# 获取索引列表中的最后一个索引，并将索引值添加到所选索引列表中。
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# 取出idxs队列中最大概率框的序号，将其添加到pick中
		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int

		# 计算取出来的框与剩下来的框区域的交集
		area_union = area[i] + area[idxs[:last]] - area_int

		# 计算重叠率
		overlap = area_int/(area_union + 1e-6)

		# 从具有的索引列表中删除所有索引 
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# 返回pick内存取的边框和对应的概率
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9):
	'''
	输入参数的含义：
	框对应的概率（是否存在物体）
	每个框对应的回归梯度
	C信息对象
	维度组织形式
	是否进行边框回归（一般为True）
	要取出多少个框
	重叠度的阈值

	函数输出：
	返回指定个数的预选框，形式是（x1,y1,x2,y2）
	'''
	regr_layer = regr_layer / C.std_scaling

	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios

	assert rpn_layer.shape[0] == 1

	if dim_ordering == 'th':
		(rows,cols) = rpn_layer.shape[2:]

	elif dim_ordering == 'tf':
		(rows, cols) = rpn_layer.shape[1:3]

	curr_layer = 0
	if dim_ordering == 'tf':
		A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
	elif dim_ordering == 'th':
		A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

	# 遍历anchor_size，再遍历anchor_ratio
	for anchor_size in anchor_sizes: 
		for anchor_ratio in anchor_ratios:

			# 得到框的长宽在原图上的映射
			anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
			anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride

			# 得到相应尺寸的框对应的回归梯度，将深度都放到第一个维度
			# 注1：regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]当某一个维度的取值为一个值时，那么新的变量就会减小一个维度
			# 注2：curr_layer代表的是特定长度和比例的框所代表的编号
			if dim_ordering == 'th':
				regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
			else:
				regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
				regr = np.transpose(regr, (2, 0, 1))

			# clos:宽度，rows:高度，这一步是为了得到每一个anchor对应是坐标
			X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

			# 得到anchor对应的（x,y,w,h）
			A[0, :, :, curr_layer] = X - anchor_x/2
			A[1, :, :, curr_layer] = Y - anchor_y/2
			A[2, :, :, curr_layer] = anchor_x
			A[3, :, :, curr_layer] = anchor_y

			# 使用regr对anchor所确定的框进行修正
			if use_regr:
				A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

			# 这段代码主要是对修正后的边框一些不合理的地方进行矫正。
			# 如，边框回归后的左上角和右下角的点不能超过图片外，框的宽高不可以小于0
			# 注：得到框的形式是（x1,y1,x2,y2）
			A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
			A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
			A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

			A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
			A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
			A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

			curr_layer += 1

	# 得到all_boxes形状是（n,4），和每一个框对应的概率all_probs形状是（n,）
	all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
	all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

	# 删除掉一些不合理的点，即右下角的点值要小于左上角的点值
	# 注：np.where() 返回位置信息，这也是删除不符合要求点的一种方法
	# np.delete(all_boxes, idxs, 0)最后一个参数是在哪一个维度删除
	x1 = all_boxes[:, 0]
	y1 = all_boxes[:, 1]
	x2 = all_boxes[:, 2]
	y2 = all_boxes[:, 3]

	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

	all_boxes = np.delete(all_boxes, idxs, 0)
	all_probs = np.delete(all_probs, idxs, 0)

	# 最后是根据要求选取指定个数的合理预选框。
	result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

	return result
