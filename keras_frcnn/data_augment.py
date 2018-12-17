import cv2
import numpy as np
import copy


def augment(img_data, config, augment=True):
	'''
	assert:断言函数，是python用来确认的。
	assert--in:断言是否在其中，assert--==:断言是否相等。
	这两个经常用到【注：augment--增加、增大的意思】
	'''
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	# 我们需要一个深拷贝，当进行图像增强的时候，不会改变原有的信息。
	img_data_aug = copy.deepcopy(img_data)

	# 利用opencv读取图片数据
	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2] # shape:先行后列

		# 图像水平翻转，对应的bbox的对角坐标也进行水平翻转，翻转概率为50%
		# np.random.randint(0, 2)：产生一个0到2的随机数【random.randint()】
		# cv2.flip(img, 1)：将图片延y轴翻转（参数大于0，等于0时延x轴翻转【cv2.flip】）
		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		# 图像垂直翻转，对应的bbox的对角坐标也进行垂直翻转，翻转概率为50%
		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		# np.random.choice([0,90,180,270],1)[0]：从给定的list选择一个数【如果没有后面[0],那么它返回的还是一个list】
		# np.transpose(img, (1,0,2)):高宽互换
		# 结束本次if操作用pass、结束本次for操作用contine(上文用到过)
		# 图像按90度旋转，对应的bbox的对角坐标也进行90度旋转，旋转概率为50%
		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			# 根据上文的操作，变换坐标
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	# 得到增强后图片的宽高
	# 返回增强后的图片信息和图片
	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img
