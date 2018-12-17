from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
	'''
	计算rpn网络的框回归的梯度
	'''
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		'''
		x是真值与预测值的差值
		x_abs是差值的绝对值
		'''
		if K.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)

			# x_abs小于1为True
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			# 差值绝对值小于1时0.5X^2,大于1的绝对值减0.5然后相加
			# 在乘上是否要计算这个loss
			# 求和在除以个数，求均值
			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	'''
	RPN分类损失，先传递框的数量
	K.binary_crossentropy：计算输出张量和目标张量的交叉熵
	求交叉熵，在乘拉入计算的loss,求和在除个数
	'''
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'tf':


			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	'''
	最后的回归损失
	'''
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	'''
	最后的分类损失
	'''
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
