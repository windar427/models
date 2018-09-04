"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


# 这个代码是块中每个小的操作,dropout 扩展了模型的通用性，泛化性
def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
	current = slim.batch_norm(current, scope=scope + '_bn')
	# 先把数据进行b优化
	current = tf.nn.relu(current)
	# relu激活
	current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
	# 卷积
	current = slim.dropout(current, scope=scope + '_dropout')
	# 输出
	return current


# 这个代码是DenseBlock
def block(net, layers, growth, scope='block'):
	'''Denseblock封装 对denseblock进行统一处理'''
	for idx in range(layers):
		bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1], scope=scope + '_conv1x1' + str(idx))
		tmp = bn_act_conv_drp(bottleneck, growth, [3, 3], scope=scope + '_conv3x3' + str(idx))
		net = tf.concat(axis=3, values=[net, tmp])
		# concat 把3个conv后的结果拼接
		# 把2个conv合在一起
	return net


def densenet(images, num_classes=1001, is_training=False, dropout_keep_prob=0.8, scope='densenet'):
	"""Creates a variant of the densenet model.

	  images: A batch of `Tensors` of size [batch_size, height, width, channels].
	  num_classes: the number of classes in the dataset.
	  is_training: specifies whether or not we're currently training the model.
		This variable will determine the behaviour of the dropout layer.
	  dropout_keep_prob: the percentage of activation values that are retained.
	  prediction_fn: a function to get predictions out of logits.
	  scope: Optional variable_scope.

	Returns:
	  logits: the pre-softmax activations, a tensor of size
		[batch_size, `num_classes`]
	  end_points: a dictionary from components of the network to the corresponding
		activation.
	"""
	# config = tf.ConfigProto(allow_soft_placement=True)
	growth = 12
	compression_rate = 0.5

	def reduce_dim(input_feature):
		return int(int(input_feature.shape[-1]) * compression_rate)

	end_points = {}

	with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
		with slim.arg_scope(bn_drp_scope(is_training=is_training, keep_prob=dropout_keep_prob)) as ssc:
			pass
			##########################
			# Put your code here.
			##########################
			print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

			# 1.init
			net = slim.conv2d(images, 2 * growth, [7, 7], stride=2, activation_fn=None, padding='same', scope='init')
			net = slim.max_pool2d(net, [3, 3], stride=2, scope="init_pool")

			# 2.DenseBlock, 进行12次小block， 每个小block包括bn ，relu， 卷积3*3，输出为 growth， 如果growth太大导致输入增长太快，运行慢
			# growth太小， 可能有些信息没有提取出来
			net = block(net, 6, growth, scope='block1')

			# 3.两block之间的卷积和池化，变换层，1.瓶颈层： 1*1 降维，BN_Relu_Conv。。。。 DenseNet-B
			net = bn_act_conv_drp(net, growth, [1, 1], scope='con2d1')
			# net = slim.conv2d(net, growth, [3,3], scope="con2d1")
			net = slim.avg_pool2d(net, [2, 2], stride=2, scope="pool1")

			# 2.DenseBlock
			net = block(net, 12, growth, scope='block2')

			# 3.两block之间的卷积和池化
			net = bn_act_conv_drp(net, growth, [1, 1], scope='con2d2')
			net = slim.avg_pool2d(net, [2, 2], stride=2, scope="pool2")

			# 2.DenseBlock
			net = block(net, 24, growth, scope='block3')

			# 3.两block之间的卷积和池化
			net = bn_act_conv_drp(net, growth, [1, 1], scope='con2d3')
			net = slim.avg_pool2d(net, [2, 2], stride=2, scope="pool3")

			# 2.DenseBlock
			net = block(net, 16, growth, scope='block4')

			# 3.全局池化
			net = slim.avg_pool2d(net, [7, 7], stride=1, scope="pool4")

			net = slim.flatten(net, scope='Flatten')

			logits = slim.fully_connected(net, num_classes, activation_fn=None, scope="logits")
			end_points['softmax'] = tf.nn.softmax(logits, name="Predictions")

	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

	return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
	keep_prob = keep_prob if is_training else 1
	with slim.arg_scope(
			[slim.batch_norm],
			scale=True, is_training=is_training, updates_collections=None):
		with slim.arg_scope(
				[slim.dropout],
				is_training=is_training, keep_prob=keep_prob) as bsc:
			return bsc


def densenet_arg_scope(weight_decay=0.004):
	"""Defines the default densenet argument scope.

	Args:
	  weight_decay: The weight decay to use for regularizing the model.

	Returns:
	  An `arg_scope` to use for the inception v3 model.
	"""
	with slim.arg_scope(
			[slim.conv2d],
			weights_initializer=tf.contrib.layers.variance_scaling_initializer(
				factor=2.0, mode='FAN_IN', uniform=False),
			activation_fn=None, biases_initializer=None, padding='same',
			stride=1) as sc:
		return sc


densenet.default_image_size = 224
