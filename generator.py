import tensorflow as tf
from utils import custom_operations

#Pyflakes analyzes programs and detects various errors. passive checker of Python programs
if False:
	custom_operations

def BEGAN_Generator(Z, batch_size, scope_name = "generator", reuse_scope = False):
	n = 128

	with tf.variable_scope(scope_name) as scope:
		if reuse_scope:
			scope.reuse_variables()

		layer_1 = (pt.wrap(Z)
			.flatten()
			.fully_connected(8 * 8 * n, activation_fn=tf.nn.elu)
			.fc_batch_norm()
			.reshape([-1, 8, 8, n]))

		conv_1 = (layer_1
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))

		conv_2 = (layer_2
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))

		layer_2 = (conv_2.
					apply(tf.image.resize_nearest_neighbor, [16, 16]))

		conv_3 = (layer_3
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))

		conv_4 = (layer_3
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))


		layer_3 =(conv_4.
					apply(tf.image.resize_nearest_neighbor, [32,32]))

		conv_5 = (layer_3
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))

		conv_6 = (layer_5
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))

		layer_4 =(conv_6.
					apply(tf.image.resize_nearest_neighbor, [64,64]))

		conv_7 = (layer_4
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))

		conv_8 = (layer_7
			.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
			.conv_batch_norm()
			.apply(tf.nn.elu))

		conv_9 = (conv_8
					.custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
					.apply(tf.sigmoid))

		return conv_9.flatten()