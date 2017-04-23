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
			.fully_connected()
			.fc_batch_norm()
			.reshape([-1, 8, 8, n]))

		conv_1 = (layer_1
			.custom_conv2d()
			.conv_batch_norm()
			.apply(tf.nn.elu))

		conv_2 = (layer_2
			.custom_conv2d()
			.conv_batch_norm()
			.apply(tf.nn.elu))

		return conv_2.flatten()