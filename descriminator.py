import tensorflow as tf
from utils import custom_operations

#Pyflakes analyzes programs and detects various errors. passive checker of Python programs
if False:
	custom_operations

def  BEGAN_Discriminator(D_I, batch_size, scope_name = "discriminator", resuce_scope = False):
	
    n = 128  # 'n' is number of filters
    with tf.variable_scope(scope_name) as scope:
        if reuse_scope:
            scope.reuse_variables()

    	layer_1 = (pt.wrap[S_I])
     				.reshape()

     	conv_0 = (layer_1
                  	.custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1)
                  	.apply(tf.nn.elu))

        conv_1 = (conv_0
                  	.custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  	.apply(tf.nn.elu))

	return decode_conv_2.flatten()