import tensorflow as tf
import prettytensor as pt
from utils import custom_operations

#Pyflakes analyzes programs and detects various errors. passive checker of Python programs
if False:
	custom_operations

def  BEGAN_Discriminator(D_I, batch_size,hidden_size, scope_name = "discriminator", reuse_scope = False):
	
    n = 128  # 'n' is number of filters
    with tf.variable_scope(scope_name) as scope:
        if reuse_scope:
            scope.reuse_variables()

        layer_1 = (pt.wrap(D_I)
     		   		.reshape([-1,64,64,3]))

       	conv_1 = (layer_1
                    	.custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1)
                    	.apply(tf.nn.elu))

        conv_2 = (conv_1
              	.custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
              	.apply(tf.nn.elu))

        conv_3 = (conv_2
                .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                .apply(tf.nn.elu))
         
        layer_2 = (conv_3
                    .conv2d(3, 2 * n, stride = 2)
                    .apply(tf.nn.elu))

        conv_4 = (layer_2
                .custom_conv2d(2 * n, k_h=3, k_w=3, d_h=1, d_w=1)
                .apply(tf.nn.elu))

        conv_5 = (conv_4
                .custom_conv2d(2 * n, k_h=3, k_w=3, d_h=1, d_w=1)
                .apply(tf.nn.elu))

        layer_3 = (conv_5
                     .conv2d(3, 3 * n, stride=2)
                     .apply(tf.nn.elu))

        conv_6 = (layer_3
                    .custom_conv2d(3 * n, k_h=3, k_w=3, d_h=1, d_w=1)
                    .apply(tf.nn.elu))

        conv_7 = (conv_6
                    .custom_conv2d(3 * n, k_h=3, k_w=3, d_h=1, d_w=1)
                    .apply(tf.nn.elu))

        layer_4 = (conv_7
                     .conv2d(3, 4 * n, stride=2)
                     .apply(tf.nn.elu))

        conv_8 = (layer_4
                    .custom_conv2d(4 * n, k_h=3, k_w=3, d_h=1, d_w=1)
                    .apply(tf.nn.elu))

        conv_9 = (conv_8
                    .custom_conv2d(4 * n, k_h=3, k_w=3, d_h=1, d_w=1)
                    .apply(tf.nn.elu))

        layer_5 = (conv_9
                     .flatten()
                     .fully_connected(hidden_size, activation_fn=tf.nn.elu))

        decode_layer_1 = (layer_5
                          .flatten()
                          .fully_connected(8 * 8 * n, activation_fn  = tf.nn.elu)
                          .reshape([-1, 8, 8, n]))

        decode_conv_1 = (decode_layer_1
                          .custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
                          .conv_batch_norm()
                          .apply(tf.nn.elu))
        decode_conv_2 = (decode_conv_1
                          .custom_conv2d(n, k_h = 3, k_w = 3, d_h = 1, d_w = 1)
                          .conv_batch_norm()
                          .apply(tf.nn.elu))
  	
        decode_layer_2 = (decode_conv_2
                          .apply(tf.image.resize_nearest_neighbor, [16, 16]))

        decode_conv_3 = (decode_layer_2
                         .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                         .conv_batch_norm()
                         .apply(tf.nn.elu))

        decode_conv_4 = (decode_conv_3
                         .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                         .conv_batch_norm()
                         .apply(tf.nn.elu))

        decode_layer_3 = (decode_conv_4
                          .apply(tf.image.resize_nearest_neighbor, [32, 32]))

        decode_conv_5 = (decode_layer_3
                         .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                         .conv_batch_norm()
                         .apply(tf.nn.elu))

        decode_conv_6 = (decode_conv_5
                         .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                         .conv_batch_norm()
                         .apply(tf.nn.elu))

        decode_layer_4 = (decode_conv_6
                          .apply(tf.image.resize_nearest_neighbor, [64, 64]))

        decode_conv_7 = (decode_layer_4
                         .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                         .apply(tf.nn.elu))

        decode_conv_8 = (decode_conv_7
                         .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                         .conv_batch_norm()
                         .apply(tf.nn.elu))

        decode_conv_9 = (decode_conv_8
                         .custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1)
                         .apply(tf.sigmoid))

        return decode_conv_9.flatten()