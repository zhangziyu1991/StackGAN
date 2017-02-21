from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify
from misc.config import cfg


class CondGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.image_shape = image_shape
        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

        # Since D is only used during training, we build a template
        # for safe reuse the variables during computing loss for fake/real/wrong images
        # We do not do this for G,
        # because batch_norm needs different options for training and testing
        if cfg.GAN.NETWORK_TYPE == "default":
            print('Using default network.')
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        elif cfg.GAN.NETWORK_TYPE == "simple":
            print('Using simple network.')
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image_simple()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # g-net
    # conditioning augmentation happens here!
    # def generate_condition(self, c_var):
    #     conditions =\
    #         (pt.wrap(c_var).
    #          # additional convolutions begin
    #          custom_conv2d(self.ef_dim / 2, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #
    #          custom_conv2d(self.ef_dim / 2, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #
    #          max_pool(2, 2, edges=pt.PAD_VALID).
    #
    #          custom_conv2d(self.ef_dim, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #
    #          custom_conv2d(self.ef_dim, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #
    #          max_pool(2, 2, edges=pt.PAD_VALID).
    #          # additional convolution end
    #          flatten().
    #          custom_fully_connected(self.ef_dim * 8).
    #          apply(leaky_rectify, leakiness=0.2).
    #          custom_fully_connected(self.ef_dim * 2).
    #          # comment this
    #          apply(leaky_rectify, leakiness=0.2))
    #          # comment this
    #     mean = conditions[:, :self.ef_dim]
    #     log_sigma = conditions[:, self.ef_dim:]
    #     return [mean, log_sigma]

    def generate_condition(self, c_var):
        return c_var

    # This is where fake images are generated!
    # def generator(self, z_var):
    #     node1_0 =\
    #         (pt.wrap(z_var).
    #          # additional convolutions begin
    #          # custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
    #          # conv_batch_norm().
    #          # apply(tf.nn.relu).
    #          # custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
    #          # conv_batch_norm().
    #          # apply(tf.nn.relu).
    #          # custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
    #          # conv_batch_norm())
    #          # additional convolution end
    #          flatten().
    #          custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
    #          fc_batch_norm().
    #          reshape([-1, self.s16, self.s16, self.gf_dim * 8]))
    #     node1_1 = \
    #         (node1_0.
    #          custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm())
    #     node1 = \
    #         (node1_0.
    #          apply(tf.add, node1_1).
    #          apply(tf.nn.relu))
    #
    #     node2_0 = \
    #         (node1.
    #          # custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
    #          apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
    #          custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm())
    #     node2_1 = \
    #         (node2_0.
    #          custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm())
    #     node2 = \
    #         (node2_0.
    #          apply(tf.add, node2_1).
    #          apply(tf.nn.relu))
    #
    #     output_tensor = \
    #         (node2.
    #          # custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
    #          apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
    #          custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          # custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
    #          apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
    #          custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
    #          conv_batch_norm().
    #          apply(tf.nn.relu).
    #          # custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
    #          apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
    #          custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
    #          apply(tf.nn.tanh))
    #     return output_tensor

    def generator(self, z_var):
        # 64 * 64
        node_1 = pt.wrap(z_var).custom_conv2d(self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2)
        # 32 * 32
        node_2 = node_1.apply(leaky_rectify, leakiness=0.2).custom_conv2d(self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm()
        # 16 * 16
        node_3 = node_2.apply(leaky_rectify, leakiness=0.2).custom_conv2d(2 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm()
        # 8 * 8
        node_4 = node_3.apply(leaky_rectify, leakiness=0.2).custom_conv2d(4 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm()
        # 4 * 4
        node_5 = node_4.apply(leaky_rectify, leakiness=0.2).custom_conv2d(4 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm()
        # 2 * 2
        node_5 = node_4.apply(leaky_rectify, leakiness=0.2).custom_conv2d(4 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm()

        # 1 * 1
        node_6 = node_5.apply(leaky_rectify, leakiness=0.2).custom_conv2d(4 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm()

        # 2 * 2
        node_7_0 = node_6.apply(tf.nn.relu).custom_deconv2d(4 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm().dropout(0.5)
        node_7 = tf.concat(3, [node_7_0, node_5])
        # 4 * 4
        node_8_0 = node_7.apply(tf.nn.relu).custom_deconv2d(4 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm().dropout(0.5)
        node_8 = tf.concat(3, [node_8_0, node_4])
        # 8 * 8
        node_9_0 = node_8.apply(tf.nn.relu).custom_deconv2d(4 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm().dropout(0.5)
        node_9 = tf.concat(3, [node_9_0, node_3])
        # 16 * 16
        node_10_0 = node_9.apply(tf.nn.relu).custom_deconv2d(2 * self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm().dropout(0.5)
        node_10 = tf.concat(3, [node_10_0, node_2])
        # 32 * 32
        node_11_0 = node_10.apply(tf.nn.relu).custom_deconv2d(self.gf_dim, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm().dropout(0.5)
        node_11 = tf.concat(3, [node_11_0, node_1])
        # 64 * 64
        node_12 = node_11.apply(tf.nn.relu).custom_deconv2d(3, k_h=4, k_w=4, d_h=2, d_w=2).conv_batch_norm().apply(tf.nn.tanh)

        return node_12

    def generator_simple(self, z_var):
        output_tensor =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             # custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             # custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             # custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             # custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def get_generator(self, z_var):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var)
        elif cfg.GAN.NETWORK_TYPE == "simple":
            return self.generator_simple(z_var)
        else:
            raise NotImplementedError

    # d-net
    def context_embedding(self):
        template = (pt.template("input").
                    # additional convolutions begin
                    custom_conv2d(self.ef_dim / 2, k_h=3, k_w=3, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(tf.nn.relu).

                    custom_conv2d(self.ef_dim / 2, k_h=3, k_w=3, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(tf.nn.relu).

                    max_pool(2, 2, edges=pt.PAD_VALID).

                    custom_conv2d(self.ef_dim, k_h=3, k_w=3, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(tf.nn.relu).

                    custom_conv2d(self.ef_dim, k_h=3, k_w=3, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(tf.nn.relu).

                    max_pool(2, 2, edges=pt.PAD_VALID).
                    custom_fully_connected(self.ef_dim * 8).
                    apply(leaky_rectify, leakiness=0.2).
                    # additional convolution end
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template

    def d_encode_image(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def d_encode_image_simple(self):
        template = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2))

        return template

    # def discriminator(self):
    #     template = \
    #         (pt.template("input").  # 128*9*4*4
    #          custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
    #          conv_batch_norm().
    #          apply(leaky_rectify, leakiness=0.2).
    #          # custom_fully_connected(1))
    #          custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))
    #
    #     return template

    def discriminator(self):
        template = (pt.template("input").
                    # 64 x 64
                    custom_conv2d(self.df_dim, k_h=4, k_w=4, d_h=2, d_w=2).
                    apply(leaky_rectify, leakiness=0.2).
                    # 32 x 32
                    custom_conv2d(2 * self.df_dim, k_h=4, k_w=4, d_h=2, d_w=2).
                    conv_batch_norm().
                    apply(leaky_rectify, leakiness=0.2).
                    # 16 x 16
                    custom_conv2d(4 * self.df_dim, k_h=4, k_w=4, d_h=2, d_w=2).
                    conv_batch_norm().
                    apply(leaky_rectify, leakiness=0.2).
                    # 8 x 8
                    custom_conv2d(8 * self.df_dim, k_h=4, k_w=4, d_h=2, d_w=2).
                    conv_batch_norm().
                    apply(leaky_rectify, leakiness=0.2).
                    # 4 x 4
                    custom_conv2d(8 * self.df_dim, k_h=4, k_w=4, d_h=2, d_w=2).
                    conv_batch_norm().
                    apply(leaky_rectify, leakiness=0.2).
                    # 2 x 2
                    custom_conv2d(8 * self.df_dim, k_h=4, k_w=4, d_h=2, d_w=2).
                    conv_batch_norm().
                    apply(leaky_rectify, leakiness=0.2).
                    # 1 x 1
                    custom_conv2d(1, k_h=4, k_w=4, d_h=2, d_w=2))

        return template

    def get_discriminator(self, x_var, c_var):
        # x_code = self.d_encode_img_template.construct(input=x_var)
        #
        # c_code = self.d_context_template.construct(input=c_var)
        # c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        # c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])
        #
        # x_c_code = tf.concat(3, [x_code, c_code])

        x_c_var = tf.concat(3, [x_var, c_var])

        # return self.discriminator_template.construct(input=x_c_code)
        return self.discriminator_template.construct(input=x_c_var)
