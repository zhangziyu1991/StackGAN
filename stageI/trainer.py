from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar

from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []

        self.fake_images = None

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images'
        )

        # This becomes the sketch entrance
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )

        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )

    # Given image embedding, get condition vector
    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            print('Using conditional augmentation.')
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon
            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            print('Not using conditional augmentation.')
            c = mean
            kl_loss = 0

        return c, cfg.TRAIN.COEFF.KL * kl_loss

    def init_opt(self):
        self.build_placeholder()

        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("g_net"):
                # ####get output from G network################################
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_z", z))
                fake_images = self.model.get_generator(tf.concat(1, [c, z]))

            # ####get discriminator_loss and generator_loss ###################
            discriminator_loss, generator_loss = \
                self.compute_losses(self.images,
                                    self.wrong_images,
                                    fake_images,
                                    self.embeddings)
            generator_loss += kl_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #######Total loss for build optimizers###########################
            self.prepare_trainer(generator_loss, discriminator_loss)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("g_net", reuse=True):
                self.sampler()
            # self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")

    def sampler(self):
        c, _ = self.sample_encoded_context(self.embeddings)
        if cfg.TRAIN.FLAG:
            print('Using all-zero z vector.')
            z = tf.zeros([self.batch_size, cfg.Z_DIM])  # Expect similar BGs
        else:
            print("Using same z's for all samples.")
            z = tf.random_normal([1, cfg.Z_DIM], seed=123)
            z = tf.tile(z, [self.batch_size, 1])
            # print("Using different z's for different samples.")
            # z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        self.fake_images = self.model.get_generator(tf.concat(1, [c, z]))

    def compute_losses(self, images, wrong_images, fake_images, embeddings):
        real_logit = self.model.get_discriminator(images, embeddings)
        wrong_logit = self.model.get_discriminator(wrong_images, embeddings)
        fake_logit = self.model.get_discriminator(fake_images, embeddings)

        real_d_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(real_logit,
                                                    tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(wrong_logit,
                                                    tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        fake_d_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.zeros_like(fake_logit))
        fake_d_loss = tf.reduce_mean(fake_d_loss)
        if cfg.TRAIN.B_WRONG:
            discriminator_loss = \
                real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
            self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        self.log_vars.append(("d_loss_real", real_d_loss))
        self.log_vars.append(("d_loss_fake", fake_d_loss))

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)

        return discriminator_loss, generator_loss

    def prepare_trainer(self, generator_loss, discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]

        generator_opt = tf.train.AdamOptimizer(self.generator_lr,
                                               beta1=0.5)
        self.generator_trainer = \
            pt.apply_optimizer(generator_opt,
                               losses=[generator_loss],
                               var_list=g_vars)
        discriminator_opt = tf.train.AdamOptimizer(self.discriminator_lr,
                                                   beta1=0.5)
        self.discriminator_trainer = \
            pt.apply_optimizer(discriminator_opt,
                               losses=[discriminator_loss],
                               var_list=d_vars)
        self.log_vars.append(("g_learning_rate", self.generator_lr))
        self.log_vars.append(("d_learning_rate", self.discriminator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.scalar_summary(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.histogram_summary(k, v))

        self.g_sum = tf.merge_summary(all_sum['g'])
        self.d_sum = tf.merge_summary(all_sum['d'])
        self.hist_sum = tf.merge_summary(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row, :, :, :]
            row_img = [img]  # real image
            # for col in range(1):
            row_img.append(img_var[row, :, :, :])
            # each row has 1 real image + 1 fake images
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.expand_dims(tf.concat(0, stacked_img), 0)
        current_img_summary = tf.image_summary(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        fake_sum_train, superimage_train = \
            self.visualize_one_superimage(self.fake_images[: 10],
                                          self.images[: 10],
                                          10,
                                          "train")
        # self.visualize_one_superimage(self.fake_images[:n * n], self.images[:n * n], n, "train")
        fake_sum_test, superimage_test = \
            self.visualize_one_superimage(self.fake_images[10: 20],
                                          self.images[10: 20],
                                          10,
                                          "test")
        # fake_sum_test, superimage_test = \
        #     self.visualize_one_superimage(self.fake_images[:n * n],
        #                                   self.images[:n * n],
        #                                   n, "test")
        self.superimages = tf.concat(0, [superimage_train, superimage_test])
        self.image_summary = tf.merge_summary([fake_sum_train, fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n, epoch):
        images_train, _, embeddings_train, captions_train, _ = \
            self.dataset.train.next_batch(10, cfg.TRAIN.NUM_EMBEDDING)
        # images_train = self.preprocess(images_train, n)
        # embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ = \
            self.dataset.test.next_batch(10, 1)
        # images_test = self.preprocess(images_test, n)
        # embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)

        # 2 * n * n images, half train, half test
        # in case of batch size > 2 * n *n, pad extra test images
        # if self.batch_size > 2 * n * n:
        images_pad, _, embeddings_pad, _, _ = self.dataset.test.next_batch(self.batch_size - 20, 1)
        images = np.concatenate([images, images_pad], axis=0)
        embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

        feed_dict = {self.images: images, self.embeddings: embeddings}
        gen_samples, img_summary = sess.run([self.superimages, self.image_summary], feed_dict)

        # save images generated for train and test captions
        print('Saving %s/train.jpg' % self.log_dir)
        scipy.misc.imsave('{}/train_epoch{}.jpg'.format(self.log_dir, epoch), gen_samples[0])
        print('Saving %s/test.jpg' % self.log_dir)
        scipy.misc.imsave('{}/test_epoch{}.jpg'.format(self.log_dir, epoch), gen_samples[1])

        # pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()

        return img_summary

    def build_model(self, sess):
        self.init_opt()
        sess.run(tf.initialize_all_variables())

        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            restore_vars = tf.all_variables()
            # all_vars = tf.all_variables()
            # restore_vars = [var for var in all_vars if
            #                 var.name.startswith('g_') or
            #                 var.name.startswith('d_')]
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.all_variables(),
                                       keep_checkpoint_every_n_hours=2)

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(self.log_dir,
                                                        sess.graph)

                keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                        # print(k, v)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                num_embedding = cfg.TRAIN.NUM_EMBEDDING
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(number_example / self.batch_size)
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch != 0:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)

                        # training d
                        images, wrong_images, embeddings, _, _ = \
                            self.dataset.train.next_batch(self.batch_size,
                                                          num_embedding)
                        feed_dict = {self.images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.generator_lr: generator_lr,
                                     self.discriminator_lr: discriminator_lr}

                        # train d
                        feed_out = [self.discriminator_trainer,
                                    self.d_sum,
                                    self.hist_sum,
                                    log_vars]
                        _, d_sum, hist_sum, log_vals = sess.run(feed_out,
                                                                feed_dict)
                        summary_writer.add_summary(d_sum, counter)
                        summary_writer.add_summary(hist_sum, counter)
                        all_log_vals.append(log_vals)

                        # train g
                        feed_out = [self.generator_trainer,
                                    self.g_sum]
                        _, g_sum = sess.run(feed_out,
                                            feed_dict)
                        summary_writer.add_summary(g_sum, counter)

                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" % \
                                            (self.checkpoint_dir,
                                             self.exp_name,
                                             str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)


                    img_sum = self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY, epoch+1)
                    summary_writer.add_summary(img_sum, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")

    def save_super_images(self, images, sampled_batch, filenames, save_dir, subset):
        super_image = []
        for i in range(self.batch_size):
            save_path = '{}-1real-{}samples/{}/{}.jpg' % (save_dir, cfg.TEST.NUM_COPY, subset, filenames[i])
            folder = save_path[:save_path.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            super_image_row = [images[i]]
            for j in range(cfg.TEST.NUM_COPY):
                super_image_row.append(sampled_batch[j][i])

            super_image.append(np.concatenate(super_image_row, axis=1))

        super_image = np.concatenate(super_image, axis=0)
        print('Saving to: {}'.format(save_path))
        print('Super image shape: {}'.format(super_image.shape))
        scipy.misc.imsave(save_path, super_image)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples: {}'.format(dataset._num_examples))
        while count < dataset.num_examples:
            start = count % dataset.num_examples
            images, embeddings, filenames = dataset.next_batch_test(self.batch_size, start)
            print('count = ', count, 'start = ', start)

            sampled_batch = []
            for j in range(cfg.TEST.NUM_COPY):
                sampled = sess.run(self.fake_images, {self.embeddings: embeddings})
                sampled_batch.append(sampled)
            self.save_super_images(images, sampled_batch, filenames, save_dir, subset)

            count += self.batch_size

    def evaluate(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.device('/gpu:{}'.format(cfg.GPU_ID)):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from {}".format(self.model_path))
                    saver = tf.train.Saver(tf.all_variables())
                    saver.restore(sess, self.model_path)
                    self.eval_one_dataset(sess, self.dataset.test, self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")
