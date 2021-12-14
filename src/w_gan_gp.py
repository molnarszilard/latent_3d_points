'''
Created on May 22, 2018

Author: Achlioptas Panos (Github ID: optas)
'''

import numpy as np
import time
import tensorflow as tf

from tflearn import is_training
from . gan import GAN


class W_GAN_GP(GAN):
    '''Gradient Penalty.
    https://arxiv.org/abs/1704.00028
    '''

    def __init__(self, name, learning_rate, lam, n_output, noise_dim, discriminator, generator, beta=0.5, gen_kwargs={}, disc_kwargs={}, graph=None):

        GAN.__init__(self, name, graph)
        
        self.noise_dim = noise_dim
        self.n_output = n_output
        self.discriminator = discriminator
        self.generator = generator
    
        with tf.variable_scope(name):
            self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])            # Noise vector.
            self.real_pc = tf.placeholder(tf.float32, shape=[None] + self.n_output)     # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output, **gen_kwargs)
                
            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.real_pc, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope, **disc_kwargs)
            
            
            # Compute WGAN losses
            self.loss_d = tf.reduce_mean(self.synthetic_logit) - tf.reduce_mean(self.real_logit)
            self.loss_g = -tf.reduce_mean(self.synthetic_logit)

            # Compute gradient penalty at interpolated points
            ndims = self.real_pc.get_shape().ndims
            batch_size = tf.shape(self.real_pc)[0]
            alpha = tf.random_uniform(shape=[batch_size] + [1] * (ndims - 1), minval=0., maxval=1.)
            differences = self.generator_out - self.real_pc
            interpolates = self.real_pc + (alpha * differences)

            with tf.variable_scope('discriminator') as scope:
                gradients = tf.gradients(self.discriminator(interpolates, reuse=True, scope=scope, **disc_kwargs)[1], [interpolates])[0]

            # Reduce over all but the first dimension
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, ndims)))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.loss_d += lam * gradient_penalty

            train_vars = tf.trainable_variables()
            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]

            self.opt_d = self.optimizer(learning_rate, beta, self.loss_d, d_params)
            self.opt_g = self.optimizer(learning_rate, beta, self.loss_g, g_params)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims))
    
    def generator_noise_distribution_hack(self, n_samples, ndims, mu, sigma):
        table = np.array(1.35595679e-01  9.96363610e-02  6.17868781e-01  3.34839523e-01
  2.89747864e-01  2.47956604e-01  1.25841022e-01  1.48885131e-01
  4.52625453e-02  1.84123263e-01  3.30897272e-01  1.48935005e-01
  3.00291896e-01  7.11820126e-02  1.28368080e-01 -2.54286639e-03
  5.50497532e-01  8.50341395e-02  4.04721275e-02  2.50388592e-01
  1.43657386e-01  1.39604166e-01  7.29010701e-02  1.30954772e-01
  3.24931771e-01  6.00119075e-03 -1.14781857e-02  8.76261294e-03
  2.65992373e-01  5.21606922e-01  2.81129003e-01  1.47969171e-01
  1.56716004e-01  7.47687459e-01  1.92879093e+00  2.68802106e-01
  5.34469843e-01  5.30086756e-01  5.60561828e-02  8.29004884e-01
  1.97211914e-02  1.22705579e-01  2.66826600e-02  2.09677324e-01
  1.76992297e-01 -1.01472214e-02  1.51021153e-01  3.36964548e-01
  6.68891221e-02  1.75669050e+00  1.37403876e-01  1.74392015e-03
 -3.84432077e-03  5.92521787e-01  2.93566734e-01  9.85375494e-02
  3.16348076e-01 -1.52053349e-02 -1.18494779e-02  5.86861074e-02
  3.73826534e-01  1.80424582e-02  9.76906121e-02  1.09256625e-01
  4.06478316e-01  4.83351320e-01 -1.57447755e-02  2.98775882e-02
  1.23412959e-01  5.64435661e-01  2.43597329e-01  1.85459033e-01
  5.97828507e-01  1.34824753e-01  3.35631877e-01  1.04109287e+00
  2.92893738e-01  2.04309940e-01  5.63133299e-01  6.30199164e-02
  1.59970343e-01  6.62322879e-01  6.74815774e-01  2.77165532e-01
  1.33198619e-01  4.24618870e-02  3.18438649e-01  2.54159629e-01
  8.13457847e-01  7.39102066e-03  2.40427569e-01  1.67281240e-01
  1.97945088e-01  8.91787767e-01  2.58496761e-01  1.16243131e-01
  2.42465734e-03  1.17538698e-01 -1.45597868e-02  5.10022193e-02
  1.01114750e-01 -1.93765759e-03  4.09807235e-01  1.79319620e-01
 -1.82525329e-02  1.76293582e-01  1.64615303e-01  3.70919108e-01
  3.39274973e-01  1.29202291e-01  4.15127903e-01  3.10765296e-01
  1.57139063e-01  3.01227383e-02  3.84545654e-01  1.27674967e-01
  5.74820161e-01  3.37971747e-03  3.98585200e-01  1.24517828e-03
  8.48911516e-03  9.18243080e-02  9.58448946e-02  1.41791776e-02
  5.70046484e-01  7.57475019e-01  4.00464982e-02  7.31340870e-02)
        return table

    def _single_epoch_train(self, train_data, batch_size, noise_params, discriminator_boost=5):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        batch_size = batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        iterations_for_epoch = n_batches / discriminator_boost

        is_training(True, session=self.sess)
        try:
            # Loop over all batches
            for _ in xrange(iterations_for_epoch):
                for _ in range(discriminator_boost):
                    feed, _, _ = train_data.next_batch(batch_size)
                    z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                    feed_dict = {self.real_pc: feed, self.noise: z}
                    _, loss_d = self.sess.run([self.opt_d, self.loss_d], feed_dict=feed_dict)
                    epoch_loss_d += loss_d

                # Update generator.
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                feed_dict = {self.noise: z}
                _, loss_g = self.sess.run([self.opt_g, self.loss_g], feed_dict=feed_dict)
                epoch_loss_g += loss_g

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        epoch_loss_d /= (iterations_for_epoch * discriminator_boost)
        epoch_loss_g /= iterations_for_epoch
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
