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
        rand=np.random.normal(mu, sigma, (n_samples, ndims))
        return rand

    def generator_noise_distribution_chair(self, n_samples, ndims, mu, sigma):
        hack_ret = np.random.normal(mu, sigma, (n_samples, ndims))
        for i in range(ndims):
                pcd = np.array([0.28073502, 0.1503914,  0.38994616, 0.2630428,  0.532085,   0.22919035,
 0.3342267,  0.11030181, 0.5701463 , 0.14549139, 0.16218595, 0.42155525,
 0.34410486, 0.20673719, 0.17087935, 0.02732829, 0.7399808 , 0.34281835,
 0.12100132, 0.35935256, 0.15533727, 0.15616655, 0.01909163, 0.36948657,
 0.4757328 , 0.0185906 , 0.3278726 , 0.10999837, 0.38742477, 0.30486527,
 0.48095626, 0.3989781 , 0.4660827 , 0.4954255 , 1.7783834 , 0.2925363,
 0.3516795 , 0.32469493, 0.42035016, 0.48876792, 0.2894983 , 0.1320645,
 0.1651369 , 0.20107032, 0.1741252 , 0.33571392, 0.22191876, 0.29733256,
 0.15338707, 1.3306084 , 0.03495988, 0.64194417, 0.05891776, 0.70556444,
 0.0596283 , 0.5389465 , 0.32437342, 0.18382023, 0.5117878 , 0.29542363,
 0.41964468, 0.0134186 , 0.06228535, 0.50052226, 0.16559374, 0.3967151,
 0.46894646, 0.00655708, 0.08396538, 0.62604284, 0.5671755 , 0.18340686,
 0.39299414, 0.03832432, 0.30434775, 0.6545097 , 0.08712709, 0.25314128,
 0.31018746, 0.22943398, 0.43724287, 0.26042423, 0.6412914 , 0.3923845,
 0.31479162, 0.03882255, 0.32985556, 0.2052794 , 0.71381736, 0.02485568,
 0.66422826, 0.2855417  ,0.26435554, 0.5776424 , 0.26670685, 0.21125698,
 0.09206337, 0.38745928 ,0.08205982, 0.09278123, 0.27389902 ,0.25978693,
 0.39538866, 0.1782232 , 0.16152474, 0.29983065, 0.63776934 ,0.4022663,
 0.29587433, 0.5022019 , 0.48172787, 0.30260748, 0.15423173, 0.22405598,
 0.4085314 , 0.14033821, 0.36663944, 0.42518225, 0.5672016 , 0.26955438,
 0.00799436, 0.367347  , 0.47533816, 0.26097646, 0.20004085, 0.18241987,
 0.21803391, 0.07547506])
                hack_ret[i]=pcd
        return hack_ret


    def generator_noise_distribution_chair_ml(self, n_samples, ndims, mu, sigma):
        hack_ret = np.random.normal(mu, sigma, (n_samples, ndims))
        for i in range(ndims):
                pcd = np.array([0.28159785,  0.14690901,  0.130701,    0.0829639,   0.16941056,  0.45057732,
  0.21158911,  0.10503028,  0.1239112,   0.19839391,  0.11634448,  0.17587692,
  0.20123845,  0.4786873,   0.12330851,  0.00766717,  0.11209862,  0.13755505,
  0.02267103,  0.33606964,  0.14393602,  0.12131983,  0.13013272,  0.08778526,
  0.42397994,  0.024064,    0.17903563,  0.2161101,   0.41645944,  0.49267334,
  0.47840765,  0.21827704,  0.83395505,  0.47282183,  1.8427669,   0.28679627,
  0.4531452,   0.32675844,  0.12721601,  0.6792021,   0.02616465,  0.13528001,
  0.19442844,  0.18250246,  0.17480409,  0.10512371,  0.25991863,  0.15376359,
  0.07865861,  1.5208445,   0.08554129,  0.6454526,   0.3402902,   0.44390553,
  0.62492716,  0.3003559,   0.6125308,   0.25404412,  0.2267111,   0.05745699,
  0.09014106,  0.04589858,  0.05275496,  0.13114913,  0.2684681,   0.5149206,
  0.0879762,   0.32578135,  0.15176809,  0.62479794,  0.2655516,   0.16246578,
  0.6548763,   0.09589986,  0.30391166,  1.270857,    0.0322324,   0.2626336,
  0.7250053,   0.17772567,  0.39394003,  0.56747144,  0.08228932,  0.3366747,
  0.12289043,  0.04970315,  0.32535103,  0.21184224,  0.85487604,  0.05242675,
  0.44518262,  0.34488127,  0.19367333,  0.48968154,  0.2886697,   0.44637293,
 -0.00886492,  0.23778234,  0.2844409,   0.07275479,  0.11743175,  0.20914353,
  0.37135783,  0.17623943,  0.03521395,  0.44102585,  0.40315607,  0.41274425,
  0.34188795,  0.14366737,  0.7475804,   0.31081247,  0.4096948,   0.2066242,
  0.4504868,   0.13312663,  0.13902667,  0.14975095,  0.4373459,   0.06343585,
  0.02113085,  0.19756187,  0.4279578,   0.06403501,  0.45624536,  0.20133847,
  0.05350068,  0.10238274])
                hack_ret[i]=pcd
        return hack_ret

    def generator_noise_distribution_table(self, n_samples, ndims, mu, sigma):
        table_ret = np.random.normal(mu, sigma, (n_samples, ndims))
	for i in range(ndims):
	        table = np.array([1.35595679e-01,  9.96363610e-02,  6.17868781e-01,  3.34839523e-01,
  2.89747864e-01,  2.47956604e-01,  1.25841022e-01,  1.48885131e-01,
  4.52625453e-02,  1.84123263e-01,  3.30897272e-01,  1.48935005e-01,
  3.00291896e-01,  7.11820126e-02,  1.28368080e-01, -2.54286639e-03,
  5.50497532e-01,  8.50341395e-02,  4.04721275e-02,  2.50388592e-01,
  1.43657386e-01,  1.39604166e-01,  7.29010701e-02,  1.30954772e-01,
  3.24931771e-01,  6.00119075e-03, -1.14781857e-02,  8.76261294e-03,
  2.65992373e-01,  5.21606922e-01,  2.81129003e-01,  1.47969171e-01,
  1.56716004e-01,  7.47687459e-01,  1.92879093e+00,  2.68802106e-01,
  5.34469843e-01,  5.30086756e-01,  5.60561828e-02,  8.29004884e-01,
  1.97211914e-02,  1.22705579e-01,  2.66826600e-02,  2.09677324e-01,
  1.76992297e-01, -1.01472214e-02,  1.51021153e-01,  3.36964548e-01,
  6.68891221e-02,  1.75669050e+00,  1.37403876e-01,  1.74392015e-03,
 -3.84432077e-03,  5.92521787e-01,  2.93566734e-01,  9.85375494e-02,
  3.16348076e-01, -1.52053349e-02, -1.18494779e-02,  5.86861074e-02,
  3.73826534e-01,  1.80424582e-02,  9.76906121e-02,  1.09256625e-01,
  4.06478316e-01,  4.83351320e-01, -1.57447755e-02,  2.98775882e-02,
  1.23412959e-01,  5.64435661e-01,  2.43597329e-01,  1.85459033e-01,
  5.97828507e-01,  1.34824753e-01,  3.35631877e-01,  1.04109287e+00,
  2.92893738e-01,  2.04309940e-01,  5.63133299e-01,  6.30199164e-02,
  1.59970343e-01,  6.62322879e-01,  6.74815774e-01,  2.77165532e-01,
  1.33198619e-01,  4.24618870e-02,  3.18438649e-01,  2.54159629e-01,
  8.13457847e-01,  7.39102066e-03,  2.40427569e-01,  1.67281240e-01,
  1.97945088e-01,  8.91787767e-01,  2.58496761e-01,  1.16243131e-01,
  2.42465734e-03,  1.17538698e-01, -1.45597868e-02,  5.10022193e-02,
  1.01114750e-01, -1.93765759e-03,  4.09807235e-01,  1.79319620e-01,
 -1.82525329e-02,  1.76293582e-01,  1.64615303e-01,  3.70919108e-01,
  3.39274973e-01,  1.29202291e-01,  4.15127903e-01,  3.10765296e-01,
  1.57139063e-01,  3.01227383e-02,  3.84545654e-01,  1.27674967e-01,
  5.74820161e-01,  3.37971747e-03,  3.98585200e-01,  1.24517828e-03,
  8.48911516e-03,  9.18243080e-02,  9.58448946e-02,  1.41791776e-02,
  5.70046484e-01,  7.57475019e-01,  4.00464982e-02,  7.31340870e-02])
		table_ret[i]=table
        return table_ret

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
