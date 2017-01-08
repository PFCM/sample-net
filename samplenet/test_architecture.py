"""
Tests for `architecture.py`.
Mostly making sure things actually run and spit out the right shape
(it's a bit tricky to check they're actually performing the correct
operations).
"""
import itertools

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import samplenet.architecture as arch


class TestArchitecture(test.TestCase):

    def test_temporal_upsample_list_perforated(self):
        """test with list of inputs"""
        inputs = [tf.random_normal([50, 128]) for _ in range(100)]
        upsampled = arch.temporal_upsample(inputs, 4)

        self.assertIsInstance(upsampled, list)
        # should get out a length 400 list of [50, 128]
        self.assertEqual(400, len(upsampled))
        self.assertEqual([50, 128], upsampled[0].get_shape().as_list())

    def test_temporal_upsample_tensor_perforated(self):
        """test with tensor of inputs"""
        inputs = tf.random_normal([50, 25, 256])
        upsampled = arch.temporal_upsample(inputs, 3)

        self.assertNotIsInstance(upsampled, list)
        # should be [150, 25, 256]
        self.assertEqual(upsampled.get_shape().as_list(), [150, 25, 256])

    def test_temporal_upsample_list_bilinear(self):
        """make sure bilinear resampling is ok"""
        inputs = [tf.random_normal([32, 150]) for _ in range(75)]
        upsampled = arch.temporal_upsample(inputs, 2, method='bilinear')

        self.assertIsInstance(upsampled, list)
        # should get out a length 150 list of [32, 150]
        self.assertEqual(150, len(upsampled))
        self.assertEqual([32, 150], upsampled[0].get_shape().as_list())

    def test_temporal_upsample_list_bicubic(self):
        """make sure bicubic resampling is fine"""
        inputs = [tf.random_normal([16, 44]) for _ in range(12)]
        upsampled = arch.temporal_upsample(inputs, 8, method='bicubic')

        self.assertIsInstance(upsampled, list)
        # should get out a length 96 list of [16, 44]
        self.assertEqual(96, len(upsampled))
        self.assertEqual([16, 44], upsampled[0].get_shape().as_list())

    def test_temporal_upsample_list_nearest_neighbour(self):
        """check the simplest of all works out"""
        inputs = [tf.random_normal([16, 44]) for _ in range(12)]
        upsampled = arch.temporal_upsample(inputs, 8,
                                           method='nearest_neighbor')

        self.assertIsInstance(upsampled, list)
        # should get out a length 96 list of [16, 44]
        self.assertEqual(96, len(upsampled))
        self.assertEqual([16, 44], upsampled[0].get_shape().as_list())

    def test_temporal_upsample_list_bilinear_conv(self):
        """make sure that we can add a convolution on the end"""
        inputs = [tf.random_normal([32, 150]) for _ in range(75)]
        upsampled = arch.temporal_upsample(inputs, 2, method='bilinear_conv')

        self.assertIsInstance(upsampled, list)
        # should get out a length 150 list of [32, 150]
        self.assertEqual(150, len(upsampled))
        self.assertEqual([32, 150], upsampled[0].get_shape().as_list())

    def test_temporal_upsample_list_perforated_different_ksize(self):
        "make sure we can specify the kernel size"
        inputs = [tf.random_normal([50, 128]) for _ in range(100)]
        upsampled = arch.temporal_upsample(inputs, 4, ksize=[5, 1])

        self.assertIsInstance(upsampled, list)
        # should get out a length 400 list of [50, 128]
        self.assertEqual(400, len(upsampled))
        self.assertEqual([50, 128], upsampled[0].get_shape().as_list())

        for var in tf.trainable_variables():
            if 'weights' in var.name:
                self.assertEqual(var.get_shape().as_list(),
                                 [5, 1, 1, 1])

    def test_temporal_upsample_wrong_method(self):
        """test some ways of getting the method wrong"""
        inputs = [tf.random_normal([50, 128]) for _ in range(100)]

        with self.assertRaises(ValueError):
            upsampled = arch.temporal_upsample(inputs, 4, ksize=[5, 1],
                                               method='perforated_conv')

        with self.assertRaises(ValueError):
            upsampled = arch.temporal_upsample(inputs, 4, ksize=[5, 1],
                                               method='something else')

    def test_non_overlapping_frames(self):
        """check we can divide a chunk of samples appropriately"""
        samples = tf.placeholder(tf.float32, [50, 10000])
        splits = arch.non_overlapping_frames(samples, 1000)

        self.assertEqual(len(splits), 10)
        self.assertEqual(splits[0].get_shape(), [50, 1000])

    def test_autoregressive_mlp(self):
        """check the mlp is constructed appropriately"""
        input_frame = tf.random_normal([50, 25600])

        outputs = arch.autoregressive_mlp(input_frame)
        self.assertEqual(outputs.get_shape().as_list(), [50, 256])

    def test_weightnorm(self):
        """checks, roughly, the weight normalisation"""
        wn_var = arch.get_weightnormed_variable([100, 200])

        self.assertEqual(wn_var.get_shape().as_list(), [100, 200])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            var = sess.run(wn_var)
            lengths = np.sqrt(np.sum(var**2, 1))
            self.assertTrue(np.all(np.isclose(lengths, 1)))

    def test_projection_one_to_one(self):
        """test linear projections doesn't go crazy"""
        inputs = [tf.random_normal([50, 100]) for _ in range(75)]
        outputs = arch.linear_projection(inputs)

        self.assertEqual(len(outputs), len(inputs))
        self.assertEqual(outputs[0].get_shape().as_list(),
                         inputs[0].get_shape().as_list())

    def test_projection_change_dims(self):
        """test a not upsampling projection which changes the number of
        features"""
        inputs = [tf.random_normal([50, 100]) for _ in range(75)]
        outputs = arch.linear_projection(inputs, output_size=200)

        self.assertEqual(len(outputs), len(inputs))
        self.assertNotEqual(outputs[0].get_shape().as_list(),
                            inputs[0].get_shape().as_list())
        self.assertEqual(outputs[0].get_shape().as_list(),
                         [50, 200])

    def test_upsampling_projection(self):
        """see if we can upsample with the linear projections."""
        inputs = [tf.random_normal([50, 100]) for _ in range(4)]
        outputs = arch.linear_projection(inputs, rate=4)

        # first make sure we have enough
        self.assertEqual(len(outputs), 16)
        # and they are the right shape
        self.assertEqual(outputs[0].get_shape().as_list(), [50, 100])

        # now check the names of the variables cycles appropriately
        scope_nums = itertools.cycle([0, 1, 2, 3])
        for var in outputs:
            # check the appropriate number is in the name of the variable
            self.assertIn('projection_{}'.format(next(scope_nums)), var.name)

    def test_samplenet_shapes(self):
        """make sure we can construct the graph and get back what we expect"""
        inputs = tf.random_uniform([50, 512*8], dtype=tf.int32, minval=0,
                                   maxval=255)
        inputs = tf.saturate_cast(inputs, tf.uint8)
        outputs = arch.sample_net_train(inputs)

        self.assertEqual(outputs.get_shape().as_list(), [50, 512*8-128+1, 256])
