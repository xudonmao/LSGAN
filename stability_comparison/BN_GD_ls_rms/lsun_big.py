from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import pdb
from utils import * 

FLAGS = tf.app.flags.FLAGS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000


class LSUN():
  def __init__(self):
    pass
  def read_and_decode(self, filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
  
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    #pdb.set_trace()
    image.set_shape([64*64*3])
    image = tf.reshape(image, [64,64,3])
  
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
  
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 127.5) - 1.0
  
    # Convert label from a scalar uint8 tensor to an int32 scalar.
  
    return image
  
  
  def load(self, filename):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
  
    filename_queue = tf.train.string_input_producer(
        [filename])
  
    image = self.read_and_decode(filename_queue)
    #image = tf.image.resize_images(image, [64,64])

  
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images = tf.train.shuffle_batch(
        [image], batch_size=FLAGS.batch_size, num_threads=10,
        capacity=30000 + 3 * FLAGS.batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=10000)
  
    return images

