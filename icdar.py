# -*- coding: UTF-8 –*-
from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import tensorflow as tf
import scipy.misc
import pdb
import os
import cv2
import struct as st

FLAGS = tf.app.flags.FLAGS


class ICDAR():
  def __init__(self, data_dir):
    self._index_in_epoch = 0

    self.char_set = []
    for i, npy in enumerate(os.listdir(data_dir)):
      self.char_set.append(data_dir + npy)

    random.shuffle(self.char_set)

  def load(self, load_num):
    for i in range(min(load_num, len(self.char_set))):
      X = np.load(self.char_set[i])
      if i == 0:
        trX = X
        trY = np.ones(X.shape[0], dtype='int32') * i
      else:
        trX = np.concatenate([trX, X])
        trY = np.concatenate([trY, np.ones(X.shape[0], dtype='int32') * i])
    X = trX
    y = trY
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), FLAGS.y_dim), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i,y[i]] = 1.0

    self._images = X/255.
    self._labels = y_vec
    self._num_examples = y_vec.shape[0]

  def next_batch(self):
    start = self._index_in_epoch
    self._index_in_epoch += FLAGS.batch_size
    if self._index_in_epoch > self._num_examples:
      seed = np.random.randint(100000)
      np.random.seed(seed)
      np.random.shuffle(self._images)
      np.random.seed(seed)
      np.random.shuffle(self._labels)
      start = 0
      self._index_in_epoch = FLAGS.batch_size
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
  


