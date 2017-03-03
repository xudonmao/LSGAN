from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import scipy.misc


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('output_size', '112', """output size""")
tf.app.flags.DEFINE_string('source_dir', './lsun/bedroom_train', """Source Dir""")
tf.app.flags.DEFINE_string('target_dir', './target', """Target Dir""")


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(path):
  with open(path) as f:
    file_set = [line.strip('\n') for line in f]

  num_examples = len(file_set)

  writer = tf.python_io.TFRecordWriter(FLAGS.target_dir + "/bedroom.tfrecords")
  for i in range(num_examples):
    image = scipy.misc.imread(file_set[i], mode='RGB')
    image = scipy.misc.imresize(image, [FLAGS.output_size, FLAGS.output_size])
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    if i % 1000 == 0:
      print '%d processed' % i
  writer.close()


os.system('mkdir -p {0}'.format(FLAGS.target_dir))
os.system('find {0} -type f > file.list'.format(FLAGS.source_dir))
convert('file.list')

