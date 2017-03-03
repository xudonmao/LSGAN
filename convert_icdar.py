import numpy as np
import tensorflow as tf
import scipy.misc
import os
import struct as st

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('output_size', '28', """output size""")
tf.app.flags.DEFINE_string('source_dir', './icdar_chinese/1.0train-gb1/', """Source Dir""")
tf.app.flags.DEFINE_string('target_dir', './icdar_data/', """Target Dir""")

def load_one_char(f, data_dict):
  first_unit = f.read(4)
  if first_unit == "":
    return False

  sample_size = st.unpack('i', first_unit)
  
  c1 = st.unpack('c', f.read(1))[0]
  c2 = st.unpack('c', f.read(1))[0]
  u = unicode(c1 + c2, 'gbk')
  
  width = st.unpack('H', f.read(2))[0]
  height = st.unpack('H', f.read(2))[0]
  
  image = np.fromfile(f, np.uint8, width*height).reshape((height, width))
  output_size = FLAGS.output_size
  image_resized = scipy.misc.imresize(image, [output_size, output_size])

  if data_dict.has_key(u):
    data_dict[u] = np.concatenate([data_dict[u], image_resized])
  else:
    data_dict[u] = image_resized

  return True

def load_one_file(path, data_dict):
  with open(path, "rb") as f:
    while True:
      succ = load_one_char(f, data_dict)
      if succ == False:
        break

def convert(data_dir):
  data_dict = {}
  for gnt in os.listdir(data_dir):
    if gnt[-3:len(gnt)] == "gnt":
      load_one_file(data_dir + gnt, data_dict)
  for (k,v) in data_dict.items():
    num = v.shape[0] / FLAGS.output_size
    v = v.reshape([num, FLAGS.output_size, FLAGS.output_size, 1])
    np.save('{0}/{1}.npy'.format(FLAGS.target_dir, k.encode('utf-8')), v)


os.system('mkdir -p {0}'.format(FLAGS.target_dir))

convert('/home/folbul/sata/data/icdar_chinese/1.0train-gb1/')
