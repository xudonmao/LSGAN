from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import dcgan64 as dcgan
import os
import pdb
import lsun_big as lsun

from utils import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', '0.0002', """learning rate""")
tf.app.flags.DEFINE_float('beta1', '0.5', """beta for Adam""")
tf.app.flags.DEFINE_integer('batch_size', '64', """batch size""")
tf.app.flags.DEFINE_integer('y_dim', '1', """y dim""")
tf.app.flags.DEFINE_integer('output_size', '64', """output size""")
tf.app.flags.DEFINE_integer('max_steps', 5000000, """Number of batches to run.""")
tf.app.flags.DEFINE_string('data', 'lsun', """data""")
tf.app.flags.DEFINE_string('loss', 'l2', """data""")
tf.app.flags.DEFINE_string('data_dir', './data/mnist/', """Path to dataset.""")
tf.app.flags.DEFINE_boolean('checkpoint', True,  """Whether to log device placement.""")
tf.app.flags.DEFINE_string('data_path', '../bedroom.tfrecords', """Path to the lsun data file""")

tf.app.flags.DEFINE_string('train_dir', './log/',
                               """Directory where to write event logs """
                                                          """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")




def sess_init():
  init = tf.initialize_all_variables()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(init)
  return sess

def train():
  with tf.Graph().as_default():
    data_set = lsun.LSUN()
    images = data_set.load(FLAGS.data_path)

    global_step = tf.Variable(0, trainable=False)

    _, real_images, label_y, random_z = dcgan.inputs()

    D_logits_real, D_logits_fake, D_logits_fake_for_G, \
    D_sigmoid_real, D_sigmoid_fake, D_sigmoid_fake_for_G = \
      dcgan.inference(images, real_images, label_y, random_z)

    if FLAGS.loss == "sigmoid":
      G_loss, D_loss = dcgan.loss(D_logits_real, D_logits_fake, D_logits_fake_for_G)
    else:
      G_loss, D_loss = dcgan.loss_l2(D_logits_real, D_logits_fake, D_logits_fake_for_G)

    t_vars = tf.trainable_variables()
    G_vars = [var for var in t_vars if 'g_' in var.name]
    D_vars = [var for var in t_vars if 'd_' in var.name]

    G_train_op, D_train_op = dcgan.train(G_loss, D_loss, G_vars, D_vars, global_step)

    sampler = dcgan.sampler(random_z, label_y)

    summary_op = tf.merge_all_summaries()

    sess = sess_init()

    tf.train.start_queue_runners(sess=sess)

    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    
    saver = tf.train.Saver()

    for step in xrange(FLAGS.max_steps):
      batch_z = np.random.uniform(-1, 1, 
                  [FLAGS.batch_size, dcgan.Z_DIM]).astype(np.float32)
      #batch_images, batch_labels = data_set.next_batch()
      batch_labels = np.zeros([FLAGS.batch_size, 1]).astype(np.float32)
      # Update D network
      _, errD = sess.run([D_train_op, D_loss],
          feed_dict={ random_z: batch_z, label_y:batch_labels })


      _, errG = sess.run([G_train_op, G_loss],
          feed_dict={ random_z: batch_z, label_y:batch_labels })

      if step % 100 == 0:
        print "step = %d, errD = %f, errG = %f" % (step, errD, errG)


      if np.mod(step, 1000) == 0:
        
        samples = sess.run(sampler, 
          feed_dict={random_z: batch_z, label_y: batch_labels})
        save_images(samples, [8, 8],
            './samples/train_{:d}.bmp'.format(step))
                            
      if step % 10000 == 0 and FLAGS.checkpoint == True:
        saver.save(sess, 'checkpoint/dcgan-{0}.model'.format(step), global_step)

def main(argv=None):
  train()

if __name__ == "__main__":
  tf.app.run()
