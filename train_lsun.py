from __future__ import absolute_import
from __future__ import division
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import vgg 
import os
import lsun

from utils import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', '0.001', """learning rate""")
tf.app.flags.DEFINE_float('beta1', '0.5', """beta for Adam""")
tf.app.flags.DEFINE_integer('batch_size', '64', """batch size""")
tf.app.flags.DEFINE_integer('c_dim', '3', """c dimsion""")
tf.app.flags.DEFINE_integer('z_dim', '1024', """z dimsion""")
tf.app.flags.DEFINE_integer('output_size', '112', """output size""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_string('log_dir', './log/', """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', """Directory where to write the checkpoint""")
tf.app.flags.DEFINE_string('data_path', '../church112.tfrecords', """Path to the lsun data file""")


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

    random_z = vgg.inputs()

    D_logits_real, D_logits_fake, D_logits_fake_for_G, \
    D_sigmoid_real, D_sigmoid_fake, D_sigmoid_fake_for_G = \
      vgg.inference(images, random_z)

    G_loss, D_loss = vgg.loss_l2(D_logits_real, D_logits_fake, D_logits_fake_for_G)

    t_vars = tf.trainable_variables()
    G_vars = [var for var in t_vars if 'g_' in var.name]
    D_vars = [var for var in t_vars if 'd_' in var.name]

    G_train_op, D_train_op = vgg.train(G_loss, D_loss, G_vars, D_vars, global_step)

    sampler = vgg.sampler(random_z)

    summary_op = tf.merge_all_summaries()

    sess = sess_init()

    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
    
    saver = tf.train.Saver()

    for step in xrange(1, FLAGS.max_steps+1):
      batch_z = np.random.uniform(-1, 1, 
                  [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)

      _, errD = sess.run([D_train_op, D_loss],
          feed_dict={ random_z: batch_z})

      _, errG = sess.run([G_train_op, G_loss],
          feed_dict={ random_z: batch_z})

      if step % 100 == 0:
        print "step = %d, errD = %f, errG = %f" % (step, errD, errG)


      if np.mod(step, 1000) == 0:
        samples = sess.run(sampler, 
          feed_dict={random_z: batch_z})
        save_images(samples, [8, 8],
            './samples/train_{:d}.bmp'.format(step))
                            
        #      if step % 1000 == 0:
        #        summary_str = sess.run(summary_op, 
        #            feed_dict={random_z: batch_z})
        #        summary_writer.add_summary(summary_str, step)

      if step % 10000 == 0:
        saver.save(sess, '{0}/vgg-{1}.model'.format(FLAGS.checkpoint_dir, step), global_step)

def main(argv=None):
  os.system('mkdir -p samples')
  os.system('mkdir -p {0}'.format(FLAGS.checkpoint_dir))
  os.system('mkdir -p {0}'.format(FLAGS.log_dir))
  train()

if __name__ == "__main__":
  tf.app.run()
