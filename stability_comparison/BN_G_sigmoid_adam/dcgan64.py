from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pdb

FLAGS = tf.app.flags.FLAGS
from ops import *
from utils import *


Z_DIM = 1024
#FLAGS.y_dim = 4
C_DIM = 3
GFC_DIM = 1024

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

#g_bn0 = batch_norm(name='g_bn0')
#g_bn1 = batch_norm(name='g_bn1')
#g_bn2 = batch_norm(name='g_bn2')
#g_bn3 = batch_norm(name='g_bn3')
def _activation_summary_list(summary_set):
  for var in summary_set:
    _activation_summary(var)

def _activation_summary(x, reuse=False, for_G=False):
  if for_G == True:
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '_forG/activations', x)
    tf.scalar_summary(tensor_name + '_forG/sparsity', tf.nn.zero_fraction(x))
  elif reuse == False:
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '_real/activations', x)
    tf.scalar_summary(tensor_name + '_real/sparsity', tf.nn.zero_fraction(x))
  else :
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '_fake/activations', x)
    tf.scalar_summary(tensor_name + '_fake/sparsity', tf.nn.zero_fraction(x))


def inputs():
  images = tf.placeholder(tf.float32, 
      shape=[FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, C_DIM])
  real_images = tf.placeholder(tf.float32, 
      shape=[FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, C_DIM])

  label_y = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.y_dim])

  random_z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, Z_DIM])
  return images, real_images, label_y, random_z

def discriminator(image, y = None, reuse=False, for_G=False):
  if reuse:
    tf.get_variable_scope().reuse_variables()
  
  df_dim = 64

  #pdb.set_trace()
  h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
  h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
  h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
  h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
  h4 = linear(tf.reshape(h3, [FLAGS.batch_size, -1]), 1, 'd_h4_logits')
  _activation_summary(h4, reuse, for_G)
  h4_sigmoid = tf.nn.sigmoid(h4, name='d_h4_sigmoid')
  _activation_summary(h4_sigmoid, reuse, for_G)

  return h4, h4_sigmoid

def generator(z, y):
  s = FLAGS.output_size
  s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

  gf_dim = 64

  h0 = tf.nn.relu((tf.reshape(linear(z, gf_dim*8*s16*s16, 'g_h0_lin'), 
         [-1,s16,s16,gf_dim*8])))

  h1 = tf.nn.relu((deconv2d(h0, [FLAGS.batch_size, s8, s8, gf_dim*4], name='g_h1')))

  h2 = tf.nn.relu((deconv2d(h1, [FLAGS.batch_size, s4, s4, gf_dim*2], name='g_h2')))

  h3 = tf.nn.relu((deconv2d(h2, [FLAGS.batch_size, s2, s2, gf_dim*1], name='g_h3')))

  h4 = deconv2d(h3, [FLAGS.batch_size, s, s, C_DIM], name='g_h4')

  return tf.nn.tanh(h4)

def sampler(z, y):
  tf.get_variable_scope().reuse_variables()

  s = FLAGS.output_size
  s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

  gf_dim = 64

  h0 = tf.nn.relu((tf.reshape(linear(z, gf_dim*8*s16*s16, 'g_h0_lin'), 
         [-1,s16,s16,gf_dim*8])))

  h1 = tf.nn.relu((deconv2d(h0, [FLAGS.batch_size, s8, s8, gf_dim*4], name='g_h1')))

  h2 = tf.nn.relu((deconv2d(h1, [FLAGS.batch_size, s4, s4, gf_dim*2], name='g_h2')))

  h3 = tf.nn.relu((deconv2d(h2, [FLAGS.batch_size, s2, s2, gf_dim*1], name='g_h3')))

  h4 = deconv2d(h3, [FLAGS.batch_size, s, s, C_DIM], name='g_h4')

  return tf.nn.tanh(h4)

def inference(image, real_images, label_y, random_z):

  G_image = generator(random_z, label_y)

  D_logits_real, D_sigmoid_real = discriminator(image, label_y)

  D_logits_fake, D_sigmoid_fake = discriminator(G_image, label_y, True)

  D_logits_fake_for_G, D_sigmoid_fake_for_G = discriminator(G_image, label_y, True, True)

  return D_logits_real, D_logits_fake, D_logits_fake_for_G, D_sigmoid_real, D_sigmoid_fake, D_sigmoid_fake_for_G

def loss_l2(D_logits_real, D_logits_fake, D_logits_fake_for_G):
  G_loss = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake_for_G - tf.ones_like(D_logits_fake_for_G))) 

  D_loss_real = tf.reduce_mean(tf.nn.l2_loss(D_logits_real - tf.ones_like(D_logits_real))) 

  D_loss_fake = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake - tf.zeros_like(D_logits_fake))) 

  D_loss = D_loss_real + D_loss_fake

  tf.scalar_summary("D_loss", D_loss)
  tf.scalar_summary("D_loss_real", D_loss_real)
  tf.scalar_summary("D_loss_fake", D_loss_fake)
  tf.scalar_summary("G_loss", G_loss)

  return G_loss, D_loss


def loss(D_logits_real, D_logits_fake, D_logits_fake_for_G):

  G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
             D_logits_fake_for_G, tf.ones_like(D_logits_fake_for_G)))

  D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  D_logits_real, tf.ones_like(D_logits_real)))

  D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  D_logits_fake, tf.zeros_like(D_logits_fake)))

  D_loss = D_loss_real + D_loss_fake

  tf.scalar_summary("D_loss", D_loss)
  tf.scalar_summary("D_loss_real", D_loss_real)
  tf.scalar_summary("D_loss_fake", D_loss_fake)
  tf.scalar_summary("G_loss", G_loss)
  
  return G_loss, D_loss

def train(G_loss, D_loss, G_vars, D_vars, global_step):

  G_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 
  D_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 

  #G_train_op = G_optim.minimize(G_loss, var_list=G_vars)
  #D_train_op = D_optim.minimize(D_loss, var_list=D_vars)

  G_grads = G_optim.compute_gradients(G_loss, var_list=G_vars)
  D_grads = D_optim.compute_gradients(D_loss, var_list=D_vars)

  for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

  for grad, var in D_grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)
  for grad, var in G_grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  G_train_op = G_optim.apply_gradients(G_grads, global_step=global_step)
  D_train_op = D_optim.apply_gradients(D_grads)

  return G_train_op, D_train_op



