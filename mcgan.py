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

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')

#def _activation_summary(x, reuse=False, for_G=False):
#  return 
#
#  if for_G == True:
#    tensor_name = x.op.name
#    tf.histogram_summary(tensor_name + '_forG/activations', x)
#    tf.scalar_summary(tensor_name + '_forG/sparsity', tf.nn.zero_fraction(x))
#  elif reuse == False:
#    tensor_name = x.op.name
#    tf.histogram_summary(tensor_name + '_real/activations', x)
#    tf.scalar_summary(tensor_name + '_real/sparsity', tf.nn.zero_fraction(x))
#  else :
#    tensor_name = x.op.name
#    tf.histogram_summary(tensor_name + '_fake/activations', x)
#    tf.scalar_summary(tensor_name + '_fake/sparsity', tf.nn.zero_fraction(x))


def inputs():
  images = tf.placeholder(tf.float32, 
      shape=[FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])

  label_y = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.y_dim])

  random_z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
  return images, label_y, random_z

def discriminator(image, y, reuse=False, for_G=False):
  with tf.variable_scope('discriminator'): 
    if reuse:
      tf.get_variable_scope().reuse_variables()

    y_size = 256
    y_linear = linear(y, y_size, 'd_liner')
    y_conv = tf.reshape(y_linear, [FLAGS.batch_size, 1, 1, y_size])

    h0 = lrelu(conv2d(image, FLAGS.c_dim + y_size, name='d_h0_conv'), name='d_h0_relu')
    #_activation_summary(h0, reuse, for_G)
    h0 = conv_cond_concat(h0, y_conv)

    h1 = lrelu(d_bn1(conv2d(h0, 64 + y_size, name='d_h1_conv')), name='d_h1_relu')
    #_activation_summary(h1, reuse, for_G)
    h1 = tf.reshape(h1, [FLAGS.batch_size, -1])            
    h1 = tf.concat(axis=1, values=[h1, y_linear])
    
    h2 = lrelu(d_bn2(linear(h1, 1024, 'd_h2_lin')), name='d_h2_relu')
    #_activation_summary(h2, reuse, for_G)
    h2 = tf.concat(axis=1, values=[h2, y_linear])

    h3_logits = linear(h2, 1, 'd_h3_logits')
    #_activation_summary(h3_logits,reuse, for_G)

    h3_sigmoid = tf.nn.sigmoid(h3_logits, name='d_h3_sigmoid')
    #_activation_summary(h3_logits, reuse, for_G)

    return h3_logits, h3_sigmoid

def generator(z, y):
  with tf.variable_scope('generator'):
    s2, s4 = int(FLAGS.output_size/2), int(FLAGS.output_size/4)
    
    y_size = 256
    y_linear = linear(y, y_size, 'g_liner')
 
    z = tf.concat(axis=1, values=[z, y_linear])

    h1 = tf.nn.relu(g_bn1(linear(z, 128*s4*s4, 'g_h1_lin')), name='g_h1_relu')            
    #_activation_summary(h1)
    h1 = tf.reshape(h1, [FLAGS.batch_size, s4, s4, 128])

    h2 = tf.nn.relu(g_bn2(deconv2d(h1, 
           [FLAGS.batch_size, s2, s2, 128], name='g_h2')),name='g_h2_relu')
    #_activation_summary(h2)

    h3 = tf.nn.sigmoid(deconv2d(h2, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='g_h3'), name='g_h3_sigmoid')
    #_activation_summary(h3)

    return h3

def sampler(z, y):
  with tf.variable_scope('generator'):
    tf.get_variable_scope().reuse_variables()

    s2, s4 = int(FLAGS.output_size/2), int(FLAGS.output_size/4)

    y_size = 256
    y_linear = linear(y, y_size, 'g_liner')
 
    z = tf.concat(axis=1, values=[z, y_linear])

    h1 = tf.nn.relu(g_bn1(linear(z, 128*s4*s4, 'g_h1_lin'), train=False))            
    h1 = tf.reshape(h1, [FLAGS.batch_size, s4, s4, 128])

    h2 = tf.nn.relu(g_bn2(deconv2d(h1, 
           [FLAGS.batch_size, s2, s2, 128], name='g_h2'), train=False))

    return tf.nn.sigmoid(deconv2d(h2, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='g_h3'))

def inference(image, label_y, random_z):

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

  #tf.scalar_summary("D_loss", D_loss)
  #tf.scalar_summary("D_loss_real", D_loss_real)
  #tf.scalar_summary("D_loss_fake", D_loss_fake)
  #tf.scalar_summary("G_loss", G_loss)

  return G_loss, D_loss

def train(G_loss, D_loss, G_vars, D_vars, global_step):

  G_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 
  D_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 

  G_grads = G_optim.compute_gradients(G_loss, var_list=G_vars)
  D_grads = D_optim.compute_gradients(D_loss, var_list=D_vars)

  #for var in tf.trainable_variables():
  #      tf.histogram_summary(var.op.name, var)

  #for grad, var in D_grads:
  #  if grad is not None:
  #    tf.histogram_summary(var.op.name + '/gradients', grad)
  #for grad, var in G_grads:
  #  if grad is not None:
  #    tf.histogram_summary(var.op.name + '/gradients', grad)

  G_train_op = G_optim.apply_gradients(G_grads, global_step=global_step)
  D_train_op = D_optim.apply_gradients(D_grads)

  return G_train_op, D_train_op



