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
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1_1 = batch_norm(name='g_bn1_1')
g_bn1_2 = batch_norm(name='g_bn1_2')
g_bn2_1 = batch_norm(name='g_bn2_1')
g_bn2_2 = batch_norm(name='g_bn2_2')
g_bn3_1 = batch_norm(name='g_bn3_1')
g_bn4_1 = batch_norm(name='g_bn4_1')

#def _activation_summary(x, reuse=False, for_G=False):
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
  random_z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
  return random_z

def discriminator(image, reuse=False, for_G=False):
  with tf.variable_scope('discriminator'): 
    if reuse:
      tf.get_variable_scope().reuse_variables()
  
    df_dim = 64

    h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
    h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
    h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
    h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
    h4 = linear(tf.reshape(h3, [FLAGS.batch_size, -1]), 1, 'd_h4_logits')
    #_activation_summary(h4, reuse, for_G)
    h4_sigmoid = tf.nn.sigmoid(h4, name='d_h4_sigmoid')
    #_activation_summary(h4_sigmoid, reuse, for_G)

    return h4, h4_sigmoid

def generator(z):
  with tf.variable_scope('generator'):
    s = FLAGS.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    gf_dim = 64
    
    h0 = tf.reshape(linear(z, 256*s16*s16, 'g_h0_lin'), [-1,s16,s16,256])
    h0 = tf.nn.relu(g_bn0(h0))

    h1_1 = deconv2d(h0, [FLAGS.batch_size, s8, s8, 256], k_h=3, k_w=3, name='g_h1_1')
    h1_1 = tf.nn.relu(g_bn1_1(h1_1))
    h1_2 = deconv2d(h1_1, [FLAGS.batch_size, s8, s8, 256], 
                      k_h=3, k_w=3, d_h=1, d_w=1, name='g_h1_2')
    h1_2 = tf.nn.relu(g_bn1_2(h1_2))

    h2_1 = deconv2d(h1_2, [FLAGS.batch_size, s4, s4, 256], k_h=3, k_w=3, name='g_h2_1')
    h2_1 = tf.nn.relu(g_bn2_1(h2_1))
    h2_2 = deconv2d(h2_1, [FLAGS.batch_size, s4, s4, 256], 
                      k_h=3, k_w=3, d_h=1, d_w=1, name='g_h2_2')
    h2_2 = tf.nn.relu(g_bn2_2(h2_2))

    h3_1 = deconv2d(h2_2, [FLAGS.batch_size, s2, s2, 128], k_h=3, k_w=3, name='g_h3_1')
    h3_1 = tf.nn.relu(g_bn3_1(h3_1))

    h4_1 = deconv2d(h3_1, [FLAGS.batch_size, s, s, 64], k_h=3, k_w=3, name='g_h4_1')
    h4_1 = tf.nn.relu(g_bn4_1(h4_1))

    h5 = deconv2d(h4_1, [FLAGS.batch_size, s, s, FLAGS.c_dim], 
                      k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5')

    return tf.nn.tanh(h5)

def sampler(z):
  with tf.variable_scope('generator'):
    tf.get_variable_scope().reuse_variables()
    s = FLAGS.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    gf_dim = 64
    
    h0 = tf.reshape(linear(z, 256*s16*s16, 'g_h0_lin'), [-1,s16,s16,256])
    h0 = tf.nn.relu(g_bn0(h0, train=False))

    h1_1 = deconv2d(h0, [FLAGS.batch_size, s8, s8, 256], k_h=3, k_w=3, name='g_h1_1')
    h1_1 = tf.nn.relu(g_bn1_1(h1_1, train=False))
    h1_2 = deconv2d(h1_1, [FLAGS.batch_size, s8, s8, 256], 
                      k_h=3, k_w=3, d_h=1, d_w=1, name='g_h1_2')
    h1_2 = tf.nn.relu(g_bn1_2(h1_2, train=False))

    h2_1 = deconv2d(h1_2, [FLAGS.batch_size, s4, s4, 256], k_h=3, k_w=3, name='g_h2_1')
    h2_1 = tf.nn.relu(g_bn2_1(h2_1, train=False))
    h2_2 = deconv2d(h2_1, [FLAGS.batch_size, s4, s4, 256], 
                      k_h=3, k_w=3, d_h=1, d_w=1, name='g_h2_2')
    h2_2 = tf.nn.relu(g_bn2_2(h2_2, train=False))

    h3_1 = deconv2d(h2_2, [FLAGS.batch_size, s2, s2, 128], k_h=3, k_w=3, name='g_h3_1')
    h3_1 = tf.nn.relu(g_bn3_1(h3_1, train=False))

    h4_1 = deconv2d(h3_1, [FLAGS.batch_size, s, s, 64], k_h=3, k_w=3, name='g_h4_1')
    h4_1 = tf.nn.relu(g_bn4_1(h4_1, train=False))

    h5 = deconv2d(h4_1, [FLAGS.batch_size, s, s, FLAGS.c_dim], 
                      k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5')

    return tf.nn.tanh(h5)

def inference(image, random_z):

  G_image = generator(random_z)

  D_logits_real, D_sigmoid_real = discriminator(image)

  D_logits_fake, D_sigmoid_fake = discriminator(G_image, True)

  D_logits_fake_for_G, D_sigmoid_fake_for_G = discriminator(G_image, True, True)

  return D_logits_real, D_logits_fake, D_logits_fake_for_G, D_sigmoid_real, D_sigmoid_fake, D_sigmoid_fake_for_G

def loss_l2(D_logits_real, D_logits_fake, D_logits_fake_for_G):
  G_loss = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake_for_G - tf.ones_like(D_logits_fake_for_G))) 

  D_loss_real = tf.reduce_mean(tf.nn.l2_loss(D_logits_real - tf.ones_like(D_logits_real))) 

  D_loss_fake = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake - tf.zeros_like(D_logits_fake))) 

  D_loss = D_loss_real + D_loss_fake


  return G_loss, D_loss


def train(G_loss, D_loss, G_vars, D_vars, global_step):

  G_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 
  D_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) 

  G_grads = G_optim.compute_gradients(G_loss, var_list=G_vars)
  D_grads = D_optim.compute_gradients(D_loss, var_list=D_vars)



  G_train_op = G_optim.apply_gradients(G_grads, global_step=global_step)
  D_train_op = D_optim.apply_gradients(D_grads)

  return G_train_op, D_train_op



