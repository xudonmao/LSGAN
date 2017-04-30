from collections import OrderedDict
import pylab
import tensorflow as tf
import numpy as np
ds = tf.contrib.distributions
slim = tf.contrib.slim
graph_replace = tf.contrib.graph_editor.graph_replace

from keras.optimizers import Adam

try:
    from moviepy.video.io.bindings import mplfig_to_npimage
    import moviepy.editor as mpy
    generate_movie = True
except:
    print("Warning: moviepy not found.")
    generate_movie = False



def sample_mog(batch_size, n_mixture=9, std=0.01, radius=1.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample_n(batch_size)


def generator(z, output_dim=2, n_hidden=128, n_layer=2):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.tanh)
        x = slim.fully_connected(h, output_dim, activation_fn=None)
    return x

def discriminator(x, n_hidden=128, n_layer=2, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h = slim.stack(x, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.tanh)
        log_d = slim.fully_connected(h, 1, activation_fn=None)
    return log_d



params = dict(
    batch_size=512,
    disc_learning_rate=1e-4,
    gen_learning_rate=1e-3,
    beta1=0.5,
    epsilon=1e-8,
    max_iter=35000,
    viz_every=5000,
    z_dim=256,
    x_dim=2,
    unrolling_steps=0,
)


tf.reset_default_graph()

data = sample_mog(params['batch_size'])

noise = ds.Normal(tf.zeros(params['z_dim']), 
                  tf.ones(params['z_dim'])).sample_n(params['batch_size'])

with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
    samples = generator(noise, output_dim=params['x_dim'])
    real_score = discriminator(data)
    fake_score = discriminator(samples, reuse=True)
    
D_loss_real = 0.5*tf.reduce_mean(tf.nn.l2_loss(real_score - tf.ones_like(real_score))) 
D_loss_fake = 0.5*tf.reduce_mean(tf.nn.l2_loss(fake_score - tf.zeros_like(fake_score))) 
D_loss = D_loss_real + D_loss_fake

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

d_opt = Adam(lr=params['disc_learning_rate'], beta_1=params['beta1'], epsilon=params['epsilon'])
updates = d_opt.get_updates(disc_vars, [], D_loss)
d_train_op = tf.group(*updates, name="d_train_op")

G_loss = 0.5*tf.reduce_mean(tf.nn.l2_loss(fake_score - tf.ones_like(fake_score))) 

g_train_opt = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])
g_train_op = g_train_opt.minimize(G_loss, var_list=gen_vars)



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



from tqdm import tqdm
xmax = 3
fs = []
frames = []
np_samples = []
n_batches_viz = 10
viz_every = params['viz_every']
for i in tqdm(xrange(params['max_iter'])):
    f, _, _ = sess.run([[G_loss, D_loss], g_train_op, d_train_op])
    fs.append(f)
    if i % viz_every == 0:
        np_samples.append(np.vstack([sess.run(samples) for _ in xrange(n_batches_viz)]))
        xx, yy = sess.run([samples, data])


import seaborn as sns


data_t = sample_mog(params['batch_size'])
data = sess.run(data_t)
fig = pylab.gcf()
fig.set_size_inches(16.0, 16.0)
pylab.clf()
pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color='blue')
pylab.xlim(-4, 4)
pylab.ylim(-4, 4)
pylab.savefig("gt_scatter.png")


color="Greens"
fig = pylab.gcf()
fig.set_size_inches(16.0, 16.0)
pylab.clf()
bg_color  = sns.color_palette(color, n_colors=256)[0]
ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
ax.set_axis_bgcolor(bg_color)
kde = ax.get_figure()
pylab.xlim(-4, 4)
pylab.ylim(-4, 4)
kde.savefig("gt_kde.png")

np_samples_ = np_samples[::1]
for i, data in enumerate(np_samples_):
  fig = pylab.gcf()
  fig.set_size_inches(16.0, 16.0)
  pylab.clf()
  pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color='blue')
  pylab.xlim(-4, 4)
  pylab.ylim(-4, 4)
  pylab.savefig("{}/{}_scatter.png".format(".", i))


  color="Greens"
  fig = pylab.gcf()
  fig.set_size_inches(16.0, 16.0)
  pylab.clf()
  bg_color  = sns.color_palette(color, n_colors=256)[0]
  ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
  ax.set_axis_bgcolor(bg_color)
  kde = ax.get_figure()
  pylab.xlim(-4, 4)
  pylab.ylim(-4, 4)
  kde.savefig("{}/{}_kde.png".format('.', i))


