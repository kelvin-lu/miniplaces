import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
import resnet as rn

batch_size = 32
load_size = 256
fine_size = 224
resnet_size = 50
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 50
step_save = 10000
path_save = 'resnet'
start_from = 'D:/Stuff/Classes/6.819/repo/miniplaces/model/tensorflow/resnet_50.txt'

# Construct dataloader
opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../../images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_test = DataLoaderDisk(**opt_data_test)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
model = rn.imagenet_resnet_v2(resnet_size, 100, data_format=None)
logits = model(x, True)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
top5_pred = tf.nn.top_k(logits, 5)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

def prepare_submission(results, weights_path=None):
    """
    :param results: softmax output on test set
    """
    # fetch top 5 indices
    #ind=np.argsort(results,axis=1)[:,-5:][:,::-1]

    # now write the submission file
    if weights_path:
        filename = 'submit-{}.txt'.format(weights_path)
    else:
        filename = 'submit.txt'
    with open(filename, 'w+') as f:
        f.write('')

    with open(filename, 'a') as f:
        for x in range(10000):
            path = 'test/' + str(x+1).zfill(8)[-8:] + '.jpg'
            labels = str(results[x])[1:-1] # cut off [] lol
            f.write(path + ' ' + labels + '\n')

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
        
    num_batch = loader_test.size()//batch_size
    
    results = []

    for i in range(num_batch):
        # Load a batch of training data
        images_batch, extra = loader_test.next_batch(batch_size)

        probs, batch_labels = sess.run([top5_pred], feed_dict = {x: images_batch, keep_dropout: 1., train_phase: False})
        
        for top5_labels in batch_labels:
            results.append(top5_labels)
            
    prepare_submission(results)
        
        


