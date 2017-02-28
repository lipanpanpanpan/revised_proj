#TensorFlow implementation of GANs for Sequence Modelling#


import tensorflow as tf
import numpy as np
from tf.contrib import rnn, legacy_seq2seq

import cPickle

class Discriminator():
    def __init__(self,args):
        self.args = args
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("Model type not supported: {}".format(args.model))

        self.cell = cell_fn(args.rnn_size,state_is_tuple = True)

        self.input_data = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        self.target = tf.placeholder(tf.float32,[args.batch_size,1])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.W1 = tf.Variable(tf.random_normal([args.seq_length,args.fc_hidden],stddev = 0.35)
        self.W2 = tf.Variable(tf.random_normal([args.fc_hidden,1],stddev = 0.2)
        self.b1 = tf.Variable(tf.random_normal([args.batch_size,1],stddev = 0.2))
        self.b2 = tf.Variable(tf.random_normal([args.batch_size,1],stddev = 0.2))
        self.lr = tf.Variable(0.0, trainable = False)
        self.emb_matrix = cPickle.load(open('W_twitter_bpe.pkl','rb'))



    def discriminate(self):
        with tf.variable_scope('disc') as scope:
            inputs = tf.split(tf.nn.embedding_lookup(self.emb_matrix, self.input_data), args.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            outputs,state = output,state = rnn.static_rnn(self.cell,inputs,dtype = tf.float32,scope = 'disc')
            output_ = tf.reshape(tf.concat(output[-1], 1), [-1, args.rnn_size])

            self.logit = tf.matmul(tf.nn.relu((tf.matmul(self.W1,output_) + self.b1)),self.W2) + self.b1
            self.prob = tf.nn.sigmoid(logit)
            self.tvars = tf.trainable_variables()
