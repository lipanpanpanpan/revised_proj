#TensorFlow implementation of GANs for Sequence Modelling#


import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, legacy_seq2seq

import cPickle

logit = tf.Variable(0.0, dtype=tf.float64)
prob = tf.Variable(0.0, dtype=tf.float64)
Dtvars = []

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

        self.input_data = tf.placeholder(tf.int32,[args.batch_size,args.dis_seq_length])
        self.target = tf.placeholder(tf.float32,[args.batch_size,1])
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float64)
        self.W1 = tf.Variable(tf.random_normal([args.dis_seq_length,args.fc_hidden],stddev = 0.35),name = 'd_W1')
        self.W2 = tf.Variable(tf.random_normal([args.fc_hidden,1],stddev = 0.2),name = 'd_W2')
        self.b1 = tf.Variable(tf.random_normal([args.batch_size,1],stddev = 0.2),name = 'd_b1')
        self.b2 = tf.Variable(tf.random_normal([args.batch_size,1],stddev = 0.2),name = 'd_b2')
        self.lr = tf.Variable(0.0, trainable = False, dtype = tf.float64)
        self.emb_matrix = cPickle.load(open('W_twitter_bpe.pkl','rb'))
#	self.emb_matrix = np.float32(self.emb_matrix)
	self.generated_batch = None
	


    def Discriminate(self):
        with tf.variable_scope('disc') as scope:
            inputs = tf.split(tf.nn.embedding_lookup(self.emb_matrix, self.input_data), self.args.dis_seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            outputs,state = output,state = rnn.static_rnn(self.cell,inputs,dtype = tf.float64,scope = 'disc')
            output_ = tf.reshape(tf.concat(outputs[-1], 1), [-1, self.args.rnn_size])
	    lstm_variables = [v for v in tf.all_variables() if v.name.startswith(scope.name)]
            logit = tf.matmul(tf.nn.relu((tf.matmul(self.W1,output_) + self.b1)),self.W2) + self.b1
            prob = tf.nn.sigmoid(logit)
            tvars = tf.trainable_variables()
	    #Dtvars = [var for v in tvars if 'g_' in v.name]
	    Dtvars = [v for v in tf.all_variables() if v.name.startswith(scope.name)]
            #return prob, logit, tvars
