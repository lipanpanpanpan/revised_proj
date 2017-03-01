#TensorFlow implementation of GANS - Generator

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, legacy_seq2seq

import cPickle

Fake_data = tf.Variable(tf.zeros([186,100],dtype = tf.int32), trainable = False, dtype = tf.int32)
Gtvars = []

class Generator():
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
	self.generated_batch = None

        self.cell_fn = cell_fn(args.rnn_size,state_is_tuple = True)
        self.input_data = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])

        self.initial_state = self.cell_fn.zero_state(args.batch_size, tf.float64)

        self.emb_matrix, _ = cPickle.load(open('W_twitter_bpe.pkl','rb'))
        self.weight = tf.Variable(tf.truncated_normal([args.rnn_size, args.vocab_size], stddev=0.1),dtype =tf.float32,name ='g_weight')
        self.bias = tf.Variable(tf.constant(0.1, shape=[args.vocab_size]),dtype = tf.float32,name = 'g_bias')
        self.lr = tf.Variable(0.0, trainable = False,dtype = tf.float32)

    def Generate(self):
        with tf.variable_scope('gen') as scope:
            inputs = tf.split(tf.nn.embedding_lookup(self.emb_matrix, self.input_data), self.args.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            outputs,_ = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell_fn, scope = 'rnn_dec')
	    

            output_ = tf.reshape(tf.concat(outputs, 1), [-1, self.args.rnn_size])
            logits = tf.matmul(tf.cast(output_,tf.float32), self.weight) + self.bias
	    
            probs = tf.nn.softmax(logits)
            pred = tf.multinomial(probs,1)
	    
            prediction = tf.reshape(pred,[self.args.batch_size,self.args.seq_length])

            fake_data = tf.concat(self.input_data,prediction,1)
	    tvars = tf.trainable_variables()
	    Gtvars = [v for v in tf.all_variables() if v.name.startswith(scope.name)]
	    print Gtvars
	    #Gtvars += [decoder_variables]
