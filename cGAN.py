#TensorFlow implementation of GANS - Generator

import argparse
import os

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, legacy_seq2seq

from utils import TextLoader

import time


import cPickle


class cGAN():
    def __init__(self,sess,args):
        self.args = args
	self.sess = sess
	self.data_loader = TextLoader(args.data_dir,args.batch_size,args.seq_length)
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
        
	self.cell = cell_fn(args.rnn_size,state_is_tuple = True)

        self.initial_state = self.cell_fn.zero_state(args.batch_size, tf.float64)

        self.emb_matrix = np.load('Emb_mat.npy')
        self.weight = tf.Variable(tf.truncated_normal([args.rnn_size, args.vocab_size], stddev=0.1),dtype =tf.float32,name ='g_weight')
        self.bias = tf.Variable(tf.constant(0.1, shape=[args.vocab_size]),dtype = tf.float32,name = 'g_bias',trainable = False)
        self.gen_lr = args.gen_learning_rate

        
        #self.target = tf.placeholder(tf.float32,[args.batch_size,1])
        self.W1 = tf.Variable(tf.random_normal([args.dis_seq_length,args.fc_hidden],stddev = 0.35,dtype = tf.float64),name = 'd_W1')
        self.W2 = tf.Variable(tf.random_normal([args.fc_hidden,1],stddev = 0.2,dtype = tf.float64),name = 'd_W2')
        self.b1 = tf.Variable(tf.random_normal([args.batch_size,1],stddev = 0.2,dtype = tf.float64),name = 'd_b1',trainable = False)
        self.b2 = tf.Variable(tf.random_normal([args.batch_size,1],stddev = 0.2,dtype = tf.float64),name = 'd_b2',trainable = False)
        self.dis_lr = args.disc_learning_rate
	#self.D_Loss, self.G_Loss = self.GAN_train()
	self.build_model()


    def build_model(self):
	self.context = tf.placeholder(tf.int32,[self.args.batch_size,self.args.seq_length])
	self.response = tf.placeholder(tf.int32,[self.args.batch_size,self.args.seq_length])
	
	get_sample = tf.make_template('gen', self.Generate)
	self.fake_data = get_sample()
	get_disc = tf.make_template('disc', self.Discriminate)
	self.D_fake, self.D_logit_fake = get_disc('fake')
	self.D_real, self.D_logit_real= get_disc('real')

	self.D_loss = -tf.reduce_sum(tf.log(self.D_real) + tf.log(1 - self.D_fake))
	self.G_loss = -tf.reduce_sum(tf.log(self.D_fake))
	
	self.Dtvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')]
	self.Gtvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')]
	
	print self.Dtvars
	

    def Generate(self):

            inputs = tf.split(tf.nn.embedding_lookup(self.emb_matrix, self.context), self.args.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            outputs,_ = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell_fn)
	    

            output_ = tf.reshape(tf.concat(outputs, 1), [-1, self.args.rnn_size])
            logits = tf.matmul(tf.cast(output_,tf.float32), self.weight) + self.bias
	    
	    logits = tf.reshape(logits, [self.args.batch_size,self.args.seq_length,self.args.vocab_size])
            #probs = tf.nn.softmax(logits)
            #pred = tf.argmax(probs,1)
	    
            #sample_res = tf.reshape(pred,[self.args.batch_size,self.args.seq_length])

            #fake_data = tf.concat(self.context,prediction,1)
    	
            #tvars = tf.trainable_variables()
            #self.Gtvars = [v for v in tvars if v.name.startswith(scope.name)]
     
            return tf.cast(logits,dtype =tf.float64)
	    #print Gtvars
	    #Gtvars += [decoder_variables]

    def Discriminate(self, flag):
	if flag == 'real':
		inputs = tf.split(tf.nn.embedding_lookup(self.emb_matrix, tf.concat([self.context,self.response],1)), self.args.dis_seq_length, 1)
		inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
	elif flag == 'fake':
		inp = tf.split(tf.nn.embedding_lookup(self.emb_matrix, self.context), self.args.seq_length, 1)
		inp = [tf.squeeze(input_, [1]) for input_ in inp]
		inp = tf.transpose(inp,[1,0,2])
		inputs = tf.split(tf.concat([inp, self.fake_data],1),self.args.dis_seq_length,1)
		inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs,state = output,state = rnn.static_rnn(self.cell,inputs,dtype = tf.float64)
        output_ = tf.reshape(tf.concat(outputs[-1], 1), [-1, self.args.rnn_size])

        logit = tf.matmul(tf.nn.tanh((tf.matmul(output_,self.W1) + self.b1)),self.W2) + self.b1
        prob = tf.nn.sigmoid(logit)

#        self.Dtvars = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
        return prob, logit

    def GAN_train(self):
	    
	    con,res = self.data_loader.next_batch()

            D_solver = tf.train.AdamOptimizer(self.dis_lr).minimize(self.D_loss, var_list = self.Dtvars)
            G_solver = tf.train.AdamOptimizer(self.gen_lr).minimize(self.G_loss, var_list = self.Gtvars)

	    tf.global_variables_initializer().run()
	    
	    for e in range(self.args.num_epochs):
		    start = time.time()
		    print str(e)+'th epoch'
		    _, g_loss = self.sess.run([G_solver, self.G_loss],feed_dict={self.context: con, self.response:res})
		    _, d_loss = self.sess.run([D_solver, self.D_loss],feed_dict={self.context: con, self.response:res})
		    con,res = self.data_loader.next_batch()
		    end = time.time()

		    print("{}/{} (epoch {}), Generator_loss = {:.3f}, Discriminator_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            self.args.num_epochs * data_loader.num_batches,
                            e, g_loss, d_loss, end - start))
		    
		    if (e * self.data_loader.num_batches + b) % self.args.save_every == 0\
                    or (e==self.args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
		            checkpoint_path = os.path.join(self.args.save_dir, 'model.ckpt')
		            saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
		            print("model saved to {}".format(checkpoint_path))
	    
	
