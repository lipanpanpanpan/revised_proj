#TensorFlow implementation of GANS - Generator

import argparse
import os

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, legacy_seq2seq, seq2seq

from utils import TextLoader

import time


import cPickle


class cGAN():
    def __init__(self,sess,args,flag):
        self.args = args
<<<<<<< HEAD
	    self.flag = flag
	    self.sess = sess
	    self.data_loader = TextLoader(args.data_dir,args.batch_size,args.seq_length)
	    self.g = tf.Graph()
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

    	self.context = tf.placeholder(tf.int32,[self.args.batch_size,self.args.seq_length])
    	self.response = tf.placeholder(tf.int32,[self.args.batch_size,self.args.seq_length])
    	self.target = tf.placeholder(tf.float32,[self.args.batch_size])
        self.initial_state = self.cell_fn.zero_state(args.batch_size, tf.int32)
	    self.dec_inp = tf.zeros([self.args.batch_size,self.args.seq_length], dtype=np.int32, name="GO")
=======
		self.flag = flag
		self.sess = sess
		self.data_loader = TextLoader(args.data_dir,args.batch_size,args.seq_length)
		self.g = tf.Graph()
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

		self.context = tf.placeholder(tf.int32,[self.args.batch_size,self.args.seq_length])
		self.response = tf.placeholder(tf.int32,[self.args.batch_size,self.args.seq_length])
		self.target = tf.placeholder(tf.float32,[self.args.batch_size])
        self.initial_state = self.cell_fn.zero_state(args.batch_size, tf.int32)
		self.dec_inp = tf.zeros([self.args.batch_size,self.args.seq_length], dtype=np.int32, name="GO")
>>>>>>> 1ccbac34d142a97896233d0f4552158bce6ce0a5
        self.emb_matrix = np.load('Emb_mat.npy')

		#self.D_Loss, self.G_Loss = self.GAN_train()
	self.build_model()


    def build_model(self):

    	get_sample = tf.make_template('gen', self.Generate)

    	get_disc = tf.make_template('disc', self.Discriminate)

        self.weight = tf.get_variable('weight_gen',initializer = tf.truncated_normal([args.rnn_size, args.vocab_size], stddev=0.1),dtype =tf.float32)
        self.bias = tf.get_variable('bias_gen',initializer = tf.constant(0.1, shape=[args.vocab_size]),dtype = tf.float32,trainable = False)
        self.gen_lr = self.args.gen_learning_rate


        #self.target = tf.placeholder(tf.float32,[args.batch_size,1])
        self.W1 = tf.get_variable(initializer = tf.random_normal([args.dis_seq_length,args.fc_hidden],stddev = 0.35,dtype = tf.float64),name = 'disc_W1')
        self.W2 = tf.Variable(initializer = tf.random_normal([args.fc_hidden,1],stddev = 0.2,dtype = tf.float64),name = 'disc_W2')
        self.b1 = tf.Variable(initializer = tf.random_normal([args.batch_size,1],stddev = 0.2,dtype = tf.float64),name = 'disc_b1',trainable = False)
        self.b2 = tf.Variable(initializer = tf.random_normal([args.batch_size,1],stddev = 0.2,dtype = tf.float64),name = 'disc_b2',trainable = False)
<<<<<<< HEAD
        self.dis_lr = self.args.disc_learning_rate
=======
        self.dis_lr = args.disc_learning_rate
		#self.D_Loss, self.G_Loss = self.GAN_train()
		self.build_model()
>>>>>>> 1ccbac34d142a97896233d0f4552158bce6ce0a5

    	if self.flag == 'GAN':
    		self.fake_data = get_sample()
    		#self.loop = tf.make_template('gen', self.loop)
    		self.D_fake, self.D_logit_fake = get_disc('fake')
    		self.D_real, self.D_logit_real= get_disc('real')

    		self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1 - self.D_fake))
    		self.G_loss = -tf.reduce_mean(tf.log(1 - self.D_fake))
    		self.Dtvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')]
    		self.Gtvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')]

    	else:
    		logits = get_sample(self.flag)
    		weights = tf.concat([tf.ones([self.args.batch_size, 16]), tf.zeros([self.args.batch_size,34])])
    		self.G_Loss = seq2seq.sequence_loss(logits, self.response,weights, average_across_timesteps = True, average_across_batch = True, name = 'Sequence_loss')
    		pred,_ = get_disc(flag)
    		self.D_Loss = -tf.reduce_mean(self.target*tf.log(pred) + (1.-self.target)*tf.log(1. - pred) )


<<<<<<< HEAD


    def Generate(self):
	    with tf.variable_scope("gen", reuse=True):
            self.weight = tf.get_variable('weight_gen',initializer = tf.truncated_normal([args.rnn_size, args.vocab_size], stddev=0.1),dtype =tf.float32)
            self.bias = tf.get_variable('bias_gen',initializer = tf.constant(0.1, shape=[args.vocab_size]),dtype = tf.float32,trainable = False)
        	def loop(prev, _):
        		prev = tf.matmul(tf.cast(prev,dtype= tf.float32), self.weight) + self.bias
        		prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        		return tf.nn.embedding_lookup(self.emb_matrix, prev_symbol)

                #inputs = tf.split(tf.cast(tf.nn.embedding_lookup(self.emb_matrix, self.context),dtype=tf.float32), self.args.seq_length, 1)
    	    #inputs = tf.split(self.context, self.args.seq_length, 1)
                #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
    	    #inputs = tf.transpose(inputs,[1,0,2])
    	    #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
    #	    print len(inputs), inputs[0].shape

    #            outputs,_ = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell_fn)#, loop_function = loop)

    	    print self.context.shape
    	    inputs = tf.split(self.context, self.args.seq_length, 1)
    	    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
    	    print self.dec_inp[0].shape
    	    dec_inp = tf.split(self.dec_inp, self.args.seq_length, 1)
    	    dec_inp = [tf.squeeze(input_, [1]) for input_ in dec_inp]
    	    outputs , state = legacy_seq2seq.embedding_rnn_seq2seq(inputs, dec_inp, self.cell_fn,self.args.vocab_size,self.args.rnn_size, self.args.seq_length,feed_previous = True,output_projection = None)
    #	    outputs ,_ = legacy_seq2seq.embedding_rnn_seq2seq(self.context,)

    	    print len(outputs)
    	    print outputs[0].shape
    	    output_ = tf.reshape(tf.concat(outputs, 1), [-1, self.args.rnn_size])
    	    print output_.shape
            logits = tf.matmul(tf.cast(output_,tf.float32), self.weight) + self.bias
    	    print logits.shape

    	    logits = tf.reshape(logits, [self.args.batch_size,self.args.seq_length,self.args.vocab_size])
            probs = tf.nn.softmax(logits)
                #pred = tf.argmax(probs,1)

                #sample_res = tf.reshape(pred,[self.args.batch_size,self.args.seq_length])

                #fake_data = tf.concat(self.context,prediction,1)

                #tvars = tf.trainable_variables()
                #self.Gtvars = [v for v in tvars if v.name.startswith(scope.name)]
    	    if self.flag != 'pretrain':
    	        return tf.cast(probs,dtype =tf.float64)
            else:
    	        return tf.cast(logits,dtype = tf.float64)
    	    #print Gtvars
    	    #Gtvars += [decoder_variables]

    def Discriminate(self, flag):
        with tf.variable_scope("disc", reuse=True):
            self.W1 = tf.get_variable(initializer = tf.random_normal([args.dis_seq_length,args.fc_hidden],stddev = 0.35,dtype = tf.float64),name = 'disc_W1')
            self.W2 = tf.Variable(initializer = tf.random_normal([args.fc_hidden,1],stddev = 0.2,dtype = tf.float64),name = 'disc_W2')
            self.b1 = tf.Variable(initializer = tf.random_normal([args.batch_size,1],stddev = 0.2,dtype = tf.float64),name = 'disc_b1',trainable = False)
            self.b2 = tf.Variable(initializer = tf.random_normal([args.batch_size,1],stddev = 0.2,dtype = tf.float64),name = 'disc_b2',trainable = False)
        	if flag == 'real' or flag == 'pretrain':
        		inputs = tf.split(tf.nn.embedding_lookup(self.emb_matrix, tf.concat([self.context,self.response],1)), self.args.dis_seq_length, 1)
        		inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        	elif flag == 'fake':
        		inp = tf.split(tf.nn.embedding_lookup(self.emb_matrix, self.context), self.args.seq_length, 1)
        		inp = [tf.squeeze(input_, [1]) for input_ in inp]
        		inp = tf.transpose(inp,[1,0,2])
        		inputs = tf.split(tf.concat([inp, self.fake_data],1),self.args.dis_seq_length,1)
        		inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        	outputs,state = rnn.static_rnn(self.cell,inputs,dtype = tf.float64)
        	output_ = tf.reshape(tf.concat(outputs[-1], 1), [-1, self.args.rnn_size])

        	logit = tf.matmul(tf.nn.tanh((tf.matmul(output_,self.W1) + self.b1)),self.W2) + self.b1
        	prob = tf.nn.sigmoid(logit)

        #        self.Dtvars = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
        	return prob, logit

    def GAN_train(self):
    	data = self.data_loader.next_batch()
        D_solver = tf.train.AdamOptimizer(self.dis_lr).minimize(self.D_loss, var_list = self.Dtvars)
	    G_solver = tf.train.AdamOptimizer(self.gen_lr).minimize(self.G_loss, var_list = self.Gtvars)
        D_solver_pre = tf.train.AdamOptimizer(self.dis_lr).minimize(self.D_loss_CE, var_list = self.Dtvars)
        G_solver_pre = tf.train.AdamOptimizer(self.gen_lr).minimize(self.G_loss_SL, var_list = self.Gtvars)

    	tf.global_variables_initializer().run()
    	saver = tf.train.Saver(tf.global_variables())
    	for e in range(self.args.num_epochs):
    		self.data_loader.reset_batch_pointer()
    		for b in range(10):#self.data_loader.num_batches):
    		    if self.flag == 'GAN':
        			start = time.time()
        			_, g_loss = self.sess.run([G_solver, self.G_loss],feed_dict={self.context: data[0], self.response:data[1]})
        		        _, d_loss = self.sess.run([D_solver, self.D_loss],feed_dict={self.context: data[0], self.response:data[1]})
        			data= self.data_loader.next_batch()
        			end = time.time()
                    print("{}/{} (epoch {}), Generator_loss = {:.3f}, Discriminator_loss = {:.3f}, time/batch = {:.3f}" \
    			    .format(e * self.data_loader.num_batches + b,
    					    self.args.num_epochs * self.data_loader.num_batches,
    					    e, g_loss, d_loss, end - start))

        			if ((e+1) * self.data_loader.num_batches + b) % self.args.save_frequency == 0\
        				    or (e==self.args.num_epochs-1 and b == self.data_loader.num_batches-1): # save for the last result
        			    checkpoint_path = os.path.join(self.args.save_dir, 'model.ckpt')
        			    saver.save(self.sess, checkpoint_path, global_step = e * self.data_loader.num_batches + b)
        			    print("model saved to {}".format(checkpoint_path))

    		    elif self.flag == 'pretrain':
    		      	start = time.time()
    			    _, g_loss = self.sess.run([G_solver_pre, self.G_loss_SL],feed_dict={self.context: data[0], self.response:data[1]})
    		        _, d_loss = self.sess.run([D_solver_pre, self.D_loss_CE],feed_dict={self.context: data[0], self.response:data[1], self.target: data[2]})
    	 		    data = self.data_loader.next_batch()
    			    end = time.time()

        			print("{}/{} (epoch {}), Generator_loss = {:.3f}, Discriminator_loss = {:.3f}, time/batch = {:.3f}" \
        				    .format(e * self.data_loader.num_batches + b,
        					    self.args.num_epochs * self.data_loader.num_batches,
        					    e, g_loss, d_loss, end - start))

        			if ((e+1) * self.data_loader.num_batches + b) % self.args.save_frequency == 0\
        				    or (e==self.args.num_epochs-1 and b == self.data_loader.num_batches-1): # save for the last result
        			    checkpoint_path = os.path.join(self.args.save_dir, 'model.ckpt')
        			    saver.save(self.sess, checkpoint_path, global_step = e * self.data_loader.num_batches + b)
        			    print("model saved to {}".format(checkpoint_path))
=======
		get_sample = tf.make_template('gen', self.Generate)
		
		get_disc = tf.make_template('disc', self.Discriminate)
		
		if self.flag == 'GAN':
			self.fake_data = get_sample()
			#self.loop = tf.make_template('gen', self.loop)
			self.D_fake, self.D_logit_fake = get_disc('fake')
			self.D_real, self.D_logit_real= get_disc('real')

			self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1 - self.D_fake))
			self.G_loss = -tf.reduce_mean(tf.log(1 - self.D_fake))

			self.Dtvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')]
			self.Gtvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')]

		else:
			logits = get_sample(self.flag)
			weights = tf.concat([tf.ones([self.args.batch_size, 16]), tf.zeros([self.args.batch_size,34]]))
			self.G_Loss = seq2seq.sequence_loss(logits, self.response,weights, average_across_timesteps = True, average_across_batch = True, name = 'Sequence_loss')
			pred,_ = get_disc(flag)
			self.D_Loss = -tf.reduce_mean(self.target*tf.log(pred) + (1.-self.target)*tf.log(1. - pred) )
			
						


    def Generate(self):
	    
	    def loop(prev, _):
	    	prev = tf.matmul(tf.cast(prev,dtype= tf.float32), self.weight) + self.bias
			prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
			return tf.nn.embedding_lookup(self.emb_matrix, prev_symbol)
	     
            #inputs = tf.split(tf.cast(tf.nn.embedding_lookup(self.emb_matrix, self.context),dtype=tf.float32), self.args.seq_length, 1)
	    #inputs = tf.split(self.context, self.args.seq_length, 1)
            #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
	    #inputs = tf.transpose(inputs,[1,0,2])
	    #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
#	    print len(inputs), inputs[0].shape	
    
#            outputs,_ = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell_fn)#, loop_function = loop)
	    
	    print self.context.shape
	    inputs = tf.split(self.context, self.args.seq_length, 1)
	    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
	    print self.dec_inp[0].shape
	    dec_inp = tf.split(self.dec_inp, self.args.seq_length, 1)
	    dec_inp = [tf.squeeze(input_, [1]) for input_ in dec_inp]
	    outputs , state = legacy_seq2seq.embedding_rnn_seq2seq(inputs, dec_inp, self.cell_fn,self.args.vocab_size,self.args.rnn_size, self.args.seq_length,feed_previous = True,output_projection = None)
#	    outputs ,_ = legacy_seq2seq.embedding_rnn_seq2seq(self.context,)

	    print len(outputs)
	    print outputs[0].shape
        output_ = tf.reshape(tf.concat(outputs, 1), [-1, self.args.rnn_size])
	    print output_.shape
        logits = tf.matmul(tf.cast(output_,tf.float32), self.weight) + self.bias
	    print logits.shape
	    
	    logits = tf.reshape(logits, [self.args.batch_size,self.args.seq_length,self.args.vocab_size])
        probs = tf.nn.softmax(logits)
            #pred = tf.argmax(probs,1)
	    
            #sample_res = tf.reshape(pred,[self.args.batch_size,self.args.seq_length])

            #fake_data = tf.concat(self.context,prediction,1)
    	
            #tvars = tf.trainable_variables()
            #self.Gtvars = [v for v in tvars if v.name.startswith(scope.name)]
     	if self.flag != 'pretrain':
	        return tf.cast(probs,dtype =tf.float64)
		else:
			return tf.cast(logits,dtype = tf.float64)
	    #print Gtvars
	    #Gtvars += [decoder_variables]

    def Discriminate(self, flag):
		if flag == 'real' or flag == 'fake':
			inputs = tf.split(tf.nn.embedding_lookup(self.emb_matrix, tf.concat([self.context,self.response],1)), self.args.dis_seq_length, 1)
			inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
		elif flag == 'fake':
			inp = tf.split(tf.nn.embedding_lookup(self.emb_matrix, self.context), self.args.seq_length, 1)
			inp = [tf.squeeze(input_, [1]) for input_ in inp]
			inp = tf.transpose(inp,[1,0,2])
			inputs = tf.split(tf.concat([inp, self.fake_data],1),self.args.dis_seq_length,1)
			inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

	    outputs,state = rnn.static_rnn(self.cell,inputs,dtype = tf.float64)
	    output_ = tf.reshape(tf.concat(outputs[-1], 1), [-1, self.args.rnn_size])

	    logit = tf.matmul(tf.nn.tanh((tf.matmul(output_,self.W1) + self.b1)),self.W2) + self.b1
	    prob = tf.nn.sigmoid(logit)

#        self.Dtvars = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
	    return prob, logit

    def GAN_train(self):

	    con,res = self.data_loader.next_batch()
	    flag = 'pretrain'
       
	    D_solver = tf.train.AdamOptimizer(self.dis_lr).minimize(self.D_loss, var_list = self.Dtvars)
		G_solver = tf.train.AdamOptimizer(self.gen_lr).minimize(self.G_loss, var_list = self.Gtvars)
	    D_solver_pre = tf.train.AdamOptimizer(self.dis_lr).minimize(self.D_loss_CE, var_list = self.Dtvars)
		G_solver_pre = tf.train.AdamOptimizer(self.gen_lr).minimize(self.G_loss_SL, var_list = self.Gtvars)

	    tf.global_variables_initializer().run()
	    saver = tf.train.Saver(tf.global_variables())
	    for e in range(self.args.num_epochs):
		    self.data_loader.reset_batch_pointer()
		    for b in range(10):#self.data_loader.num_batches):
			    if self.flag == 'GAN':
					start = time.time()
					_, g_loss = self.sess.run([G_solver, self.G_loss],feed_dict={self.context: con, self.response:res})
				    _, d_loss = self.sess.run([D_solver, self.D_loss],feed_dict={self.context: con, self.response:res})
				    con,res = self.data_loader.next_batch()
				    end = time.time()

				    print("{}/{} (epoch {}), Generator_loss = {:.3f}, Discriminator_loss = {:.3f}, time/batch = {:.3f}" \
				    .format(e * self.data_loader.num_batches + b,
					    self.args.num_epochs * self.data_loader.num_batches,
					    e, g_loss, d_loss, end - start))
				    
				    if ((e+1) * self.data_loader.num_batches + b) % self.args.save_frequency == 0\
				    or (e==self.args.num_epochs-1 and b == self.data_loader.num_batches-1): # save for the last result
					    checkpoint_path = os.path.join(self.args.save_dir, 'model.ckpt')
					    saver.save(self.sess, checkpoint_path, global_step = e * self.data_loader.num_batches + b)
					    print("model saved to {}".format(checkpoint_path))
		    
			    elif self.flag == 'pretrain':
					start = time.time()
					_, g_loss = self.sess.run([G_solver_pre, self.G_loss_SL],feed_dict={self.context: con, self.response:res})
				    _, d_loss = self.sess.run([D_solver_pre, self.D_loss_CE],feed_dict={self.context: con, self.response:res. self.target: y})
				    con,res = self.data_loader.next_batch()
				    end = time.time()

				    print("{}/{} (epoch {}), Generator_loss = {:.3f}, Discriminator_loss = {:.3f}, time/batch = {:.3f}" \
				    .format(e * self.data_loader.num_batches + b,
					    self.args.num_epochs * self.data_loader.num_batches,
					    e, g_loss, d_loss, end - start))
				    
				    if ((e+1) * self.data_loader.num_batches + b) % self.args.save_frequency == 0\
				    or (e==self.args.num_epochs-1 and b == self.data_loader.num_batches-1): # save for the last result
					    checkpoint_path = os.path.join(self.args.save_dir, 'model.ckpt')
					    saver.save(self.sess, checkpoint_path, global_step = e * self.data_loader.num_batches + b)
					    print("model saved to {}".format(checkpoint_path))
>>>>>>> 1ccbac34d142a97896233d0f4552158bce6ce0a5
