#GAN TensorFlow implementation for sequence Modelling
import argparse
import os

import numpy as np
import tensorflow as tf

from Discriminator import *
from Generator import *
from utils import TextLoader

import time

t_vars = tf.global_variables()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',type = str, default = './', help = 'data directory for input')
    parser.add_argument('--save_dir',type = str, default = 'save/', help = 'directory to store checkpoint models')
    parser.add_argument('--rnn_size',type = int, default = 128, help = 'size of RNN hidden state')
    parser.add_argument('--model',type = str, default = 'lstm', help = 'Recurrent unit (rnn, gru, lstm)')
    parser.add_argument('--batch_size',type = int, default = 50, help = 'MiniBatch size')
    parser.add_argument('--seq_length',type = int, default = 50, help = 'RNN sequence length')
    parser.add_argument('--dis_seq_length',type = int, default = 100, help = 'RNN discriminator sequence length')
    parser.add_argument('--num_epochs',type = int, default = 50, help = 'number of epochs')
    parser.add_argument('--save_frequency',type = int, default = 1000, help = 'save frequency')
    parser.add_argument('--disc_learning_rate', type=float, default=0.002,help= 'Discriminator learning rate')
    parser.add_argument('--gen_learning_rate', type=float, default=0.002,help= 'Generator learning rate')
    parser.add_argument('--fc_hidden', type=int, default=500,help= 'Num hidden layer nodes for FC discriminator')
    parser.add_argument('--vocab_size', type = int, default = 1000, help = 'Size of vocabulary')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir,args.batch_size,args.seq_length)

    if args.init_from is not None:
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

    Disc = Discriminator(args)
    Gen = Generator(args)
#    D_tvars = [Disc.W1,Disc.W2]
#    G_tvars = [Gen.weight]
    fp1 = open('G_loss_training','w')
    fp2 = open('D_loss_training','w')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
	    print str(e)+'th epoch'
            sess.run(tf.assign(Disc.lr, args.disc_learning_rate))
            sess.run(tf.assign(Gen.lr, args.gen_learning_rate))
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                con,res = data_loader.next_batch()
                real_data = np.concatenate((con,res),axis=1)
                fake_data = sess.run(Fake_data,feed_dict = {Gen.input_data : con})

                D_real, D_logit_real= sess.run([prob,logit], feed_dict = {Disc.input_data : real_data})
                D_fake, D_logit_fake = sess.run([prob,logit], feed_dict = {Disc.input_data : fake_data})

                D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))
                G_loss = -tf.reduce_mean(tf.log(D_fake))

		
		D_tvars = [v for v in t_vars if v.name.startswith('disc')]
		G_tvars = [v for v in t_vars if v.name.startswith('gen')]
		D_solver = tf.train.AdamOptimizer(Disc.lr).minimize(D_loss, var_list = D_tvars)
                G_solver = tf.train.AdamOptimizer(Gen.lr).minimize(G_loss, var_list = G_tvars)

                _, d_loss = sess.run([D_solver, D_loss], feed_dict = {Disc.input_data : real_data, Gen.input_data : con})
                _, g_loss = sess.run([G_solver, G_loss], feed_dict = {Disc.input_data : fake_data, Gen.input_data : con})

                fp1.write(str(g_loss)+'\n')
                fp2.write(str(d_loss)+'\n')
                end = time.time()
                print("{}/{} (epoch {}), Generator_loss = {:.3f}, Discriminator_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, g_loss, d_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
        fp1.close()
        fp2.close()

if __name__ == '__main__':
    main()
