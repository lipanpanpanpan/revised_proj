#GAN TensorFlow implementation for sequence Modelling
import argparse
import os

import numpy as np
import tensorflow as tf

from utils import TextLoader
from cGAN import cGAN
import time


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
    parser.add_argument('--disc_learning_rate', type=float, default=0.02,help= 'Discriminator learning rate')
    parser.add_argument('--gen_learning_rate', type=float, default=0.00001,help= 'Generator learning rate')
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
  #  if args.init_from is not None:
  #      ckpt = tf.train.get_checkpoint_state(args.init_from)
  #      assert ckpt,"No checkpoint found"
  #      assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

    with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
        cGan = cGAN(sess,args)
	cGan.GAN_train()

if __name__ == '__main__':
    main()
