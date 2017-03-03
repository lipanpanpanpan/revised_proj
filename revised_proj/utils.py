#GAN tensorflow

import cPickle
import os

import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding = 'utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        #input_file = os.path.join(data_dir,'train.txt')
        tensor_file = os.path.join(data_dir,'Dataset_pretrain.pkl')

        self.load_preprocessed(tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def load_preprocessed(self, tensor_file):
        self.tensor, b = cPickle.load(open(tensor_file,'rb'))#np.load(tensor_file)
        self.num_batches = int(self.tensor.size/(self.tensor.shape[0]*self.batch_size * self.seq_length))

    def create_batches(self):
        if self.num_batches == 0:
            assert False, 'Not enough Data! Make seq_length and/or batch_size small'
        #self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        self.data_set = [self.tensor[i] for i in range(len(self.tensors))]
	self.data_set += [b]
	self.data_set = [np.vsplit(self.data_set[i],self.num_batches) for i in range(len(self.data_set))]
#        self.con_batches = np.vsplit(context,self.num_batches)
 #       self.res_batches = np.vsplit(response,self.num_batches)

    def next_batch(self):
        if self.pointer<self.num_batches:
            d_batch = [self.dataset[i][self.pointer] for i in range(len())]#self.con_batches[self.pointer], self.res_batches[self.pointer]
            self.pointer+=1
        return d_batch

    def reset_batch_pointer(self):
        self.pointer = 0
