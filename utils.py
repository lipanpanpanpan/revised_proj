#GAN tensorflow

import cPickle
import os

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding = 'utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        #input_file = os.path.join(data_dir,'train.txt')
        tensor_file = os.path.join(data_dir,'Dataset_twitter.npy')

        self.load_preprocessed(tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def load_preprocessed(self, tensor_file):
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size/(2*self.batch_size * self.seq_length))

    def create_batches(self):
        if num_batches == 0:
            assert False, 'Not enough Data! Make seq_length and/or batch_size small'
        #self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        context = self.tensor[0]
        response = self.tensor[1]
        self.con_batches = np.vsplit(context,self.num_batches)
        self.res_batches = np.vsplit(response,self.num_batches)

    def next_batch(self):
        if self.pointer<self.num_batches:
            c_batch,r_batch = self.con_batches[self.pointer], self.res_batches[self.pointer]
            self.pointer+=1
        return c_batch, r_batch

    def reset_batch_pointer(self):
        self.pointer = 0
