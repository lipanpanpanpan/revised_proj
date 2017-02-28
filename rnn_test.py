import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, legacy_seq2seq

x = tf.placeholder(tf.int64,[4,4])
y = tf.placeholder(tf.float32,[4,1])

cell = rnn.BasicLSTMCell(200,state_is_tuple = True)#,scope = 'myrnn')

#inputs = tf.split(x,4,1)

#inputs = [tf.squeeze(input_, [1]) for input_ in inputs]


with tf.variable_scope("mrnn") as scope:
     softmax_w = tf.Variable(tf.random_normal([200, 4], stddev=0.1))
     softmax_b = tf.Variable(tf.constant(0.1, shape =[4]))
     matrix = np.matrix(np.random.random([4, 64]), dtype = np.float32)
     #for i in range(4):
    #     if i>0:
    #        scope.reuse_variables()
     inputs = tf.split(tf.nn.embedding_lookup(matrix, x), 4, 1)
     inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
     #output,state = legacy_seq2seq.rnn_decoder(inputs,cell.zero_state(4,tf.float32), cell,scope="mrnn")
     output,state = rnn.static_rnn(cell,inputs,dtype = tf.float32,scope = 'myrnn')
     output_ = tf.reshape(tf.concat(output, 1), [-1, 200])
     logits = tf.matmul(output_, softmax_w) + softmax_b
     pred = tf.nn.sigmoid(logits)
     cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits))
     optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.5).minimize(cost)
     #probs = tf.nn.softmax(logits)
     #pred = tf.multinomial(probs,1)
     #pred = tf.reshape(pred,[4,4])
     #conc = tf.concat([x,pred],1)


if __name__ == '__side__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred = sess.run([pred],feed_dict = {x:np.array([[0,2,3,0],[1,2,0,0],[2,1,0,3],[3,1,0,2]])})
        pred = tf.reshape(pred[0],[4,4])
        conc = tf.concat([x,pred],1)
        conc = sess.run([conc],feed_dict = {x:np.array([[0,2,3,0],[1,2,0,0],[2,1,0,3],[3,1,0,2]])})
        print conc

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step =1
        while step < 20000:
            sess.run(optimizer, feed_dict = {x:np.array([[0,2,3,0],[1,2,0,0],[2,1,0,3],[3,1,0,2]]), y: np.array([[0],[1],[0],[1]])})#,[[0,0,1,0],[0,1,0,0],[0,1,0,0],[1,0,0,0]]])})
            if step % 5 == 0:
                loss = sess.run(cost, feed_dict = {x:np.array([[0,2,3,0],[1,2,0,0],[2,1,0,3],[3,1,0,2]]), y: np.array([[0],[1],[0],[1]])})
                print loss
            step+=1
        print "Optimization done!!"
        pred = sess.run([pred],feed_dict = {x:np.array([[0,2,3,0],[1,2,0,0],[2,1,0,3],[3,1,0,2]])})
        print pred
