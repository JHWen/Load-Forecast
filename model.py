from data import get_logger
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
from data import get_train_data, get_test_data
from data import load_data
import matplotlib.pyplot as plt


class LSTM_Model(object):
    def __init__(self, batch_size=100, epoch=50, hidden_dim=100, time_step=20, input_size=6, output_size=1,
                 lr_rate=0.0006, dropout=0.8, lstm_layers_num=2, shuffle=True):
        self.batch_size = batch_size
        self.epoch_num = epoch
        self.hidden_dim = hidden_dim
        self.time_step = time_step
        self.input_size = input_size
        self.output_size = output_size
        self.lr_rate = lr_rate
        self.dropout = dropout
        self.lstm_layers_num = lstm_layers_num
        self.shuffle = shuffle
        self.model_path = './model_path'
        self.logger = get_logger('./log_path/log.txt')
        self.result_path = './result_path'

    def build_graph(self):
        # add_placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_size])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.time_step, self.output_size])

        # input_layer
        w_in = tf.get_variable(name='w_in', shape=[self.input_size, self.hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(),
                               dtype=tf.float32)
        b_in = tf.get_variable(name='b_in', shape=[self.hidden_dim, ],
                               initializer=tf.zeros_initializer(),
                               dtype=tf.float32)

        input = tf.reshape(self.X, [-1, self.input_size])
        lstm_input = tf.matmul(input, w_in) + b_in
        lstm_input = tf.reshape(lstm_input, shape=[-1, self.time_step, self.hidden_dim])

        # lstm layer
        lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.lstm_layers_num, state_is_tuple=True)

        lstm_output, final_states = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, dtype=tf.float32)
        output = tf.reshape(lstm_output, [-1, self.hidden_dim])

        # output layer
        w_out = tf.get_variable(name='w_out', shape=[self.hidden_dim, 1],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        b_out = tf.get_variable(name='b_out', shape=[1, ],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
        output = tf.matmul(output, w_out) + b_out

        self.predict = tf.nn.relu(output)

        # loss op
        self.loss = tf.reduce_mean(tf.square(tf.reshape(self.predict, [-1]) - tf.reshape(self.Y, [-1])))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss)

    def train(self):
        data, max_value, min_value = load_data('./data_path/HourLoadSet.csv')
        test_x, test_y = get_test_data(data=data, input_size=self.input_size, time_step=self.time_step,
                                       test_begin=15000, test_end=17000)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoch_num):
                print('----------------epoch %d----------------' % epoch)
                batches = get_train_data(data=data, shuffle=self.shuffle, input_size=self.input_size,
                                         batch_size=self.batch_size,
                                         time_step=self.time_step,
                                         train_begin=0, train_end=15000)
                for i, (train_x, train_y) in enumerate(batches):
                    predict_, _, loss_ = sess.run([self.predict, self.train_op, self.loss], feed_dict={self.X: train_x,
                                                                                                       self.Y: train_y})
                    # print('predict:', predict_, ' train_y', train_y)
                    if i % 10 == 0:
                        print('iter:', i, 'loss:', loss_)
                        # print('predict:-----', predict_)
            # predict
            print('============test===============')
            test_predict = []
            for step in range(len(test_x)):
                prob = sess.run(self.predict, feed_dict={self.X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)

            # test_y = np.array(test_y) * std[self.input_size] + mean[self.input_size]
            # test_predict = np.array(test_predict) * std[self.input_size] + mean[self.input_size]
            test_y = np.array(test_y) * (max_value - min_value) + min_value
            test_predict = np.array(test_predict) * (max_value - min_value) + min_value
            mape = np.mean(np.abs((test_y - test_predict)) / test_y) * 100
            print('rmse:', mape)
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b')
            plt.plot(list(range(len(test_predict))), test_y, color='r')
            plt.show()


if __name__ == '__main__':
    model = LSTM_Model()
    model.build_graph()
    model.train()
