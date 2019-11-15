import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from helper import key_func, reduce_func, get_shape
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
# with open('./data/dev_labels.pkl', 'rb') as f:
#     labels = pickle.load(f)
epsilon = 1e-9


class CapsuleModel:
    def __init__(self, vocabs_size, embedding):
        self.batch_size = 2
        self.vocab_size = vocabs_size
        self.mask_with_y = True
        self.num_label = 2
        self.vec_len = 128
        self.lstm_hidden_size = 100
        self.embedding = embedding
    def squash(self, vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return (vec_squashed)

    def routing(self, input, b_IJ, num_outputs=10, num_dims=16):
        ''' The routing algorithm.

        Args:
            input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
                   shape, num_caps_l meaning the number of capsule in the layer l.
            num_outputs: the number of output capsules.
            num_dims: the number of dimensions for output capsule.
        Returns:
            A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
            representing the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer l, and
            v_j the vector output of capsule j in the layer l+1.
         '''
        # input: [batch_size, 6, 1, 6, 1]
        # W: [1, 6, 16 * 2, 6, 1]
        # b: [1, 1, 2, 16, 1]
        input_shape = get_shape(input)
        W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs, input_shape[3], 1],
                            dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
        biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

        # Eq.2, calc u_hat
        # Since tf.matmul is a time-consuming op,
        # A better solution is using element-wise multiply, reduce_sum and reshape
        # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
        # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
        # reshape to [a, c]
        input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
        # assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

        u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
        u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
        # assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

        # line 3,for r iterations do
        for r_iter in range(3):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                # => [batch_size, 1152, 10, 1, 1]
                c_IJ = tf.nn.softmax(b_IJ, axis=2)

                # At last iteration, use `u_hat` in order to receive gradients from the following graph
                if r_iter == 3 - 1:
                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 1152, 10, 16, 1]
                    s_J = tf.multiply(c_IJ, u_hat)
                    # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                    # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]

                    # line 6:
                    # squash using Eq.1,
                    v_J = self.squash(s_J)
                    # assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
                elif r_iter < 3 - 1:  # Inner iterations, do not apply backpropagation
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                    v_J = self.squash(s_J)

                    # line 7:
                    # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                    # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                    # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                    v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                    u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                    # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                    # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                    b_IJ += u_produce_v

        return (v_J)

    def build_graph(self):
        self.context_string = tf.placeholder(dtype=tf.int32, shape=[None, 300])
        self.next_tring = tf.placeholder(dtype=tf.int32, shape=[None, 30])
        self.context_string_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        self.next_tring_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        # self.vocab_string = tf.placeholder(tf.string)
        # context = tf.data.TextLineDataset(self.context_string)
        # next = tf.data.TextLineDataset(self.next_tring)
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None])
        context = tf.data.Dataset.from_tensor_slices(self.context_string)
        next = tf.data.Dataset.from_tensor_slices(self.next_tring)
        context_lengths = tf.data.Dataset.from_tensor_slices(self.context_string_lengths)
        next_lengths = tf.data.Dataset.from_tensor_slices(self.next_tring_lengths)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        # vocab_table = lookup_ops.index_table_from_file(self.vocab_string, default_value=0)
        # reverse_vocab_table = lookup_ops.index_to_string_table_from_file(self.vocab_string,
        #                                                                  default_value='UNK')
        data = tf.data.Dataset.zip((context, next, labels, context_lengths, next_lengths))
        data = data.shuffle(100, seed=123).batch(self.batch_size)
        #
        # data = data.map(lambda context, next, label: (
        #     tf.string_split([context]).values, tf.string_split([next]).values,
        #     tf.to_int32(label)))
        # data = data.filter(lambda context, next, label: tf.logical_and(tf.size(context) > 0, tf.size(next) > 0))
        # data = data.map(lambda context, next, label: (tf.cast(vocab_table.lookup(context), tf.int32),
        #                                               tf.cast(vocab_table.lookup(next), tf.int32),
        #                                               label))
        #
        # data = data.map(lambda context, next, label: (context, next, label, tf.size(context), tf.size(next)))
        # data = data.map(lambda context, next, label, c_len, n_len: (
        # context[-280:] if (c_len >= 280) is not None else tf.concat([tf.zeros([280 - c_len], dtype=tf.int32), context], axis=-1),next, label))
        # data = data.prefetch(1000)
        # # batched_dataset = data.apply(
        # #     tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=self.batch_size))
        # batched_dataset = data.batch(10)
        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.001, self.global_step, 2000, 0.8, staircase=False)
        self.batched_iter = data.make_initializable_iterator()
        self.get_next = self.batched_iter.get_next()
        (self.c, self.x, self.label, self.c_lens, self.x_lens) = self.get_next
        self.y = tf.one_hot(self.label, depth=2)
        embedding = tf.Variable(initial_value=self.embedding, trainable=False, dtype=tf.float32)
        # embedding = tf.get_variable(name='embedding', shape=[self.vocab_size, 1024], dtype=tf.float32)
        c = tf.nn.embedding_lookup(embedding, self.c)
        c = tf.layers.dense(c, 300, tf.nn.relu)
        x = tf.nn.embedding_lookup(embedding, self.x)
        x = tf.layers.dense(x, 300, tf.nn.relu)
        # conv_c4 = tf.layers.conv1d(c, filters=128, kernel_size=4, strides=1, padding='SAME')
        # conv_x4 = tf.layers.conv1d(x, filters=128, kernel_size=4, strides=1, padding='SAME')
        # conv_c4_max = tf.reduce_max(conv_c4, axis=1, keep_dims=True)
        # conv_x4_max = tf.reduce_max(conv_x4, axis=1, keep_dims=True)
        # conv_c4_mean = tf.reduce_mean(conv_c4, axis=1, keep_dims=True)
        # conv_x4_mean = tf.reduce_mean(conv_x4, axis=1, keep_dims=True)
        # conv_c3 = tf.layers.conv1d(c, filters=128, kernel_size=3, strides=1, padding='SAME')
        # conv_x3 = tf.layers.conv1d(x, filters=128, kernel_size=3, strides=1, padding='SAME')
        # conv_c3_max = tf.reduce_max(conv_c3, axis=1, keep_dims=True)
        # conv_x3_max = tf.reduce_max(conv_x3, axis=1, keep_dims=True)
        # conv_c3_mean = tf.reduce_mean(conv_c3, axis=1, keep_dims=True)
        # conv_x3_mean = tf.reduce_mean(conv_x3, axis=1, keep_dims=True)
        # conv_c2 = tf.layers.conv1d(c, filters=128, kernel_size=2, strides=1, padding='SAME')
        # conv_x2 = tf.layers.conv1d(x, filters=128, kernel_size=2, strides=1, padding='SAME')
        # conv_c2_max = tf.reduce_max(conv_c2, axis=1, keep_dims=True)
        # conv_x2_max = tf.reduce_max(conv_x2, axis=1, keep_dims=True)
        # conv_c2_mean = tf.reduce_mean(conv_c2, axis=1, keep_dims=True)
        # conv_x2_mean = tf.reduce_mean(conv_x2, axis=1, keep_dims=True)

        # conv_c = tf.concat([conv_c4, conv_c3, conv_c2], axis=-1)
        # conv_x = tf.concat([conv_x4, conv_x3, conv_x2], axis=-1)
        # atten = tf.concat([c, x], axis=1)
        # conv_c3 = tf.transpose(conv_c, [0,2,1])
        # atten = tf.matmul(conv_x, conv_c3)
        # atten = tf.layers.dense(atten, self.vec_len)
        # conv_c = tf.transpose(conv_c, [0, 2, 1])
        # atten = tf.matmul(conv_x, conv_c)
        # conv_c = tf.layers.dense(conv_c, 256,activation=None)
        # conv_x = tf.layers.dense(conv_x, 256, activation=None)
        # atten = tf.matmul(conv_x, conv_c)

        with tf.variable_scope('lstmc1'):
            lstm_fw_cell_c = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            lstm_bw_cell_c = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            # lstm_fw_cell_c = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_c, input_keep_prob=0.90, output_keep_prob=0.80)
            # lstm_bw_cell_c = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_c, input_keep_prob=0.90, output_keep_prob=0.80)
            output_c, state_c = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_c,
                lstm_bw_cell_c,
                c,
                dtype=tf.float32,
                time_major=False,
                # sequence_length = self.c_lens,
            )

            output_c = tf.concat(output_c, -1)
        with tf.variable_scope('lstmx1'):
            lstm_fw_cell_x = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            lstm_bw_cell_x = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            # lstm_fw_cell_x = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_x, input_keep_prob=0.90, output_keep_prob=0.80)
            # lstm_bw_cell_x = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_x, input_keep_prob=0.90, output_keep_prob=0.80)
            output_x, state_x = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_x,
                lstm_bw_cell_x,
                x,
                dtype=tf.float32,
                time_major=False,
                # sequence_length=self.x_lens,
            )
            output_x = tf.concat(output_x, -1)

        output_c_T = tf.transpose(output_c, [0, 2, 1])
        atten = tf.matmul(output_x, output_c_T)
        alpha = tf.nn.softmax(tf.reduce_sum(atten, axis=-1, keep_dims=True))
        X = tf.multiply(alpha, output_x)
        beta = tf.nn.softmax(tf.reduce_sum(atten, axis=1, keep_dims=True))
        C = tf.multiply(beta, output_c_T)
        C = tf.transpose(C, [0, 2, 1])
        C_atten = tf.concat([C, output_c, C - output_c, C * output_c], axis=-1)
        X_atten = tf.concat([X, output_x, X - output_x, X * output_x], axis=-1)
        C_atten = tf.layers.dense(C_atten, self.lstm_hidden_size, tf.nn.relu)
        X_atten = tf.layers.dense(X_atten, self.lstm_hidden_size, tf.nn.relu)

        with tf.variable_scope('lstmc2'):
            lstm_fw_cell_c_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            lstm_bw_cell_c_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            # lstm_fw_cell_c_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_c_2, input_keep_prob=0.90, output_keep_prob=0.80)
            # lstm_bw_cell_c_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_c_2, input_keep_prob=0.90, output_keep_prob=0.80)
            output_c_2, state_c_2 = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_c_2,
                lstm_bw_cell_c_2,
                C_atten,
                dtype=tf.float32,
                time_major=False,
                # sequence_length=self.c_lens,
            )
            output_c_2 = tf.concat(output_c_2, -1)
        with tf.variable_scope('lstmx2'):
            lstm_fw_cell_x_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            lstm_bw_cell_x_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            # lstm_fw_cell_x_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_x_2, input_keep_prob=0.90, output_keep_prob=0.80)
            # lstm_bw_cell_x_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_x_2, input_keep_prob=0.90, output_keep_prob=0.80)
            output_x_2, state_x_2 = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_x_2,
                lstm_bw_cell_x_2,
                X_atten,
                dtype=tf.float32,
                time_major=False,
                # sequence_length=self.x_lens,
            )
            output_x_2 = tf.concat(output_x_2, -1)

        w = tf.concat(
            [tf.reduce_max(output_c_2, axis=1), tf.reduce_mean(output_c_2, axis=1), tf.reduce_max(output_x_2, axis=1),
             tf.reduce_mean(output_x_2, axis=1)], axis=-1)
        y0 = tf.layers.dense(w, self.lstm_hidden_size, tf.nn.relu)
        y = tf.layers.dense(y0, self.lstm_hidden_size, tf.nn.tanh)
        y = y0 + y
        # y = tf.layers.dense(y, self.lstm_hidden_size, tf.nn.tanh)
        logits = tf.layers.dense(y, 2)
        self.cl = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits)
        self.cl = tf.reduce_mean(self.cl)
        self.pred = tf.argmax(logits,-1, output_type=tf.int32)
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(self.label, self.pred), tf.float32))

        # atten = tf.expand_dims(w, -1) #[batch_size, 6, 6, 1]
        # atten = self.squash(atten) #[batch_size, 6, 6, 1]
        # x_size = get_shape(atten)[1]
        # c_size = get_shape(atten)[2]
        # atten = tf.reshape(atten, shape=(self.batch_size, x_size, 1, c_size, 1))#[batch_size, 6, 1, 6, 1]
        #
        # with tf.variable_scope('routing'):
        #     # b_IJ: [batch_size, 1152, 10, 1, 1],
        #     # about the reason of using 'batch_size', see issue #21
        #     b_IJ = tf.Variable(np.zeros(shape=[self.batch_size, x_size, 2, 1, 1], dtype=np.float32))
        #     # b_IJ = tf.constant(np.zeros([self.batch_size, self.x_size, 2, 1, 1], dtype=np.float32))
        #     capsules = self.routing(atten, b_IJ, num_outputs=2, num_dims=16)
        #     self.caps2 = tf.squeeze(capsules, axis=1)#[batch_size, num_label, 16, 1]
        #
        # with tf.variable_scope('Masking'):
        #     # a). calc ||v_c||, then do softmax(||v_c||)
        #     # [batch_size, 10, 16, 1] => [batch_size, 2, 1, 1]
        #     self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
        #                                           axis=2, keepdims=True) + epsilon)
        #     self.softmax_v = tf.nn.softmax(self.v_length, axis=1)
        #     # assert self.softmax_v.get_shape() == [cfg.batch_size, self.num_label, 1, 1]
        #     # logits = tf.reshape(self.v_length, [self.batch_size,2])
        #     # self.cl = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits)
        #     # b). pick out the index of max softmax val of the 10 caps
        #     # [batch_size, 10, 1, 1] => [batch_size] (index)
        #     self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
        #     # assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
        #     self.argmax_idx = tf.reshape(self.argmax_idx, shape=(self.batch_size,))
        #
        #     # Method 1.
        #     if not self.mask_with_y:
        #         # c). indexing
        #         # It's not easy to understand the indexing process with argmax_idx
        #         # as we are 3-dim animal
        #         masked_v = []
        #         for batch_size in range(self.batch_size):
        #             v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
        #             masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))
        #
        #         self.masked_v = tf.concat(masked_v, axis=0)
        #         assert self.masked_v.get_shape() == [self.batch_size, 1, 16, 1]
        #     # Method 2. masking with true label, default mode
        #     else:
        #         self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.y, (-1, self.num_label, 1)))
        #         self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)
        #
        # # 2. Reconstructe the MNIST images with 3 FC layers
        # # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        # with tf.variable_scope('Decoder'):
        #     vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
        #     fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=18)
        #     fc1 = tf.nn.relu(fc1)
        #     fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=64)
        #     fc2 = tf.nn.relu(fc2)
        #     self.decoded = tf.contrib.layers.fully_connected(fc2,
        #                                                      num_outputs=x_size * c_size,
        #                                                      activation_fn=tf.sigmoid)
        #
        # max_l = tf.square(tf.maximum(0., 0.9 - self.v_length))
        # # max_r = max(0, ||v_c||-m_minus)^2
        # max_r = tf.square(tf.maximum(0., self.v_length - 0.1))
        # assert max_l.get_shape() == [self.batch_size, self.num_label, 1, 1]
        #
        # # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        # max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
        # max_r = tf.reshape(max_r, shape=(self.batch_size, -1))
        #
        # # calc T_c: [batch_size, 10]
        # # T_c = Y, is my understanding correct? Try it.
        # T_c = self.y
        # # [batch_size, 10], element-wise multiply
        # L_c = T_c * max_l + 0.5 * (1 - T_c) * max_r
        #
        # self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        #
        # # 2. The reconstruction loss
        # orgin = tf.reshape(atten, shape=(self.batch_size, -1))
        # squared = tf.square(self.decoded - orgin)
        # self.reconstruction_err = tf.reduce_mean(squared)
        #
        # # 3. Total loss
        # # The paper uses sum of squared error as reconstruction error, but we
        # # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # # mean squared error. In order to keep in line with the paper,the
        # # regularization scale should be 0.0005*784=0.392
        # self.total_loss = self.margin_loss + 0.0005*6*6 * self.reconstruction_err
        # self.acc = tf.reduce_mean(tf.cast(tf.equal(self.argmax_idx, self.label),dtype=tf.float32))
        self.train_op = tf.train.AdamOptimizer(0.005).minimize(self.cl, self.global_step)
