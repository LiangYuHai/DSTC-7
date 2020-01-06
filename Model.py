import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from helper import key_func, reduce_func, get_shape
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# with open('./data/dev_labels.pkl', 'rb') as f:
#     labels = pickle.load(f)
epsilon = 1e-9


class Model:
    def __init__(self, vocabs_size, embedding):
        self.batch_size = 128
        self.vocab_size = vocabs_size
        self.mask_with_y = True
        self.num_label = 2
        self.vec_len = 128
        self.lstm_hidden_size = 300
        self.embedding = embedding
        self.keep_rate = 0.8

    def build_graph(self):
        self.context_string = tf.placeholder(dtype=tf.int32, shape=[None, 300])
        self.next_tring = tf.placeholder(dtype=tf.int32, shape=[None, 30])
        self.context_string_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        self.next_tring_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        self.context_masks = tf.placeholder(dtype=tf.float32, shape=[None, 300])
        self.next_masks = tf.placeholder(dtype=tf.float32, shape=[None, 30])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None])
        context = tf.data.Dataset.from_tensor_slices(self.context_string)
        next = tf.data.Dataset.from_tensor_slices(self.next_tring)
        context_masks = tf.data.Dataset.from_tensor_slices(self.context_masks)
        next_masks = tf.data.Dataset.from_tensor_slices(self.next_masks)
        context_lengths = tf.data.Dataset.from_tensor_slices(self.context_string_lengths)
        next_lengths = tf.data.Dataset.from_tensor_slices(self.next_tring_lengths)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        # vocab_table = lookup_ops.index_table_from_file(self.vocab_string, default_value=0)
        # reverse_vocab_table = lookup_ops.index_to_string_table_from_file(self.vocab_string,
        #                                                                  default_value='UNK')
        data = tf.data.Dataset.zip((context, next, labels, context_masks, next_masks, context_lengths, next_lengths))
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
        (self.c, self.x, self.label, self.c_masks, self.x_masks, self.c_lens, self.x_lens) = self.get_next
        self.y = tf.one_hot(self.label, depth=2)
        embedding = tf.Variable(initial_value=self.embedding, trainable=False, dtype=tf.float32)

        c = tf.nn.embedding_lookup(embedding, self.c)
        c = tf.nn.dropout(c, keep_prob=0.5)

        x = tf.nn.embedding_lookup(embedding, self.x)
        x = tf.nn.dropout(x, keep_prob=0.5)

        with tf.variable_scope('lstmc1'):

            lstm_fw_cell_c_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            lstm_bw_cell_c_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            output_c_2, state_c_2 = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_c_2,
                lstm_bw_cell_c_2,
                c,
                dtype=tf.float32,
                time_major=False,
                # sequence_length=self.c_lens,
            )
            output_c = tf.concat(output_c_2, -1)
        with tf.variable_scope('lstmx1'):

            lstm_fw_cell_x = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            lstm_bw_cell_x = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            output_x, state_x = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_x,
                lstm_bw_cell_x,
                x,
                dtype=tf.float32,
                time_major=False,
                sequence_length=self.x_lens,
            )
            output_x = tf.concat(output_x, -1)
        output_c = output_c * tf.expand_dims(self.c_masks, -1)
        output_x = output_x * tf.expand_dims(self.x_masks, -1)

        output_x_T = tf.transpose(output_x, [0, 2, 1])
        atten = tf.matmul(output_c, output_x_T)
        atten1 = tf.exp(atten)

        atten1 = atten1 * tf.expand_dims(self.c_masks, -1)
        alpha = atten1 / (tf.reduce_sum(atten1, -1, keep_dims=True) + 1e-8)
        X = tf.matmul(tf.transpose(alpha, [0,2,1]),output_c)

        # atten2 = atten - tf.reduce_max(atten, axis=1, keep_dims=True)
        atten2 = tf.exp(tf.transpose(atten, [0, 2, 1]))
        atten2 = atten2 * tf.expand_dims(self.x_masks, -1)

        beta = atten2 / (tf.reduce_sum(atten2, -1, keep_dims=True) + 1e-8)
        C = tf.matmul(tf.transpose(beta, [0,2,1]), output_x)

        C_atten = tf.concat([C, output_c, C * output_c, C - output_c], axis=-1)
        X_atten = tf.concat([X, output_x, X * output_x, X - output_x], axis=-1)

        C_atten = tf.layers.dense(C_atten, self.lstm_hidden_size, activation=tf.nn.relu, name='fnn',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        X_atten = tf.layers.dense(X_atten, self.lstm_hidden_size, activation=tf.nn.relu, name='fnn',
                                   reuse=True)
        C_atten = tf.nn.dropout(C_atten, self.keep_rate)
        X_atten = tf.nn.dropout(X_atten, self.keep_rate)

        # C_atten = tf.transpose(C_atten, [1, 0, 2])
        # X_atten = tf.transpose(X_atten, [1, 0, 2])
        with tf.variable_scope('lstmc2'):

            lstm_fw_cell_c_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
            lstm_bw_cell_c_2 = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size)
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
            output_x_2, state_x_2 = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_x_2,
                lstm_bw_cell_x_2,
                X_atten,
                dtype=tf.float32,
                time_major=False,
                sequence_length=self.x_lens,
            )
            output_x_2 = tf.concat(output_x_2, -1)
        output_c_2 = output_c_2 * tf.expand_dims(self.c_masks, -1)
        output_x_2 = output_x_2 * tf.expand_dims(self.x_masks, -1)
        w = tf.concat(
            [tf.reduce_max(output_c_2, axis=1, keep_dims=False),
             tf.reduce_sum(output_c_2, 1, keep_dims=False) / tf.cast(tf.expand_dims(self.c_lens, -1), tf.float32),
             tf.reduce_max(output_x_2, axis=1, keep_dims=False),
             tf.reduce_sum(output_x_2, 1, keep_dims=False) / tf.cast(tf.expand_dims(self.x_lens, -1), tf.float32)],
            axis=-1)
        w = tf.nn.dropout(w, self.keep_rate)
        y = tf.layers.dense(w, self.lstm_hidden_size, tf.nn.tanh, name='fnn1',
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        y = tf.nn.dropout(y, self.keep_rate)
        logits = tf.layers.dense(y, 2, name='fnn2', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.cl = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        self.cl = tf.reduce_mean(self.cl)
        self.pred = tf.argmax(logits, -1, output_type=tf.int32)
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(self.label, self.pred), tf.float32))

        op = tf.train.AdamOptimizer(0.0002)
        grads = op.compute_gradients(self.cl)
        clipped_grads = [(tf.clip_by_value(grad, 10, -10), var) for grad, var in grads if grad is not None]
        self.train_op = op.apply_gradients(clipped_grads, global_step=self.global_step)
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cl, tvars), clip_norm=20)
        # self.train_op = op.apply_gradients(grads_and_vars=zip(grads, tvars), global_step=self.global_step)
