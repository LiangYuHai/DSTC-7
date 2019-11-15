import tensorflow as tf


def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
    # Calculate bucket_width by maximum source sequence length.
    # Pairs with length [0, bucket_width) go to bucket 0, length
    # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
    # over ((num_bucket-1) * bucket_width) words all go into the last bucket.

    bucket_width = 10

    # Bucket sentence pairs by the length of their source sentence and target
    # sentence.
    bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
    return tf.to_int64(bucket_id)


def reduce_func(unused_key, windowed_data):
    return windowed_data.padded_batch(batch_size=100, padded_shapes=(tf.TensorShape([None]),
                                                                    tf.TensorShape([None]),
                                                                    tf.TensorShape([]),
                                                                    tf.TensorShape([]),
                                                                    tf.TensorShape([])), padding_values=(0, 0, 0, 0, 0),drop_remainder=True
                                      )


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return (shape)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set)) #也是命中个数 除以 ground-truth 的个数 求和后再除以用户个数。
            true_users += 1
    return sum_recall / true_users