import os
import tensorflow as tf
from Model import CapsuleModel
import pickle
import numpy as np
import sys
from sklearn.metrics import classification_report
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# train_context_path = './data/ubuntu_train_subtask_1._context.txt'
# train_next_path = './data/ubuntu_train_subtask_1._next.txt'
# dev_context_path = './data/ubuntu_dev_subtask_1._context.txt'
# dev_next_path = './data/ubuntu_dev_subtask_1._next.txt'
# vocab_path = "./data/ubuntu_subtask_1_vocab.txt"
with open('./data/train_dev_data.pkl', 'rb') as f:
    train_context = pickle.load(f)
    train_next = pickle.load(f)
    train_labels = pickle.load(f)
    dev_context = pickle.load(f)
    dev_next = pickle.load(f)
    dev_labels = pickle.load(f)
    vocabs_size = pickle.load(f)
    vocabs_dict = pickle.load(f)
    index_dict = pickle.load(f)
    vectors = pickle.load(f)
    train_context_lengths = pickle.load(f)
    train_next_lengths = pickle.load(f)
    dev_context_lengths = pickle.load(f)
    dev_next_lengths = pickle.load(f)

if len(sys.argv) == 2 and sys.argv[1] == 'train':
    model = CapsuleModel(vocabs_size, vectors)
    model.build_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.tables_initializer(), feed_dict={model.vocab_string: vocab_path})
    # sess.run(model.batched_iter.initializer, feed_dict={model.labels: labels})
    saver = tf.train.Saver(max_to_keep=1)
    max_acc = 0
    for e in range(50):
        try:
            sess.run(model.batched_iter.initializer,
                     feed_dict={model.context_string: train_context, model.next_tring: train_next,
                                model.labels: train_labels, model.context_string_lengths:train_context_lengths,
                                model.next_tring_lengths: train_next_lengths})
            while True:
                step_, _, cl_, acc_, _ = sess.run(
                    [model.global_step, model.get_next, model.cl, model.acc, model.train_op])
                if step_ % 50 == 0:
                    print('training step:{} | cl:{:.3f} | acc:{:.2f}%'.format(step_, cl_, acc_))
                    logger.info(('training step:{} | cl:{:.3f} | acc:{:.2f}%'.format(step_, cl_, acc_)))
        except:
            print('Training Epoch {} Done!'.format(e))
            logger.info('Training Epoch {} Done!'.format(e))
        if e % 2 == 0:
            true_labels = []
            pred_labels = []
            try:
                sess.run(model.batched_iter.initializer,
                         feed_dict={model.context_string: dev_context, model.next_tring: dev_next,
                                    model.labels: dev_labels, model.context_string_lengths:dev_context_lengths,
                                model.next_tring_lengths: dev_next_lengths})

                while True:
                    step_, data_, cl_, pred_, acc_, _ = sess.run(
                        [model.global_step, model.get_next, model.cl, model.pred, model.acc, model.train_op])
                    true_labels.extend(data_[2])
                    pred_labels.extend(pred_)
                    if step_ % 10 == 0:
                        print('Validating step:{} | cl:{:.3f} | acc:{:.2f}%'.format(step_, cl_, acc_))
                        logger.info('Validating step:{} | cl:{:.3f} | acc:{:.2f}%'.format(step_, cl_, acc_))
                    if acc_ > max_acc:
                        max_acc = acc_
                        saver.save(sess, save_path='./result/', global_step=step_)
            except:
                print('Validating Done')
                print(classification_report(true_labels,pred_labels))
                logger.info('Validating Done')
                logger.info(classification_report(true_labels,pred_labels))
