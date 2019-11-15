import json
import numpy as np
import pickle

np.random.seed(1234)

output1 = open('subtask1_pretrain.txt', 'w')

def make1(path):

    path_prefix = path.strip('json')

    # output2 = open(path_prefix + '_next.txt', 'w')
    # output3 = open(path_prefix + '_labels.txt', 'w')
    all_context = []
    all_next = []
    labels = []

    f = open(path, 'r', encoding='utf-8')
    input = json.load(f)
    for example in input:
        messages_so_far = example['messages-so-far']
        context = ''
        neg_answer = []
        for index, m in enumerate(messages_so_far):
            output1.write(m['utterance'] + '\n')
            if index == len(messages_so_far)-1:
                context += m['utterance']
            elif m['speaker'] == messages_so_far[index+1]['speaker']:
                context += m['utterance'] + ' _eou_ '
            elif m['speaker'] != messages_so_far[index+1]['speaker']:
                context += m['utterance'] + ' _eou_  _eot_ '
            else: print('???')

        options_for_correct_answers = example['options-for-correct-answers'][0]['utterance']
        output1.write(options_for_correct_answers + '\n')
        options_for_next = example["options-for-next"]
        for n in options_for_next:
            output1.write(n["utterance"] + '\n')
            neg_answer.append(n["utterance"])
        r = np.random.choice(a=len(neg_answer), size=4)
        # output1.write(context.strip(' _eou_ ') + '\n')
        all_context.append(context.strip(' _eou_ ').split())
        # output2.write(neg_answer[r[0]] + '\n')
        all_next.append(neg_answer[r[0]].strip().split())
        # output3.write('0' + '\n')
        labels.append(0)
        # all_context.append(context.strip(' _eou_ ').split())
        # all_next.append(neg_answer[r[1]].strip().split())
        # # output3.write('0' + '\n')
        # labels.append(0)
        # all_context.append(context.strip(' _eou_ ').split())
        # all_next.append(neg_answer[r[2]].strip().split())
        # # output3.write('0' + '\n')
        # labels.append(0)
        # all_context.append(context.strip(' _eou_ ').split())
        # all_next.append(neg_answer[r[3]].strip().split())
        # # output3.write('0' + '\n')
        # labels.append(0)
        # output1.write(context.strip(' _eou_ ') + '\n')
        all_context.append(context.strip(' _eou_ ').split())
        # output2.write(options_for_correct_answers + '\n')
        all_next.append(options_for_correct_answers.strip().split())
        # output3.write('1' + '\n')
        labels.append(1)
        # vocabs.extend(context.split() + options_for_correct_answers.split() + neg_answer[r[0]].split()+ neg_answer[r[1]].split()+ neg_answer[r[2]].split()+ neg_answer[r[3]].split())
        assert len(all_context) == len(all_next) == len(labels)

    # output2.close()
    # output3.close()
    f.close()
    return all_context, all_next, labels

def save_vectors(path, dim, vocabs):
    file = open(path, 'r', encoding='utf-8')
    if not vocabs:
        line = file.readline()
        while line:
            elements = line.strip().split()
            word = elements[0]
            vocabs.append(word)
            line = file.readline()
        file.close()

        return vocabs

    wv = {}
    # vectors.append(np.zeros([300],dtype=np.float32))
    # vectors.append(np.random.normal(size=[300]))
    line = file.readline()
    while line:
        elements = line.strip().split()
        try:
            word = elements[0]
            if word in vocabs:
                vector = [float(i) for i in elements[1:]]
                wv[word] = vector
        except:
            pass
        line = file.readline()
    file.close()

    return wv

vocabs_dict = {}
dev_context, dev_next, dev_labels = make1('./ubuntu_dev_subtask_1.json')
train_context, train_next, train_labels = make1('./ubuntu_train_subtask_1.json')
output1.close()
vocabs = []
vocabs = save_vectors('./vectors.txt', 300, vocabs)
glove_6B_300d_wv = save_vectors('./glove.6B.300d.txt', 300, vocabs)
glove_840B_300d_wv = save_vectors('./glove.840B.300d.txt', 300, vocabs)
glove_twitter_270B_200d_wv = save_vectors('./glove.twitter.27B.200d.txt', 200, vocabs)
# vectors = []
# vectors.append(np.zeros([800],dtype=np.float32))
# vectors.append(np.random.normal(size=[800]))

e1 = []
e2 = []
e3 = []
e1.append(np.zeros([800],dtype=np.float32))
e1.append(np.random.normal(size=[800]))
e2.append(np.zeros([800],dtype=np.float32))
e2.append(np.random.normal(size=[800]))
e3.append(np.zeros([800],dtype=np.float32))
e3.append(np.random.normal(size=[800]))
for i in vocabs:
    v1 = glove_6B_300d_wv.get(i, np.random.normal(size=[300]))
    v2 = glove_840B_300d_wv.get(i, np.random.normal(size=[300]))
    v3 = glove_twitter_270B_200d_wv.get(i, np.random.normal(size=[200]))
    # v = np.concatenate([v1, v2, v3], axis=-1)
    # vectors.append(v)
    e1.append(v1)
    e2.append(v2)
    e3.append(v3)
e1 = np.asarray(e1,dtype=np.float32)
e2 = np.asarray(e2,dtype=np.float32)
e3 = np.asarray(e3,dtype=np.float32)
for index, i in enumerate(vocabs):
    vocabs_dict[i] = index + 2
vocabs_dict['UNK'] = 1
vocabs_dict['PAD'] = 0
index_dict = {item[1]: item[0] for item in vocabs_dict.items()}
# assert len(vectors) == len(vocabs_dict.items())
import tensorflow as tf
def preprocess_data(data, max_len, padding, truncating):
    data_ids = []
    seq_lengts = []
    for example in data:
        ids_ = []
        for word in example:
            ids_.append(vocabs_dict.get(word, 1))
        seq_lengts.append(len(ids_))
        data_ids.append(ids_)
    data_ids = tf.keras.preprocessing.sequence.pad_sequences(data_ids,
                                                             value=vocabs_dict['PAD'],
                                                             padding=padding,
                                                             truncating=truncating,
                                                             maxlen=max_len)
    return data_ids, seq_lengts


train_context, train_context_lengths = preprocess_data(train_context, 300, 'pre', 'pre')
train_next, train_next_lengths = preprocess_data(train_next, 30, 'post', 'post')
dev_context, dev_context_lengths = preprocess_data(dev_context, 300, 'pre', 'pre')
dev_next, dev_next_lengths = preprocess_data(dev_next, 30, 'post', 'post')
vocabs_size = len(vocabs_dict.items())
with open('./train_dev_data.pkl', 'wb') as f:
    pickle.dump(train_context, f)
    pickle.dump(train_next, f)
    pickle.dump(train_labels, f)
    pickle.dump(dev_context, f)
    pickle.dump(dev_next, f)
    pickle.dump(dev_labels, f)
    pickle.dump(vocabs_size, f)
    pickle.dump(vocabs_dict, f)
    pickle.dump(index_dict, f)
    pickle.dump(e1, f),
    pickle.dump(e2, f),
    pickle.dump(e3, f),
    pickle.dump(train_context_lengths,f)
    pickle.dump(train_next_lengths, f)
    pickle.dump(dev_context_lengths, f)
    pickle.dump(dev_next_lengths, f)
