import json
import numpy as np
import pickle

np.random.seed(1234)

output1 = open('subtask1_pretrain.txt', 'w')
output3 = open('./dev.txt', 'w', encoding='utf-8')
output4 = open('./train.txt', 'w', encoding='utf-8')
output3.write('context\tnext\tlabel\n')
output4.write('context\tnext\tlabel\n')

def makemb(path, output):

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
        output.write(context.strip(' _eou_ ') + '\t' + neg_answer[r[0]].strip() + '\t' + '0\n')
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
        output.write(context.strip(' _eou_ ') + '\t' + options_for_correct_answers.strip() + '\t' + '1\n')
        # vocabs.extend(context.split() + options_for_correct_answers.split() + neg_answer[r[0]].split()+ neg_answer[r[1]].split()+ neg_answer[r[2]].split()+ neg_answer[r[3]].split())
        assert len(all_context) == len(all_next) == len(labels)
    output.close()

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
dev_context, dev_next, dev_labels = makemb('./ubuntu_dev_subtask_1.json', output3)
train_context, train_next, train_labels = makemb('./ubuntu_train_subtask_1.json', output4)
assert len(dev_context) == len(dev_next) == len(dev_labels)
assert len(train_context) == len(train_next) == len(train_labels)

output1.close()
vocabs = []
vocabs = save_vectors('./vectors.txt', 300, vocabs)
glove_6B_300d_wv = save_vectors('./glove.6B.300d.txt', 300, vocabs)


emb = []
emb.append(np.zeros([300],dtype=np.float32))
emb.append(np.random.normal(size=[300]))

for i in vocabs:
    v1 = glove_6B_300d_wv.get(i, np.random.normal(size=[300]))
    emb.append(v1)
emb = np.asarray(emb,dtype=np.float32)

output2 = open('./vocab.txt','w', encoding='utf-8')
for i in vocabs:
    output2.write(i + '\n')
output2.close()
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
    masks = []
    for example in data:
        ids_ = []
        mask = []
        for word in example:
            ids_.append(vocabs_dict.get(word, 1))
            mask.append(1)
        seq_lengts.append(max_len if len(ids_)>max_len else len(ids_))
        data_ids.append(ids_)
        masks.append(mask)
    data_ids = tf.keras.preprocessing.sequence.pad_sequences(data_ids,
                                                             value=vocabs_dict['PAD'],
                                                             padding=padding,
                                                             truncating=truncating,
                                                             maxlen=max_len)
    mask_pad = tf.keras.preprocessing.sequence.pad_sequences(masks,
                                                             value=0,
                                                             padding=padding,
                                                             truncating=truncating,
                                                             maxlen=max_len)
    return data_ids, seq_lengts, mask_pad


train_context, train_context_lengths, train_context_masks = preprocess_data(train_context, 300, 'pre', 'pre')
train_next, train_next_lengths, train_next_masks = preprocess_data(train_next, 30, 'post', 'post')
dev_context, dev_context_lengths, dev_context_masks = preprocess_data(dev_context, 300, 'pre', 'pre')
dev_next, dev_next_lengths, dev_next_masks = preprocess_data(dev_next, 30, 'post', 'post')
vocabs_size = len(vocabs_dict.items())
with open('./train_dev_data.pkl', 'wb') as f:
    pickle.dump(train_context, f)
    pickle.dump(train_next, f)
    pickle.dump(train_labels, f)
    pickle.dump(train_context_masks, f)
    pickle.dump(train_next_masks, f)
    pickle.dump(dev_context, f)
    pickle.dump(dev_next, f)
    pickle.dump(dev_labels, f)
    pickle.dump(dev_context_masks, f)
    pickle.dump(dev_next_masks, f)
    pickle.dump(vocabs_size, f)
    pickle.dump(vocabs_dict, f)
    pickle.dump(index_dict, f)
    pickle.dump(emb, f),
    pickle.dump(train_context_lengths,f)
    pickle.dump(train_next_lengths, f)
    pickle.dump(dev_context_lengths, f)
    pickle.dump(dev_next_lengths, f)
