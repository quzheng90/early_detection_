import argparse
import numpy as np
import torch
from torch.autograd import Variable, Function
from random import sample
import config
import copy
import pickle
import process_data_weibo as process_data

ospath='E:/quz/early_detection_/early_detection'
# parse = argparse.ArgumentParser()
# parser = config.parse_arguments(parse)
# args = parser.parse_args()

def word2vec(post, args, word_id_map, W):
    word_embedding = []
    mask = []
    #length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) -1
        mask_seq = np.zeros(args.sequence_length, dtype = np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_length:
            sen_embedding.append(0)
        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        #length.append(seq_len)
    return word_embedding, mask

def load_data(args):
    train, validate, test = process_data.get_data(args.text_only)
    #print(train[4][0])
    word_vector_path = ospath+'/Data/weibo/word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)  # W, W2, word_idx_map, vocab
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
    args.vocab_size = len(vocab)
    args.sequence_length = max_len
    print("translate data to embedding")

    word_embedding, mask = word2vec(validate['post_text'], args, word_idx_map, W)
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    print("translate test data to embedding")
    word_embedding, mask = word2vec(test['post_text'], args, word_idx_map, W)
    test['post_text'] = word_embedding
    test['mask']=mask
    #test[-2]= transform(test[-2])
    word_embedding, mask = word2vec(train['post_text'], args, word_idx_map, W)
    train['post_text'] = word_embedding
    train['mask'] = mask
    print("sequence length " + str(args.sequence_length))
    print("Train Data Size is "+str(len(train['post_text'])))
    print("Finished loading data ")
    return train, validate, test, W

def transform(event):
    matrix = np.zeros([len(event), max(event) + 1])
    #print("Translate  shape is " + str(matrix))
    for i, l in enumerate(event):
        matrix[i, l] = 1.00
    return matrix