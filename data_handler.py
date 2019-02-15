import collections
import numpy as np
import math
import random
import os
import logging
from six.moves import xrange
import spacy
spacy_en = spacy.load('en')
import itertools


class DataHandler(object):
    """
    builds a dataset from . code is a modification of the Options class in the original code
    """

    def __init__(self, fname, bs, ws, vocabulary_size, exp_path, tokenize_text, iname=None, dictionary=None, counts=None):

        self.vocabulary_size = vocabulary_size
        self.batch_size = bs
        self.window_size = ws
        self.save_path = exp_path
        self.tokenize_text = tokenize_text
        if iname is not None:
            self.images = np.load(open(iname, 'rb'))
        else:
            self.images = None

        self.sents = self.read_sentences(fname)
        words = list(itertools.chain.from_iterable(self.sents))

        data_or, self.count, self.vocab_words = self.build_dataset(self.sents, words, self.vocabulary_size, dictionary)

        if counts is not None:
            #  If we are building a data handler for some held-out data, 
            #  we want to use the counts from the training data set.
            logging.info('Using existing counts with {} words'.format(len(counts)))
            self.count = counts

        self.data = self.subsampling(data_or)

        self.sample_table = self.init_sample_table()

        self.sent_index = 0
        self.buffer = []

        self.process = True

    def read_sentences(self, filename):
        logging.info('Reading data sentence wise')
        with open(filename) as f:
            sent_data = f.read().split('\n')
            sent_data = [self.tokenize(x.strip()) for x in sent_data]
        return sent_data

    def tokenize(self, text): # create a tokenizer function
        if self.tokenize_text:
            return [tok.text for tok in spacy_en.tokenizer(text)]
        else:
            return text.split()

    def build_dataset(self, sents, words, n_words, dictionary=None):
        """Process raw inputs into a dictionary and count list."""
        count = [['UNK', -1]]
        if dictionary == None:
            # Only estimate the dictionary if we are not given an existing
            # dictionary. We will get an existing dictonary with its own
            # counts if we are currently building a dataset for unseen data.
            logging.info('Computing word frequencies')
            count.extend(collections.Counter(words).most_common(n_words - 1))
            dictionary = dict()
            logging.info('Building the vocabulary')
            c = 0
            for word, _ in count:
                c += 1
                if c % 10000 == 0:
                    print('Processed {} words'.format(c))
                dictionary[word] = len(dictionary)
        else:
            logging.info('Using existing dictionary with {} words'.format(len(dictionary)))
        data = list()
        unk_count = 0
        for sent in sents:
            word_indexes = []
            for word in sent:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count += 1
                word_indexes.append(index)
            data.append(word_indexes)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, reversed_dictionary

    def contexts_for_sentence(self, sent):
        '''
        compute contexts of window_size on both sides of the input_token for all tokens in the sentence
        '''
        contexts = []
        labels = []
        for i, input_word in enumerate(sent):
            prev_words = sent[np.max([0, i - self.window_size]):i]
            subseq_words = sent[i + 1:np.min([i + 1 + self.window_size, len(sent) ])]
            contexts.append(prev_words + subseq_words)
            labels.append(input_word)
        return contexts, labels

    def generate_batch(self, neg):
        """
        generates subseqeunt batches of input tokens and corresponding contexts respecting sentence boundaries
        :param count:
        :return:
        """
        images = []
        while len(self.buffer) < self.batch_size:
            sent = self.data[self.sent_index]
            contexts, labels = self.contexts_for_sentence(sent)

            for i, label in enumerate(labels):
                if self.images is not None:
                    self.buffer += zip(contexts[i], [label] * len(labels), len(labels) * [self.images[int(np.floor(self.sent_index / 5))]])
                else:
                    self.buffer += zip(contexts[i], [label] * len(labels))

            self.sent_index += 1
            if self.sent_index >= len(self.data):
                self.sent_index = 0
                self.process=False

        pos_u = []
        pos_v = []
        for i in range(self.batch_size):
            if self.images is not None:
                c, l, v = self.buffer.pop(0)
            else:
                c, l = self.buffer.pop(0)
            pos_u.append(c)
            pos_v.append(l)
            if self.images is not None:
                images.append(v)
        neg_v = np.random.choice(self.sample_table, size=(self.batch_size, neg))
        if self.images is not None:
            return np.array(pos_u), np.array(pos_v) , neg_v, np.array(images)
        else:
            return np.array(pos_u), np.array(pos_v) , neg_v

    def save_vocab(self):
        with open(os.path.join(self.save_path, "vocab.txt"), "w") as f:
            for i in xrange(len(self.count)):
                vocab_word = self.vocab_words[i]
                f.write("%s %d\n" % (vocab_word, self.count[i][1]))

    def init_sample_table(self):
        count = [ele[1] for ele in self.count]
        pow_frequency = np.array(count) ** 0.75
        power = sum(pow_frequency)
        ratio = pow_frequency / power
        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def subsampling(self, data):
        count = [ele[1] for ele in self.count]
        frequency = np.array(count) / sum(count)
        P = dict()
        for idx, x in enumerate(frequency):
            y = (math.sqrt(x / 0.001) + 1) * 0.001 / (x + 1e-8)
            P[idx] = y
        subsampled_data = list()
        for sent in data:
            subsampled_sent = []
            for word in sent:
                if random.random() < P[word]:
                    subsampled_sent.append(word)
            subsampled_data.append(subsampled_sent)
        return subsampled_data


if __name__=="__main__":
    fname = '/home/lvx122/data/coco/text/val_en_captions.txt'
    iname = '/home/lvx122/data/coco/imgfeats/val-resnet50-avgpool.npy'
    dh = DataHandler(fname=fname, bs=15, ws=2, vocabulary_size=10000, exp_path='.', tokenize_text=False)
    i2w = dict(zip(dh.vocab_words.values(), dh.vocab_words.keys()))
    for i in range(1):
        u, v, neg = dh.generate_batch(2)
        print([dh.vocab_words[x] for x in u])
        print([dh.vocab_words[x] for x in v])
    dh = DataHandler(fname=fname, bs=3, ws=5, vocabulary_size=100, exp_path='.', tokenize_text=False, iname=iname)
    for i in range(1):
        u, v, neg, vis = dh.generate_batch(3)
        print([dh.vocab_words[x] for x in u])
        print([dh.vocab_words[x] for x in v])
        print(vis)
