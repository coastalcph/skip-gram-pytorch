import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time
import os
import numpy as np
import subprocess
from inputdata import Options, scorefunction
from data_handler import DataHandler
from model import skipgram
import logging




class word2vec:

  def __init__(self, exp_path, inputfile, vocabulary_size, embedding_dim, epoch_num, batch_size, windows_size,neg_sample_num):
      self.exp_path = exp_path
      self.data_handler = DataHandler(fname=inputfile, bs=batch_size, ws=windows_size, vocabulary_size=vocabulary_size, exp_path=self.exp_path)
      self.embedding_dim = embedding_dim
      self.windows_size = windows_size
      self.vocabulary_size = np.min([vocabulary_size, len(self.data_handler.vocab_words)])
      self.batch_size = batch_size
      self.epoch_num = epoch_num
      self.neg_sample_num = neg_sample_num


  def train(self, lr):
      model = skipgram(self.vocabulary_size, self.embedding_dim)
      if torch.cuda.is_available():
          model.cuda()
      optimizer = optim.SGD(model.parameters(),lr=lr)
      logging.info('Starting training iterations')
      for epoch in range(self.epoch_num):

          start = time.time()
          self.data_handler.process = True
          batch_num = 0
          batch_new = 0

          while self.data_handler.process:

              pos_u, pos_v, neg_v = self.data_handler.generate_batch(self.neg_sample_num)

              pos_u = Variable(torch.LongTensor(pos_u))
              pos_v = Variable(torch.LongTensor(pos_v))
              neg_v = Variable(torch.LongTensor(neg_v))


              if torch.cuda.is_available():
                  pos_u = pos_u.cuda()
                  pos_v = pos_v.cuda()
                  neg_v = neg_v.cuda()

              optimizer.zero_grad()
              loss = model(pos_u, pos_v, neg_v, self.batch_size)

              loss.backward()
   
              optimizer.step()


              if batch_num%30000 == 0:
                  torch.save(model.state_dict(), os.path.join(self.exp_path, 'skipgram.epoch{}.batch{}'.format(epoch,batch_num)))

              if batch_num%2000 == 0:
                  end = time.time()
                  word_embeddings = model.input_embeddings()
                  sp1, sp2 = scorefunction(word_embeddings, self.exp_path)
                  logging.info('epoch,batch=%2d %5d: sp=%1.3f %1.3f  pair/sec = %4.2f loss=%4.3f'%(epoch, batch_num, sp1, sp2, (batch_num-batch_new)*self.batch_size/(end-start),loss.item()))
                  batch_new = batch_num
                  start = time.time()

              batch_num = batch_num + 1


      logging.info("Optimization Finished!")
      logging.info('Saving embeddings to {}'.format(os.path.join(self.exp_path, 'sgns{}.txt'.format(self.embedding_dim))))
      model.save_embedding(os.path.join(self.exp_path, 'sgns{}.txt'.format(self.embedding_dim)), self.data_handler.vocab_words)

def create_exp_dir(exp_path):
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return

def set_up_logging(exp_path):
    ######################################################################################################
    ################################# LOGGING ############################################################
    ######################################################################################################
    # create a logger and set parameters
    logfile = os.path.join(exp_path, 'log.txt')
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

def main(args):
    create_exp_dir(args.exp_path)
    set_up_logging(args.exp_path)

    wc = word2vec(exp_path=args.exp_path, inputfile=args.fname, vocabulary_size=args.vocab_size, embedding_dim=args.emb_dim,
                  epoch_num=args.epochs, batch_size=args.bs, windows_size=args.ws,neg_sample_num=args.neg)
    wc.train(lr=args.lr)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Learning word embeddings via sgns')

    parser.add_argument('--fname', type=str, default='',
                            help="Text input file with one sentence per line")
    parser.add_argument('--exp_path', type=str, default='tmp',
                        help="Vocabulary and embeddings are written to this directory")
    parser.add_argument('--vocab_size', type=int, default=100000,
                            help="Number of words in the vocabulary")
    parser.add_argument('--emb_dim', type=int, default=300,
                        help="Dimension of the learned embeddings")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--bs', type=int, default=16,
                        help="Batch size")
    parser.add_argument('--ws', type=int, default=5,
                        help="Window size")
    parser.add_argument('--neg', type=int, default=10,
                        help="Number of negative samples considered per input word in the negative sampling objective")
    parser.add_argument('--lr', type=float, default=0.2,
                        help="Learning rate")

    args = parser.parse_args()
    main(args)
















