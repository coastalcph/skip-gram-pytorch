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
from utils import load_embeddings_from_file
from data_handler import DataHandler
from model import skipgram, skipgram_visual_gated
import logging
import tensorboard_logger as tb_logger


class word2vec:

  def __init__(self, exp_path, inputfile, vocabulary_size, embedding_dim, epoch_num, batch_size, windows_size, neg_sample_num, pretrained_embeddings, init_scheme, args, tokenize_text=False, rho_step=5000, val_step=5000, val_inputfile=None):
      self.exp_path = exp_path

      # Setup the data handler for the training dataset and save the vocabulary to disk
      self.data_handler = DataHandler(fname=inputfile, bs=batch_size, ws=windows_size, vocabulary_size=vocabulary_size, exp_path=self.exp_path, tokenize_text=tokenize_text)
      self.data_handler.save_vocab()

      # Setup the data handler for the validation data using the training vocabulary and counts
      self.val_data_handler = DataHandler(fname=val_inputfile, bs=batch_size, ws=windows_size, vocabulary_size=vocabulary_size, exp_path=self.exp_path, tokenize_text=tokenize_text, dictionary=self.data_handler.vocab_words, counts=self.data_handler.count)

      self.embedding_dim = embedding_dim
      self.windows_size = windows_size
      self.vocabulary_size = np.min([vocabulary_size, len(self.data_handler.vocab_words)])
      self.batch_size = batch_size
      self.epoch_num = epoch_num
      self.neg_sample_num = neg_sample_num

      if pretrained_embeddings != '':
          self.pretrained_embeddings = get_pretrained_embeddings(pretrained_embeddings, self.data_handler.vocab_words.items(), self.embedding_dim, self.vocabulary_size)
      else:
          self.pretrained_embeddings = None

      self.init_scheme = init_scheme
      self.rho_step = rho_step
      self.val_step = val_step
      self.checkpoint_step = args.checkpoint_step

  def train(self, lr):
      best_rho = 0.
      best_val_loss = 1e6
      model = skipgram(self.vocabulary_size, self.embedding_dim, self.pretrained_embeddings, self.init_scheme)
      if torch.cuda.is_available():
          model.cuda()
      optimizer = optim.Adagrad(model.parameters(),lr=lr)
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

              batch_num = batch_num + 1

              if batch_num % 1000 == 0:
                  end = time.time()
                  logging.info('Train: epoch %2d batch %5d: loss = %4.3f (%4.2f pair/s)'%(epoch, batch_num, loss.item(), (batch_num-batch_new)*self.batch_size/(end-start)))
                  tb_logger.log_value("loss", loss, step=batch_num)
                  batch_new = batch_num
                  start = time.time()

              if batch_num % self.val_step == 0:
                  val_loss, val_batch_num, val_end, val_start = validate(model, self.val_data_handler, self.neg_sample_num, self.batch_size)
                  tb_logger.log_value("val_loss", val_loss / val_batch_num, step=batch_num)
                  if val_loss < best_val_loss:
                      best_val_loss = val_loss
                      torch.save(model.state_dict(), os.path.join(self.exp_path, 'skipgram.epoch{}_batch{}-val_loss{}'.format(epoch, batch_num, best_val_loss)))
                      logging.info('Eval:  epoch %2d batch %5d: loss = %4.3f (%4.2f pair/s) (New best, saving.)' % (epoch, batch_num, val_loss, val_batch_num * self.batch_size / (val_end - val_start)))
                  else:
                      logging.info('Eval:  epoch %2d batch %5d: loss = %4.3f (%4.2f pair/s)' % (epoch, batch_num, val_loss, val_batch_num * self.batch_size / (val_end - val_start)))

              if batch_num % self.rho_step == 0:
                  end = time.time()
                  word_embeddings = model.input_embeddings()
                  sp1, sp2 = scorefunction(word_embeddings, self.exp_path)
                  tb_logger.log_value("ws353", sp1, step=batch_num)
                  tb_logger.log_value("rare", sp2, step=batch_num)
                  if sp1 > best_rho:
                      # WARNING: We probably should not be saving the model parameters based on the best correlation but loss on a held-out set.
                      best_rho = sp1
                  logging.info('Eval:  epoch %2d batch %5d: ws353 = %1.3f, rare = %1.3f'%(epoch, batch_num, sp1, sp2))
                  batch_new = batch_num
                  start = time.time()

              if batch_num % self.checkpoint_step == 0:
                  torch.save(model.state_dict(), os.path.join(self.exp_path, 'skipgram.epoch{}_batch{}'.format(epoch, batch_num)))
                  

      logging.info("Optimization Finished!")

class visword2vec:

  def __init__(self, exp_path, inputfile, imagefile, vocabulary_size, embedding_dim, img_dim, epoch_num, batch_size, windows_size, neg_sample_num, pretrained_embeddings, init_scheme, args, tokenize_text=False, rho_step=5000, val_step=5000, val_inputfile=None, val_imagefile=None):
      self.exp_path = exp_path

      # Setup the data handler for the training dataset and save the vocabulary to disk
      self.data_handler = DataHandler(fname=inputfile, iname=imagefile, bs=batch_size, ws=windows_size, vocabulary_size=vocabulary_size, exp_path=self.exp_path,  tokenize_text=tokenize_text)
      self.data_handler.save_vocab()

      # Setup the data handler for the validation data using the training vocabulary and counts
      self.val_data_handler = DataHandler(fname=val_inputfile, iname=val_imagefile, bs=batch_size, ws=windows_size, vocabulary_size=vocabulary_size, exp_path=self.exp_path, tokenize_text=tokenize_text, dictionary=self.data_handler.vocab_words, counts=self.data_handler.count)

      self.img_dim = img_dim
      self.embedding_dim = embedding_dim
      self.windows_size = windows_size
      self.vocabulary_size = np.min([vocabulary_size, len(self.data_handler.vocab_words)])
      self.batch_size = batch_size
      self.epoch_num = epoch_num
      self.neg_sample_num = neg_sample_num

      if pretrained_embeddings != '':
          self.pretrained_embeddings = get_pretrained_embeddings(pretrained_embeddings, self.data_handler.vocab_words.items(), self.embedding_dim, self.vocabulary_size)
      else:
          self.pretrained_embeddings = None

      self.init_scheme = init_scheme
      self.rho_step = rho_step
      self.val_step = val_step
      self.checkpoint_step = args.checkpoint_step

  def train(self, lr):
      best_rho = 0.
      best_val_loss = 1e6
      model = skipgram_visual_gated(self.vocabulary_size, self.embedding_dim, self.img_dim, self.pretrained_embeddings, self.init_scheme)
      if torch.cuda.is_available():
          model.cuda()
      optimizer = optim.Adagrad(model.parameters(),lr=lr)
      logging.info('Starting training iterations')
      for epoch in range(self.epoch_num):

          start = time.time()
          self.data_handler.process = True
          batch_num = 0
          batch_new = 0

          while self.data_handler.process:

              pos_u, pos_v, neg_v, visual = self.data_handler.generate_batch(self.neg_sample_num)

              pos_u = Variable(torch.LongTensor(pos_u))
              pos_v = Variable(torch.LongTensor(pos_v))
              neg_v = Variable(torch.LongTensor(neg_v))
              visual = Variable(torch.FloatTensor(visual))

              if torch.cuda.is_available():
                  pos_u = pos_u.cuda()
                  pos_v = pos_v.cuda()
                  neg_v = neg_v.cuda()
                  visual = visual.cuda()

              optimizer.zero_grad()
              loss = model(pos_u, pos_v, neg_v, visual, self.batch_size)
              loss.backward()
              optimizer.step()

              batch_num = batch_num + 1

              if batch_num % 1000 == 0:
                  end = time.time()
                  logging.info('Train: epoch %2d batch %5d: loss = %4.3f (%4.2f pair/s)'%(epoch, batch_num, loss.item(), (batch_num-batch_new)*self.batch_size/(end-start)))
                  tb_logger.log_value("loss", loss, step=batch_num)
                  batch_new = batch_num
                  start = time.time()

              if batch_num % self.val_step == 0:
                  val_loss, val_batch_num, val_end, val_start = validate(model, self.val_data_handler, self.neg_sample_num, self.batch_size, use_visual=True)
                  tb_logger.log_value("val_loss", val_loss, step=batch_num)
                  if val_loss < best_val_loss:
                      best_val_loss = val_loss
                      torch.save(model.state_dict(), os.path.join(self.exp_path, 'skipgram.epoch{}_batch{}-val_loss{}'.format(epoch, batch_num, best_val_loss)))
                      logging.info('Eval:  epoch %2d batch %5d: loss = %4.3f (%4.2f pair/s) (New best, saving.)' % (epoch, batch_num, val_loss, val_batch_num * self.batch_size / (val_end - val_start)))
                  else:
                      logging.info('Eval:  epoch %2d batch %5d: loss = %4.3f (%4.2f pair/s)' % (epoch, batch_num, val_loss, val_batch_num * self.batch_size / (val_end - val_start)))
                  start = time.time()

              if batch_num % self.rho_step == 0:
                  end = time.time()
                  word_embeddings = model.input_embeddings()
                  sp1, sp2 = scorefunction(word_embeddings, self.exp_path)
                  tb_logger.log_value("ws353", sp1, step=batch_num)
                  tb_logger.log_value("rare", sp2, step=batch_num)
                  if sp1 > best_rho:
                      best_rho = sp1
                  logging.info('Eval:  epoch %2d batch %5d: ws353 = %1.3f, rare = %1.3f'%(epoch, batch_num, sp1, sp2))
                  batch_new = batch_num
                  start = time.time()

              if batch_num % self.checkpoint_step == 0:
                  torch.save(model.state_dict(), os.path.join(self.exp_path, 'skipgram.epoch{}_batch{}'.format(epoch, batch_num)))

      logging.info("Optimization Finished!")


def create_exp_dir(exp_path):
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return


def get_pretrained_embeddings(fname, embedding_dim, vocabulary, size):
    '''
    loads the pretrained embeddingsrom file and sorts them such that they correspond to the indexes in the dataset
    :param fname:
    :return:
    '''
    logging.info('Loading pretrained embeddings from {}'.format(fname))
    embeds, word2id, id2word = load_embeddings_from_file(fname)

    sorted_embeds = np.zeros((size, embedding_dim))
    # initial embedding for tokens at are not contained in the pretrained embeddings
    initrange = 0.5 / embedding_dim
    unk_embedding = np.random.unifo(-initrange, initrange, embedding_dim)

    for idx, word in vocabulary.items():
        if word in word2id.keys():
            sorted_embeds[idx,:] = embeds[word2id[word],:]
        else:
            sorted_embeds[idx, :] = unk_embedding
            logging.info('No pretrained embedding for word {}'.format(word))
    return sorted_embeds


def validate(model, data_handler, neg_sample_num, batch_size, use_visual=False):
    '''
    TODO: the loss on the validation dataset does not really change. 
    Could be a bug in the data handler.
    '''
    start = time.time()
    total_loss = 0.
    batch_num = 0

    model.eval()
    with torch.no_grad():
        while data_handler.process:
            if use_visual:
                pos_u, pos_v, neg_v, visual = data_handler.generate_batch(neg_sample_num)

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))
                visual = Variable(torch.FloatTensor(visual))

                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()
                    visual = visual.cuda()

                loss = model(pos_u, pos_v, neg_v, visual, batch_size)
            else:
                pos_u, pos_v, neg_v = data_handler.generate_batch(neg_sample_num)

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))

                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                loss = model(pos_u, pos_v, neg_v, batch_size)

            total_loss += loss
            batch_num += 1

    data_handler.process = True
    end = time.time()
    model.train()

    return total_loss / float(batch_num), batch_num, end, start


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
    tb_logger.configure(os.path.join(exp_path, "tensorboard_log"), flush_secs=5)


def main(args):
    create_exp_dir(args.exp_path)
    set_up_logging(args.exp_path)

    print(args)

    if args.iname is None:
        wc = word2vec(exp_path=args.exp_path, inputfile=args.fname, vocabulary_size=args.vocab_size, embedding_dim=args.emb_dim,
                      epoch_num=args.epochs, batch_size=args.bs, windows_size=args.ws, neg_sample_num=args.neg, args=args, tokenize_text=args.tokenize_text,
                      pretrained_embeddings=args.pretrained_embeddings, init_scheme=args.init_scheme, rho_step=args.rho_step,
                      val_inputfile=args.val_fname, val_step=args.val_step)
    else:
        wc = visword2vec(exp_path=args.exp_path, inputfile=args.fname, imagefile=args.iname, vocabulary_size=args.vocab_size, embedding_dim=args.emb_dim, img_dim=args.img_dim,
                  epoch_num=args.epochs, batch_size=args.bs, windows_size=args.ws, neg_sample_num=args.neg, args=args, tokenize_text=args.tokenize_text,
                  pretrained_embeddings=args.pretrained_embeddings, init_scheme=args.init_scheme, rho_step=args.rho_step, val_inputfile=args.val_fname, val_imagefile=args.val_iname,
                  val_step=args.val_step)
    wc.train(lr=args.lr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Learning word embeddings via sgns')

    parser.add_argument('--fname', type=str, default='',
                            help="Text input file with one sentence per line")
    parser.add_argument("--iname", type=str, default=None,
                        help="File containing the image vectors")
    parser.add_argument('--val_fname', type=str, default='',
                            help="Text input file with one sentence per line")
    parser.add_argument("--val_iname", type=str, default='',
                        help="File containing the image vectors")
    parser.add_argument("--tokenize", action='store_true', dest="tokenize_text",
                        help="Tokenize the text using the SpaCy tokenizer?")
    parser.add_argument('--exp_path', type=str, default='tmp',
                        help="Vocabulary and embeddings are written to this directory")
    parser.add_argument('--vocab_size', type=int, default=100000,
                            help="Number of words in the vocabulary")
    parser.add_argument('--emb_dim', type=int, default=300,
                        help="Dimension of the learned embeddings")
    parser.add_argument('--img_dim', type=int, default=2048,
                        help="Dimension of the image vectors")
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
    parser.add_argument('--pretrained_embeddings', type=str, default='',
                        help="Path to pretrained embeddings used to initialize the embeddings of the input words. If not specified, embeddings are learned from scratch")
    parser.add_argument('--init_scheme', type=str, default='in', choices=['in', 'out', 'in_out'],
                        help="Specifies which embeddings are initialized using the pretrained embeddings. By default only the input embeddings.")
    parser.add_argument("--rho_step", type=int, default=5000, help="Evaluate correlation every rho_step updates")
    parser.add_argument("--val_step", type=int, default=100000, help="Evaluate language modelling performance on held out data every val_step updates")
    parser.add_argument("--checkpoint_step", type=int, default=200000, help="Checkpoint the model every checkpoint_step updates")

    args = parser.parse_args()
    main(args)
















