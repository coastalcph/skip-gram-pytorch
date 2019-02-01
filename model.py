import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inputdata import Options, scorefunction

class skipgram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(skipgram, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
    self.embedding_dim = embedding_dim
    self.init_emb()
  def init_emb(self):
    initrange = 0.5 / self.embedding_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.v_embeddings.weight.data.uniform_(-0, 0)
  def forward(self, u_pos, v_pos, v_neg, batch_size):

    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)

    score  = torch.mul(embed_u, embed_v)
    score = torch.sum(score, dim=1)
    log_target = F.logsigmoid(score).squeeze()
    
    neg_embed_v = self.v_embeddings(v_neg)
    
    neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

    loss = log_target + sum_log_sampled

    return -1*loss.sum()/batch_size

  def input_embeddings(self):
    return self.u_embeddings.weight.data.cpu().numpy()

  def save_embedding(self, file_name, id2word):
    embeds = self.u_embeddings.weight.data
    fo = open(file_name, 'w')
    for idx in range(embeds.shape[0]):
      word = id2word[idx]
      embed = ' '.join('%.5f' % x for x in embeds[idx,:])
      fo.write(word+' '+embed+'\n')


class skipgram_visual_gated(nn.Module):

  def __init__(self, vocab_size, embedding_dim, img_dim ):
    super(skipgram_visual_gated, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)

    # the gate parameters W_g to compute sigmoid(W_g*input_embedding) comp_mul output_embedding
    self.gate_params = nn.Linear(embedding_dim, img_dim, bias=True)

    self.embedding_dim = embedding_dim
    self.img_dim = img_dim
    self.init_emb()
    self.init_gate()

  def init_emb(self):
    initrange = 0.5 / self.embedding_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.v_embeddings.weight.data.uniform_(-0, 0)

  def init_gate(self):
    self.gate_params.weight.data.uniform_(-0, 0)

  def forward(self, u_pos, v_pos, v_neg, visual_pos, batch_size):
    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)

    visual_data = self.visual_data(visual_pos)
    # gate the visual information as sigmoid(W_gate * visual_data) componentwise embed_u
    gated_visual_information = F.sigmoid(self.gate_params(visual_data))

    score = torch.mul(embed_u, embed_v)
    score = torch.sum(score, dim=1)
    log_target = F.logsigmoid(score).squeeze()

    neg_embed_v = self.v_embeddings(v_neg)

    neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()

    loss = log_target + sum_log_sampled

    return -1 * loss.sum() / batch_size

  def input_embeddings(self):
    return self.u_embeddings.weight.data.cpu().numpy()

  def save_embedding(self, file_name, id2word):
    embeds = self.u_embeddings.weight.data
    fo = open(file_name, 'w')
    for idx in range(len(embeds)):
      word = id2word(idx)
      embed = ' '.join(embeds[idx])
      fo.write(word + ' ' + embed + '\n')
