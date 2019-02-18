import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inputdata import Options, scorefunction

class skipgram(nn.Module):
  def __init__(self, vocab_size, embedding_dim, pretrained_embeddings, init_scheme):
    super(skipgram, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
    self.embedding_dim = embedding_dim
    self.init_emb(pretrained_embeddings, init_scheme)

  def init_emb(self, pretrained_embeddings, init_scheme):
    if pretrained_embeddings is not None and init_scheme =="in":
      self.u_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
      self.v_embeddings.weight.data.uniform_(-0, 0)
    elif pretrained_embeddings is not None and init_scheme == "out":
      self.u_embeddings.weight.data.normal_(0, 0.1)
      self.v_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
    elif pretrained_embeddings is not None and init_scheme =="in_out":
      self.u_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
      self.v_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
    else:
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

  def __init__(self, vocab_size, embedding_dim, img_dim, pretrained_embeddings, init_scheme):
    """

    :param vocab_size: number of words in the vocabulary
    :param num_imgs: number of images
    :param embedding_dim: dimension of the learned embeddings
    :param img_dim: dimension of the image representations
    :param visual_data: array of size num_imgs x img_dim
    """
    super(skipgram_visual_gated, self).__init__()
    self.embedding_dim = embedding_dim
    self.img_dim = img_dim

    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    # the gate parameters W_g to compute sigmoid(input_embedding*W_gate) comp_mul visual_data
    self.gate_params = nn.Linear(embedding_dim, embedding_dim, bias=True)
    # parameters to reduce dimensionality of the visual data as activation(img_dim, emb_dim)
    self.img_embedding = nn.Linear(img_dim, embedding_dim, bias=True)

    self.init_emb(pretrained_embeddings, init_scheme)
    self.init_gate()

  def init_emb(self, pretrained_embeddings, init_scheme):
    if pretrained_embeddings is not None and init_scheme == "in":
      self.u_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
      self.v_embeddings.weight.data.uniform_(-0, 0)
    elif pretrained_embeddings is not None and init_scheme == "out":
      self.u_embeddings.weight.data.normal_(0, 0.1)
      self.v_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
    elif pretrained_embeddings is not None and init_scheme == "in_out":
      self.u_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
      self.v_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
    else:
      initrange = 0.5 / self.embedding_dim
      self.u_embeddings.weight.data.uniform_(-initrange, initrange)
      self.v_embeddings.weight.data.uniform_(-0, 0)
    self.img_embedding.weight.data.normal_(0, 0.1)



  def init_gate(self):
    self.gate_params.weight.data.normal_(0, 0.1)
    nn.init.constant_(self.gate_params.bias, 0)

  def forward_(self, u_pos, v_pos, v_neg, visual_pos, batch_size):
    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)
    visual_embed = self.img_embedding(visual_pos)
    visual_embed = self.l2norm(visual_embed)

    # gate the visual information as sigmoid(W_gate * embed_v) componentwise visual
    gate = torch.sigmoid(self.gate_params(embed_u))
    gated_visual_embed = gate * visual_embed
    joint_embed = torch.add(embed_u, gated_visual_embed)
    score = torch.mul(joint_embed, embed_v)
    score = torch.sum(score, dim=1)
    log_target = F.logsigmoid(score).squeeze()

    neg_embed_v = self.v_embeddings(v_neg)

    neg_score = torch.bmm(neg_embed_v, joint_embed.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()

    loss = log_target + sum_log_sampled

    return embed_u, gate, visual_embed

  def forward(self, u_pos, v_pos, v_neg, visual_pos, batch_size):
    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)
    visual_embed = self.img_embedding(visual_pos)
    visual_embed = self.l2norm(visual_embed)

    # gate the visual information as sigmoid(W_gate * embed_v) componentwise visual
    gate = torch.sigmoid(self.gate_params(embed_u))
    gated_visual_embed = gate * visual_embed
    joint_embed = torch.add(embed_u, gated_visual_embed)
    score = torch.mul(joint_embed, embed_v)
    score = torch.sum(score, dim=1)
    log_target = F.logsigmoid(score).squeeze()

    neg_embed_v = self.v_embeddings(v_neg)

    neg_score = torch.bmm(neg_embed_v, joint_embed.unsqueeze(2)).squeeze()
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

  def l2norm(self, X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


if __name__=="__main__":
    u_pos = np.array([0, 1, 2])
    v_pos = np.array([0, 1, 2])
    v_neg = np.random.choice([0, 1, 2, 3, 4, 5], size=(3, 3))
    visual_pos = np.arange(20.0)

    u_pos = Variable(torch.LongTensor(u_pos))
    v_pos = Variable(torch.LongTensor(v_pos))
    v_neg = Variable(torch.LongTensor(v_neg))
    visual_pos = Variable(torch.FloatTensor(visual_pos))

    svg = skipgram_visual_gated(vocab_size=100, embedding_dim=10, img_dim=20)

    svg.forward(u_pos=u_pos, v_pos=v_pos, v_neg=v_neg, visual_pos=visual_pos, batch_size=5)
