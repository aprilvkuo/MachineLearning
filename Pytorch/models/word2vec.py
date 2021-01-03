import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(object):
    def __init__(self):
        self.__emb = nn.Embedding(emb_size, emb_dimension, sparse=True)
    