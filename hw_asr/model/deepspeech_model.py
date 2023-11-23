from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel
import torch
from torch import Tensor
from typing import Tuple

import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr
    
class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths
    
class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_
        
        
class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        #print(x.shape)
#         x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
#         x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x
    
class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
# class DeepSpeech(nn.Module):
#     def __init__(self, rnn_type, rnn_hidden_size, nb_layers,
#                  bidirectional, context=20):
#         super(DeepSpeech, self).__init__()

        nb_layers=5
        self.hidden_size = fc_hidden
        self.hidden_layers = 5
        self.rnn_type = nn.LSTM
        self.bidirectional = True

#         sample_rate = args.sample_rate
        num_classes = n_class

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        
        rnn_input_size = n_feats
        
        #print(rnn_input_size)

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=fc_hidden, rnn_type=self.rnn_type,
                       bidirectional=self.bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=fc_hidden, hidden_size=fc_hidden, rnn_type=self.rnn_type,
                           bidirectional=self.bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(fc_hidden, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        
        self.bn = nn.BatchNorm1d(fc_hidden)
        self.lin = nn.Linear(fc_hidden, num_classes, bias=False)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_hidden),
            nn.Linear(fc_hidden, num_classes, bias=False)
        )
        

    def forward(self, spectrogram, **batch):
#         lengths = lengths.cpu().int()
#         output_lengths = self.get_seq_lens(lengths)
#         x, _ = self.conv(x, output_lengths)

#         sizes = x.size()
#         x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
#         x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        
#         #print(x.shape)

        x = spectrogram.transpose(0,2)
        x = x.transpose(1,2)
        print(x.shape, self.rnns[0].rnn)
        for rnn in self.rnns:
            x = rnn(x)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)
        
        print(x.shape)
#         x=x.unsqueeze(3)
        x=x.transpose(0,1)
        x = x.transpose(1,2)
        print(x.shape)

#         print(nn.Linear(512, 28, bias=False).to('cuda')(x).shape)
        x = self.bn(x)
        x = x.transpose(1,2)
        x = self.lin(x)
        #x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        #x = self.inference_softmax(x)
        print(x.shape)
        return x#, output_lengths
    
    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here