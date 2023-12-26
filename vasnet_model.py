

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *
from layer_norm import  *
from qlib import *
from Quaternion_ops import *
from Quaternion_layers import *

class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = QuaternionLinear(self.m, self.output_size,bias=False)
        self.Q = QuaternionLinear(self.m, self.output_size,bias=False)
        self.V = QuaternionLinear(self.m, self.output_size,bias=False)
        self.output_linear = QuaternionLinear(self.m, self.output_size,bias=False)

        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        
    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = hamilton_product(Q, K)

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")
    
        att_weights_ = q_normalize(logits,-1)
        weights = self.drop50(att_weights_)
        y = hamilton_product(att_weights_, V)
        y = self.output_linear(y)

        return y, att_weights_



class VASNet(nn.Module):

    def __init__(self):
        super(VASNet, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = 1024

        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = QuaternionLinear(self.m, out_features=1024)
        self.kb = QuaternionLinear(in_features=self.ka.out_features, out_features=1024)
        self.kc = QuaternionLinear(in_features=self.kb.out_features, out_features=1024)
        self.kd = QuaternionLinear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)


    def forward(self, x, seq_len):

        m = x.shape[2] # Feature size

        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        x = x.view(-1, m)
        y, att_weights_ = self.att(x)

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y, att_weights_



if __name__ == "__main__":
    pass