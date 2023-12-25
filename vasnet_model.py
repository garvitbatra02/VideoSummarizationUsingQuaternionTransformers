

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
        


    def compute_attention_component(self, antecedent, total_depth, filter_width=1, padding="VALID", name="c", vars_3d_num_heads=0):
        """
        Computes attention component (query, key, or value).

        Args:
            antecedent: A torch.Tensor with shape [batch, length, channels]
            total_depth: An integer
            filter_width: An integer specifying the desired attention component width.
            padding: One of "VALID", "SAME", or "LEFT". Default is "VALID".
            name: A string specifying the scope name.
            vars_3d_num_heads: An optional integer for using 3D variables.

        Returns:
            c: A torch.Tensor with shape [batch, length, depth]
        """

        input_depth = antecedent.shape[-1]
        initializer_stddev = 1.0 / input_depth**0.5

        if "q" in name:
            depth_per_head = total_depth
            initializer_stddev *= depth_per_head**-0.5

        if vars_3d_num_heads > 0:
            assert filter_width == 1
            depth_per_head = total_depth // vars_3d_num_heads
            var = nn.Parameter(torch.randn(input_depth, vars_3d_num_heads, depth_per_head) * initializer_stddev)
            c = torch.tensordot(antecedent, var, dims=([2], [0]))
            return c.view(antecedent.shape[0], antecedent.shape[1], total_depth)

        if filter_width == 1:
            # Assuming quarternion_ffn_3d is defined
            c = quarternion_ffn_3d(antecedent, total_depth, name=name, init=initializer_stddev)
        else:
            conv1d = nn.Conv1d(input_depth, total_depth, filter_width, padding=padding)
            c = conv1d(antecedent.transpose(1, 2)).transpose(1, 2)
        return c

    def quaternion_dot_product_attention(self,q,
                                        k,
                                        v,
                                        bias,
                                        dropout_rate=0.0,
                                        image_shapes=None,
                                        name=None,
                                        make_image_summary=True,
                                        save_weights_to=None,
                                        dropout_broadcast_dims=None):
        """Dot-product attention.
        Args:
        q: Tensor with shape [..., length_q, depth_k].
        k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
        match with q.
        v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
        match with q.
        bias: bias Tensor (see attention_bias())
        dropout_rate: a float.
        image_shapes: optional tuple of integer scalars.
        see comments for attention_image_summary()
        name: an optional string
        make_image_summary: True if you want an image summary.
        save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
        dropout_broadcast_dims: an optional list of integers less than rank of q.
        Specifies in which dimensions to broadcast the dropout decisions.
        Returns:
        Tensor with shape [..., length_q, depth_v].
        """
        v_vals = torch.split(v, 4, dim=-1)
        output = []
        
        # Note: PyTorch doesn't have the equivalent of tf.variable_scope, so we skip it.
        # Use plain variable names and avoid tf.variable_scope's default_name behavior.

        all_logits = quarternion_attention(q, k)
        for i, logits in enumerate(all_logits):
            if bias is not None:
                bias = bias.to(logits.dtype)
                logits += bias
            weights = F.softmax(logits, dim=-1)
            
            # Drop out attention links for each head.
            weights = F.dropout(weights, p=dropout_rate, training=True)
            
            o = torch.matmul(weights, v_vals[i])
            output.append(o)
        
        output = torch.cat(output, dim=-1)
        return output,weights


    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = hamilton_product(Q, K)

        # if self.ignore_itself:
        #     # Zero the diagonal activations (a distance of each frame with itself)
        #     logits[torch.eye(n).byte()] = -float("Inf")

        # if self.apperture > 0:
        #     # Set attention to zero to frames further than +/- apperture from the current one
        #     onesmask = torch.ones(n, n)
        #     trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
        #     logits[trimask == 1] = -float("Inf")
    
        att_weights_ = q_normalize(logits,-1)
        # weights = self.drop50(att_weights_)
        y = hamilton_product(att_weights_, V)
        y = self.output_linear(y)

        return y, att_weights_



class VASNet(nn.Module):

    def __init__(self):
        super(VASNet, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = 1024

        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

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