#!/usr/bin/env python3

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn, einsum

__author__ = "Matthias Karlbauer, Jannik Thümmel"


class FeedForward(nn.Module):
	"""
	A specific feed forward module that consists of a relu layer followed by a
	dropout and a linear layer.
	"""

	def __init__(self, d_model, linear_layer_size, dropout=0.1):
		"""
		Constructor method of the feed forward module.
		:param linear_layer_size: The internal size of the feed forward module
		:param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()

		self.input = nn.Linear(d_model, linear_layer_size)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.output = nn.Linear(linear_layer_size, d_model)

	def forward(self, x):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""

		x = self.input(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.output(x)

		return x


class MultiHeadSelfAttention(nn.Module):
	"""
	The core component of the transformer realizing the attention mechanism.
	"""

	def __init__(self, n_heads, d_model, dropout = 0.1):
		"""
		Constructor method of the attention module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()

		# set up the layers for the multi-head-attention module here
		# n_k, n_v, n_q, n_out = d_model ? todo: change to Linear
		self.K = th.empty((n_heads, d_model, d_model), requires_grad=True)
		self.V = th.empty((n_heads, d_model, d_model), requires_grad=True)
		self.Q = th.empty((n_heads, d_model, d_model), requires_grad=True)

		self.dropout = nn.Dropout(dropout)
		self.output = th.empty((n_heads * d_model), requires_grad=True)

	def forward(self, x, mask=None):
		"""
		Forward pass of the multi head attention module.
		:param k: Key vector
		:param v: Value vector
		:param q: Query vector
		:param mask: Mask to hide future entries
		:return: The attention weighted output as linear combination of v
		"""

		# TODO: define the forward pass of the multi-head-attention here

		return x

	def attention(self, q, k, v, mask=None):
		"""
		The attention mechanism computing a weighted linear combination of v
		based on the similarity of the according k and v entries.
		:param k: Key vector
		:param v: Value vector
		:param q: Query vector
		:param mask: Mask to hide future entries
		:return: Weighted linear combination v_hat and the attention weights
		"""

		# TODO: compute the attention scores, apply the mask and perform the
		#		attention multiplication here. Remember the scaling factor!

		return v


class DecoderLayer(nn.Module):
	"""
	A decoder layer part (of a Transformer) which predicts next observations
	based on the previous inputs.
	"""
	
	def __init__(self, n_heads, d_model, linear_layer_size, dropout=0.1):
		"""
		Constructor method of the attention module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param linear_layer_size: The internal size of the feed forward module
		:param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()

		# Define the layers multi-head-attention, feed-forward, dropout
		# and normalization layers here.
		# Note that we do not use an Encoder
		# and therefore do not require the Encoder-Decoder Attention module!
		self.attention = MultiHeadSelfAttention(n_heads, d_model, dropout)
		self.linear = FeedForward(d_model, linear_layer_size, dropout)

	def forward(self, x, mask):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:param mask: Mask to hide future entries
		:return: The module's output
		"""
		
		# Define the forward pass. Keep in mind to produce residuals
		# instead of the absolute values directly.
		residual = x
		x = self.attention(x, mask)
		x = self.linear(x)

		x = x + residual

		return x


class Model(nn.Module):
	"""
	The actual model consisting of a selected number of sequential decoder
	layers.
	"""

	def __init__(self, n_heads, d_model, linear_layer_size, d_one_hot, num_blocks,
				 dropout=0.1):
		"""
		Constructor method of the Model module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param linear_layer_size: The internal size of the feed forward module
		:param d_one_hot: The size of the input and output vector
		:param num_blocks: How many Transformer blocks to stack.
		:param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()

		# define linear and decoder layers for the overall model
		self.decoders = [DecoderLayer(n_heads, d_model, linear_layer_size, dropout) for _ in num_blocks]
		self.ff_layers = [FeedForward(d_model, linear_layer_size, dropout) for _ in num_blocks]

	def forward(self, x):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""
		# TODO: implement the forward pass of the model here

		return x
		
	def _mask(self, x):
		"""
		Helper function to compute the mask applied to the decoder layer
		:param x: The input data that should be masked
		"""
		device, seq_len = x.device, x.shape[0]    

		# TODO: implement the mask for the decoder here

		return mask
