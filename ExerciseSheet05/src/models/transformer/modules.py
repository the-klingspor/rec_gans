#!/usr/bin/env python3

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

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

	def __init__(self, n_heads, d_model, dropout=0.1):
		"""
		Constructor method of the attention module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()

		self.n_heads = n_heads
		self.d_model = d_model
		self.scaling = d_model ** -0.5

		# set up the layers for the multi-head-attention module here
		self.to_K = nn.Linear(d_model, d_model * n_heads, bias=False)
		self.to_V = nn.Linear(d_model, d_model * n_heads, bias=False)
		self.to_Q = nn.Linear(d_model, d_model * n_heads, bias=False)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, mask=None):
		"""
		Forward pass of the multi head attention module.
		:param k: Key vector
		:param v: Value vector
		:param q: Query vector
		:param mask: Mask to hide future entries
		:return: The attention weighted output as linear combination of v
		"""

		# define the forward pass of the multi-head-attention here

		# Get multi-headed Q, K, V based on input,
		# 	[batch, tokens, d_model * n_heads]
		k = self.to_K(x)
		v = self.to_V(x)
		q = self.to_Q(x)

		# Rearrange to separate Q, K, V by heads:
		# 	from
		# 		[batch, tokens, d_model * n_heads]
		# 	to  [batch, n_heads, tokens, d_model]
		k = rearrange(k, 'b t (d h) -> b h t d', d=self.d_model, n=self.n_heads)
		v = rearrange(v, 'b t (d h) -> b h t d', d=self.d_model, n=self.n_heads)
		q = rearrange(q, 'b t (d h) -> b h t d', d=self.d_model, n=self.n_heads)

		# Apply call to self-attention
		x = self.attention(q, k, v, mask)

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

		# compute the attention scores, apply the mask and perform the
		# attention multiplication here. Remember the scaling factor!

		# compute the raw attention values using queries and keys
		# 	from
		# 	   [batch, n_heads, tokens, n_dim] x [batch, n_heads, tokens, n_dim]
		# 	to [batch, n_heads, tokens, tokens]
		attention_raw = th.einsum('b h t1 d, b h t2 d -> b h t1 t2', q, k)
		attention_raw *= self.scaling

		# apply mask, if available
		if mask is not None:
			attention_raw = attention_raw.masked_fill(mask, -np.inf)  # todo warum?

		# get attention scores by normalizing with softmax on last dimension
		attention = th.softmax(attention_raw, dim=-1)

		# apply attention to the values v
		v = th.einsum('b h t1 t2, b h t2 d -> b h t1 d', attention, v)

		# stack heads back together
		# from	[batch, n_heads, tokens, n_dim]
		# to 	[batch, tokens, n_dim * n_heads
		v = rearrange(v, 'b h t d -> b t (h d)')

		# apply dropout, not sure if we are supposed to do this here, but
		# dropout is part of the classes' constructor
		v = self.dropout(v)

		# todo layer normalization

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
		# pretty sure this is not needed, because we already use dropout in the
		# feedforward layer and after applying attention to the values
		self.dropout = nn.Dropout(dropout)

		self.n_heads = n_heads
		self.d_model = d_model
		self.linear_layer_size = linear_layer_size

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

		# add residual connection + dropout ?
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
		self.input = nn.Linear(d_one_hot, d_model)
		self.decoders = [DecoderLayer(n_heads, d_model, linear_layer_size, dropout) for _ in num_blocks]
		self.output = nn.Linear(d_model, d_one_hot)

	def forward(self, x):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""
		# implement the forward pass of the model here
		x = self.input(x)
		for decoder in self.decoders:
			x = decoder(x)
		x = self.output(x)

		return x
		
	def _mask(self, x):
		"""
		Helper function to compute the mask applied to the decoder layer
		:param x: The input data that should be masked
		"""
		device, seq_len = x.device, x.shape[0]    

		# TODO: implement the mask for the decoder here

		return mask
