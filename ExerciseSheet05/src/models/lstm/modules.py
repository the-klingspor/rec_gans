#!/usr/bin/env python3

import torch as th
import torch.nn as nn

__author__ = "Matthias Karlbauer"


###########
# CLASSES #
###########

class Model(nn.Module):
	"""
	The actual model consisting of some feed forward and recurrent layers.
	"""

	def __init__(self, d_one_hot, d_lstm, num_lstm_layers, dropout=0.1,
				 bias=True):
		"""
		Constructor method of the Model module.
		:param d_one_hot: The size of the input and output vector
		:param d_lstm: The hidden size of the lstm layers
		:param num_lstm_layers: The number of sequential lstm layers
		:param dropout: Probability of dropping out certain neurons
		:param bias: Whether to use bias neurons
		"""
		super().__init__()

		self.input = nn.Linear(d_one_hot, d_lstm, bias=bias)
		self.lstm = nn.LSTM(num_layers=num_lstm_layers, input_size=d_lstm,
							hidden_size=d_lstm, bias=bias, dropout=dropout)
		self.output = nn.Linear(d_lstm, d_one_hot)

	def forward(self, x, state=None):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""

		if state:
			h0, c0 = state

		x = self.input(x)
		if state:
			x, (h, c) = self.lstm(x, (h0.detach(), c0.detach()))
		else:
			x, (h, c) = self.lstm(x)
		x = self.output(x)

		return x, (h, c)
