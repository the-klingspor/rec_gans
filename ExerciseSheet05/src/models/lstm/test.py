#!/usr/bin/env python3

"""
Main file for testing (evaluating) a model
"""

import numpy as np
import torch as th
import torch.nn as nn
import glob
import os
import sys
import time
import matplotlib.pyplot as plt
from threading import Thread

sys.path.append("../../utils")
from modules import Model
from configuration import Configuration
import helper_functions as helpers

_author_ = "Matthias Karlbauer"


def run_testing():

    # Load the user configurations
    cfg = Configuration("config.json")

    # Print some information to console
    print("Model name:", cfg.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config file
    if cfg.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set device on GPU if specified in the configuration file, else CPU
    device = helpers.determine_device()
    
    # Initialize and set up the model
    model = Model(
        d_one_hot=cfg.model.d_one_hot,
        d_lstm=cfg.model.d_lstm,
        num_lstm_layers=cfg.model.num_lstm_layers
    ).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                              "checkpoints",
                                               cfg.model.name,
                                               cfg.model.name+".pt"),
                                  map_location=device))
    model.eval()

    # Initialize the criterion (loss)
    criterion = nn.CrossEntropyLoss()

    #
    # Set up the dataloader for the testing
    dataset, dataloader = helpers.build_dataloader(
        cfg=cfg, batch_size=1
    )

    """
    TESTING
    """

    tf_steps = cfg.testing.teacher_forcing_steps
    cl_steps = cfg.testing.closed_loop_steps

    # Iterate over the batches
    for batch_idx, (net_input, net_label) in enumerate(dataloader):

        # Move data to the desired device and convert from
        # [batch_size, time, dim] to [time, batch_size, dim]
        net_input = net_input.to(device=device).transpose(0, 1)
        net_label = net_label.to(device=device).transpose(0, 1)

        # Start with the first teacher forcing characters and let the model
        # continue
        x = net_input[:tf_steps]
        y_hat, (h, c) = model(x=x)
        y_hat = nn.functional.softmax(y_hat, dim=-1)
        y_hat_last = th.tensor(
            [[helpers.softmax_to_one_hot(soft=y_hat[-1, 0])]]
        ).to(device=device)

        # Append the model output to the input and continue
        x = th.cat((x, y_hat_last), dim=0)

        for t in range(cl_steps):

            # Generate a prediction and apply the softmax
            y_hat, (h, c) = model(x=y_hat_last.float(), state=(h, c))
            y_hat = nn.functional.softmax(y_hat, dim=-1)

            # Convert the last softmax output to a one-hot vector
            y_hat_last = th.tensor(
                [[helpers.softmax_to_one_hot(soft=y_hat[-1, 0])]]
            ).to(device=device)

            # Append the model output to the input and continue
            x = th.cat((x, y_hat_last), dim=0)

        net_input = net_input.detach().cpu().numpy()
        tolkien_text = []

        for t in range(len(net_input[:tf_steps + cl_steps])):
            y_t = helpers.one_hot_to_char(
                one_hot_vector=net_input[t, 0],
                alphabet=dataset.alphabet
            )
            tolkien_text.append(y_t[0])

        model_text = []
        for t in range(len(x)):
            x_t = helpers.one_hot_to_char(
                one_hot_vector=helpers.softmax_to_one_hot(soft=x[t, 0]),
                alphabet=dataset.alphabet
            )
            model_text.append(x_t[0])

        tolkien_text = "".join(tolkien_text)
        model_text = "".join(model_text)

        print(f"Initialization: {tolkien_text[:tf_steps]}...")
        print(f"\nTolkien:\n --- {tolkien_text}")
        print(f"\nLSTM model:\n --- {model_text}")

        exit()


if __name__ == "__main__":
    th.set_num_threads(1)
    run_testing()

    print("Done.")