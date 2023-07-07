import numpy as np

import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import init
from torch import _VF



class AdaptationRNN(torch.nn.RNN):
    def __init__(self, alpha=0.0, beta=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = alpha
        self.alpha = beta

    def forward(self, input, hx=None):
        batch_sizes = None
        unsorted_indices = None

        self.check_forward_args(input, hx, batch_sizes)

        W, Wh = self._flat_weights

        hidden = hx[0]
        result = []

        time_steps = input.size(1)

        # Defining v 
        v = torch.zeros_like(hidden)
        z_prev = torch.zeros_like(hidden) # This is just temp :)

        for i in range(time_steps):
            # apply W and Wh to all batches at once
            z = torch.matmul(W, input[:, i].T).T
            z += torch.matmul(Wh, hidden.T).T
            z -= self.beta * v

            # Defining v
            v = v + self.alpha * (z_prev - v)

            # Activation function
            s_z = torch.relu(z)

            # s_z is now [200, 4096], unsqueeze(1) gives us [200, 1, 4096]
            # so that we can concatenate the timesteps later
            result.append(s_z.unsqueeze(1))
            hidden = s_z
            z_prev = z

        # Concatenating the timesteps
        output = torch.cat(result, dim=1)
        hidden = hidden.unsqueeze(0) 

        return output, self.permute_hidden(hidden, unsorted_indices)

class SorscherRNN(torch.nn.Module):
    """
    Model based on:
    https://github.com/ganguli-lab/grid-pattern-formation/blob/master/model.py
    """

    def __init__(self, alpha=0.0, beta=0.0, weight_decay=0.0, Ng=4096, Np=512, **kwargs):
        super(SorscherRNN, self).__init__(**kwargs)
        self.Ng, self.Np = Ng, Np
        #Set torch seed
        torch.manual_seed(0)

        self.weight_decay = weight_decay

        # define network architecture
        self.init_position_encoder = torch.nn.Linear(Np, Ng, bias=False)
        self.RNN = AdaptationRNN(
            alpha=alpha,
            beta=beta,
            input_size=2,
            hidden_size=Ng,
            num_layers=1,
            nonlinearity="relu",
            bias=False,
            batch_first=True,
        )
        # Linear read-out weights
        self.decoder = torch.nn.Linear(Ng, Np, bias=False)

        # initialise model weights
        for param in self.parameters():
            torch.nn.init.xavier_uniform_(param.data, gain=1.0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def g(self, v, p0):
        """
        Parameters:
            v (mini-batch(optional), seq_len, 2): Cartesian velocities
            p0 (mini-batch(optional), 2): Initial cartesian position
        Returns:
            RNN activities (mini-batch(optional), seq_len, self.Ng)
        """
        if len(v.shape) == 2 and len(p0.shape) == 1:
            # model requires tensor of degree 3 (B,S,N)
            # assume inputs missing empty batch-dim
            v, p0 = v[None], p0[None]

        if self.device != v.device:
            # put v and p0 on same device as model
            v = v.to(self.device, dtype=self.dtype)
            p0 = p0.to(self.device, dtype=self.dtype)

        p0 = self.init_position_encoder(p0)
        p0 = p0[None]  # add dummy (unit) dim for number of stacked rnns (D)
        # output of torch.RNN is a 2d-tuple. First element =>
        # return_sequences=True (in tensorflow). Last element => False.
        out, _ = self.RNN(v, p0)
        return out

    def p(self, g_inputs, log_softmax=False):
        place_preds = self.decoder(g_inputs)
        return (
            torch.nn.functional.log_softmax(place_preds, dim=-1)
            if log_softmax
            else place_preds
        )

    def forward(self, v, p0, log_softmax=False):
        gs = self.g(v, p0)
        return self.p(gs, log_softmax)

    def loss_fn(self, log_predictions, labels):
        """
        Parameters:
            log_predictions, (mini-batch, seq_len, npcs): model predictions in log softmax
            labels, (mini-batch, seq_len, npcs): true place cell activities
            weight_decay, int: regularization scaling parameter
        Returns:
            loss, float: The loss
        """
        if labels.device != self.device:
            labels = labels.to(self.device, dtype=self.dtype)
        CE = torch.mean(-torch.sum(labels * log_predictions, axis=-1))
        l2_reg = torch.sum(self.RNN.weight_hh_l0**2)
        return CE + self.weight_decay*l2_reg 

    def train_step(self, v, p0, labels):
        self.optimizer.zero_grad()
        log_predictions = self.forward(v, p0, log_softmax=True)
        loss = self.loss_fn(log_predictions, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()


