import torch
from torch import nn

class AdaptationRNNCell(nn.Module):
    def __init__(self, units, activation, alpha, beta):
        super().__init__()
        self.units = units
        self.state_size = (units, units, units)
        self.activation = activation
        self.alpha = alpha
        self.beta = beta

        # equivalent to tf.keras.layers.Dense with use_bias = False
        self.recurrent = nn.Linear(self.units, self.units, bias=False)
        self.input_layer = nn.Linear(self.units, self.units, bias=False)

        # Initialize with identity matrix
        nn.init.eye_(self.recurrent.weight)
        
        # regularization in PyTorch is usually added during the optimization step, not in the model itself.

    def forward(self, inputs, states):
        g_prev, u_prev, v_prev = states 
        u = -self.beta*v_prev + self.recurrent(g_prev) + self.input_layer(inputs)
        v = v_prev + self.alpha*(u_prev - v_prev)
        g = self.activation(u)
        return [g, v], [g, u, v] 

class AdaptationRNN(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.activation = torch.relu 
        self.init_layer = nn.Linear(options['num_nodes'], options['num_nodes'])
        self.rnn_cell = AdaptationRNNCell(options['num_nodes'], self.activation, options['alpha'], options['beta'])
        self.rnn_layer = nn.RNNCell(self.rnn_cell, options['num_nodes'], nonlinearity = self.activation)
        self.output_layer = nn.Linear(options['num_nodes'], options['num_pcs'])

    def forward(self, inputs):
        h_init = self.init_layer(inputs[1]) 
        u_init = self.activation(h_init)
        v_init = torch.zeros(h_init.shape) 
        g, adaptation = self.rnn_layer(inputs[0], (u_init, h_init, v_init))
        p = self.output_layer(g)
        return p, g, adaptation
