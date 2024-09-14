from .implicit_function import ImplicitFunctionInf
from .implicit_model import ImplicitModel, ImplicitModelLoRA
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ImplicitRNNCell(nn.Module):
    def __init__(self, input_size, n, hidden_size, **kwargs):
        """
        Create a new ImplicitRNNCell:
            Mimics a vanilla RNN, but where the recurrent operation is performed by an implicit model instead of a linear layer.
            Follows the "batch first" convention, where the input x has shape (batch_size, seq_len, input_size).
            The hidden state doubles as the output, but it is not recommended to use it directly as the model's prediction
            for low-dimensionality problems, since it is responsible for passing information between timesteps. Instead, it is
            recommended to use a larger hidden state with a linear layer that scales it down to the desired output dimension.
        
        Args:
            input_size: the input dimension into the RNN cell.
            n: the number of hidden features in the underlying implicit model (equivalent to n in the ImplicitModel class).
            hidden_size: the recurrent hidden dimension of the ImplicitRNNCell. Equivalent to the hidden size in a GRU, for example.
            **kwargs: other keyword arguments passed to the undelying ImplicitModel.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layer = ImplicitModel(n, input_size+hidden_size, hidden_size, **kwargs)

    def forward(self, x):
        outputs = torch.empty(*x.shape[:-1], self.hidden_size, device=x.device)
        
        h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        for t in range(x.shape[1]):
            h = self.layer(torch.cat((x[:, t, :], h), dim=-1))
            outputs[:, t, :] = h
            
        return outputs, h
    
class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        """
        Create a new ImplicitRNNCell:
            Mimics a vanilla RNN, but where the recurrent operation is performed by an implicit model instead of a linear layer.
            Follows the "batch first" convention, where the input x has shape (batch_size, seq_len, input_size).
            The hidden state doubles as the output, but it is not recommended to use it directly as the model's prediction
            for low-dimensionality problems, since it is responsible for passing information between timesteps. Instead, it is
            recommended to use a larger hidden state with a linear layer that scales it down to the desired output dimension.
        
        Args:
            input_size: the input dimension into the RNN cell.
            n: the number of hidden features in the underlying implicit model (equivalent to n in the ImplicitModel class).
            hidden_size: the recurrent hidden dimension of the ImplicitRNNCell. Equivalent to the hidden size in a GRU, for example.
            **kwargs: other keyword arguments passed to the undelying ImplicitModel.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(
            nn.Linear(input_size+hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.ReLU()
            )

    def forward(self, x):
        outputs = torch.empty(*x.shape[:-1], self.hidden_size, device=x.device)
        
        h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        for t in range(x.shape[1]):
            h = self.layer(torch.cat((x[:, t, :], h), dim=-1))
            outputs[:, t, :] = h
            
        return outputs, h
    
class ImplicitRNNCellLoRA(nn.Module):
    def __init__(self, k, input_size, n, hidden_size, **kwargs):
        """
        Create a new ImplicitRNNCell:
            Mimics a vanilla RNN, but where the recurrent operation is performed by an implicit model instead of a linear layer.
            Follows the "batch first" convention, where the input x has shape (batch_size, seq_len, input_size).
            The hidden state doubles as the output, but it is not recommended to use it directly as the model's prediction
            for low-dimensionality problems, since it is responsible for passing information between timesteps. Instead, it is
            recommended to use a larger hidden state with a linear layer that scales it down to the desired output dimension.
        
        Args:
            input_size: the input dimension into the RNN cell.
            n: the number of hidden features in the underlying implicit model (equivalent to n in the ImplicitModel class).
            hidden_size: the recurrent hidden dimension of the ImplicitRNNCell. Equivalent to the hidden size in a GRU, for example.
            **kwargs: other keyword arguments passed to the undelying ImplicitModel.
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.layer = ImplicitModelLoRA(k, n, input_size+hidden_size, hidden_size, **kwargs)

    def forward(self, x):
        outputs = torch.empty(*x.shape[:-1], self.hidden_size, device=x.device)
        
        h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        for t in range(x.shape[1]):
            h = self.layer(torch.cat((x[:, t, :], h), dim=-1))
            outputs[:, t, :] = h
            
        return outputs, h
    

from torch.autograd import Variable

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, implicit=False):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size)
        #       cy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))
            hx = (hx, hx)

        hx, cx = hx

        gates = self.xh(input) + self.hh(hx)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)


        return (hy, cy)

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        hy = (self.x2h(input) + self.h2h(hx))

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size, activation='tanh'):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if activation == 'tanh':
            self.rnn_cell_list.append(RNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(RNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias,
                                                       "tanh"))

        elif activation == 'relu':
            self.rnn_cell_list.append(RNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(RNNCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.bias,
                                                   "relu"))
        else:
            raise ValueError("Invalid activation.")

        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()).detach()
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size)).detach()

        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()

        out = self.fc(out)


        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size, implicit=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(LSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias, implicit))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias, implicit))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()).detach()
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size)).detach()
        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()

        out = self.fc(out)

        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()

        out = self.fc(out)

        return out

#############

# class LSTMCell(nn.Module):

#     def __init__(self, input_dim, hidden_dim):
#         super(LSTMCell, self).__init__()
#         self.input_dim, self.hidden_dim = input_dim, hidden_dim
#         self.forget_input, self.forget_hidden, self.forget_bias = self.create_gate_parameters()
#         self.input_input, self.input_hidden, self.input_bias = self.create_gate_parameters()
#         self.output_input, self.output_hidden, self.output_bias = self.create_gate_parameters()
#         self.cell_input, self.cell_hidden, self.cell_bias = self.create_gate_parameters()

#     def create_gate_parameters(self):
#         input_weights = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
#         hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
#         nn.init.xavier_uniform_(input_weights)
#         nn.init.xavier_uniform_(hidden_weights)
#         bias = nn.Parameter(torch.zeros(self.hidden_dim))
#         return input_weights, hidden_weights, bias

#     def forward(self, x, h, c):
#         # x has shape [batch_size, seq_len, input_size]
#         output_hiddens, output_cells = [], []
#         for i in range(x.shape[1]):
#             forget_gate = F.sigmoid((x[:, i] @ self.forget_input) + (h @ self.forget_hidden) + self.forget_bias)
#             input_gate = F.sigmoid((x[:, i] @ self.input_input) + (h @ self.input_hidden) + self.input_bias)
#             output_gate = F.sigmoid((x[:, i] @ self.output_input) + (h @ self.output_hidden) + self.output_bias)
#             input_activations = F.tanh((x[:, i] @ self.cell_input) + (h @ self.cell_hidden) + self.cell_bias)
#             c = (forget_gate * c) + (input_gate * input_activations)
#             h = F.tanh(c) * output_gate
#             output_hiddens.append(h.unsqueeze(1))
#             output_cells.append(c.unsqueeze(1))
#         return torch.concat(output_hiddens, dim=1), torch.concat(output_cells, dim=1)



# class MultiLayerLSTM(nn.Module):

#     def __init__(self, input_dim, hidden_dim, num_layers, dropout):
#         super(MultiLayerLSTM, self).__init__()
#         self.input_dim, self.hidden_dim, self.num_layers = input_dim, hidden_dim, num_layers
#         self.layers = nn.ModuleList()
#         self.layers.append(LSTMCell(input_dim, hidden_dim))
#         for _ in range(num_layers - 1):
#             self.layers.append(LSTMCell(hidden_dim, hidden_dim))
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(hidden_dim, input_dim)
#         nn.init.xavier_uniform_(self.linear.weight.data)
#         self.linear.bias.data.fill_(0.0)

#     def forward(self, x):
#         # x has shape [batch_size, seq_len, embed_dim]
#         # h is a tuple containing h and c, each have shape [layer_num, batch_size, hidden_dim]
#         hidden, cell = (torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)), torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)))
#         output_hidden, output_cell = self.layers[0](x, hidden[0], cell[0])
#         new_hidden, new_cell = [output_hidden[:, -1].unsqueeze(0)], [output_cell[:, -1].unsqueeze(0)]
#         for i in range(1, self.num_layers):
#             output_hidden, output_cell = self.layers[i](self.dropout(output_hidden), hidden[i], cell[i])
#             new_hidden.append(output_hidden[:, -1].unsqueeze(0))
#             new_cell.append(output_cell[:, -1].unsqueeze(0))
#         return self.linear(self.dropout(output_hidden))[:, -1, :]