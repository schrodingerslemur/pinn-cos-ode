import torch
import torch.nn as nn

from jacrev_finite import JacrevFinite
from torch.autograd import grad

class Network(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        fc1 = nn.Linear(n_input, n_hidden)
        fc2 = nn.Linear(n_hidden, n_hidden)
        fc3 = nn.Linear(n_hidden, n_hidden)
        fc4 = nn.Linear(n_hidden, n_hidden)
        fc5 = nn.Linear(n_hidden, n_output)

        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4,fc5])

    def forward(self, x):
        # layer_norm = nn.LayerNorm(x.shape)
        # x = layer_norm(x)
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = torch.relu(x)
        return x 

def Derive_pred(input, net):
    output = net(input)
    d_input = grad(output.sum(), input, create_graph=True)[0]
    return d_input

def Derive_actual(input):
    d_input = torch.cos(input)
    return d_input

def IC_pred(net):
    input = torch.zeros(1)
    output = net(input)
    return output

def Derive_pred_jac(input, net):
    jacobian = JacrevFinite(network=net, num_args=0)(input)
    print(jacobian.shape)

        
