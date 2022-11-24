import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, nlayers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        output = self.transformer_encoder(src)
        output = F.leaky_relu(self.decoder(output))
        return output


"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, d_model, ntoken, action_dim, nhead, nlayers, dropout, state_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.d_model = d_model
        self.ntoken = ntoken
        self.fc_0 = nn.Linear(state_dim, self.d_model)
        self.norm_0 = nn.LayerNorm([self.d_model])
        self.transformer = TransformerModel(self.ntoken, self.d_model, nhead, hidden_dim, nlayers, dropout)

        self.fc_1 = nn.Linear(self.ntoken, action_dim)
        self.norm_1 = nn.LayerNorm([self.ntoken])
        
        self. hidden_dim =  hidden_dim
        self.state_dim = state_dim

    def forward(self, x):
        x = self.fc_0(x)
        x = self.norm_0(x)
        x = torch.flatten(x, 0).view(-1,self.d_model)
        x = self.transformer(x)
        if x.shape[0]==1:
            x = x[0]
        x = self.norm_1(x)
        x = torch.tanh(self.fc_1(x))
        
        return x