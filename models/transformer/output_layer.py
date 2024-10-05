import torch.nn as nn


ACT_LAYERS = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    None: nn.Identity(),
}

class AttentionOutput(nn.Module):
    def __init__(self, d_model, dropout=None, activation_fn='relu'):
        super(AttentionOutput, self).__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = ACT_LAYERS[activation_fn]
        self.squeeze = nn.Linear(d_model * 2, d_model)
        if dropout is None or dropout <= 0:
            self.dropout =  nn.Identity()
        else:
            self.dropout =  nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states):
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(input_states + hidden_states)
        return output_states
