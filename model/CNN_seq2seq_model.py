import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, hidd_dim, n_layers, kernel_size, dropout, device, max_length = 150):

        super().__init__()

        # kernel must be odd numbered
        assert kernel_size % 2 == 1; "kernel size has to be odd numbered."

        self.device = device
        # to control the varince over different seeds
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.token_emb = nn.Embedding(input_dim, emb_dim)
        self.pos_emb = nn.Embedding(max_length, emb_dim)    # max no. of token that can be passed

        self.fc_emb2hid = nn.Linear(emb_dim, hidd_dim)
        self.fc_hid2emb = nn.Linear(hidd_dim, emb_dim)

        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels = hidd_dim,
                                      out_channels = 2*hidd_dim,
                                      kernel_size = kernel_size,
                                      padding = (kernel_size -1) //2) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)


    def forward(self, src):

        # src = [batch, seq_len]
        batch_size = src.shape[0]
        seq_len = src.shape[1]

        # token_emb/pos_emb=[batch, seq_len, emb_dim]
        # create a position tensor of shape [batch, seq_len].
        # can be build in the following way:
        # [batch, 1] x [1, seq_len] = [batch, seq_len]
        pos_tensor = torch.arrange(0, seq_len).unsqueeze(0).repeat(batch_size,1)
        # since we have created pos tensor outside of preprocessing
        # it will not be on "device", hence we nned to move pos_tensor to "device"
        pos_tensor.to(self.device)

        embedded = self.drop(self.token_emb(src) + self.pos_emb(pos_tensor))
        # embedded = [batch, seq_len, emb_dim]

        # since Conv1d requires input in [N, Cin, L] i.e. [batch, hidd_dim, seq_len]
        # we will use permute
        conv_input = self.fc_emb2hid(embedded).permute(0, 2, 1)


        # pass the conv_input to the conv_layers i.e. iterator
        # the input to the first layer of the conv_layers will going to be conv_input
        # and for the next layer, the ouptput of the the prev layer will act as inputs
        for idx, layer in enumerate(conv_layers):
            # conv_input = [batch, hidd_dim, seq_len]
            # layer(conv_input) = [batch, hidd_dim*2, seq_len]
            # layer_output = [batch, hidd_dim, seq_len]
            layer_output = F.glu(self.drop(layer(conv_input)))
            conved = (layer_output + conv_input) * self.scale
            # conved = [batch, hidd_dim, seq_len]
            conv_input = conved

        # final output
        conved = self.fc_hid2emb(conved)  # [batch, emb_dim, seq_len]
        combined_output = (conved + embedded.permute(0,2,1)) * self.scale # [batch, emb_dim, deq_len]
        combined_output = combined_output.permute(0,2, 1) # [batch, seq_len, emb_dim]

        return conved, combined_output
            
