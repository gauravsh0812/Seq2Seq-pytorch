import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

class Encoder_CNN(nn.Module):

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

        # why kernel is (filter - 1//2):  https://discuss.pytorch.org/t/how-can-i-ensure-that-my-conv1d-retains-the-same-shape-with-unknown-sequence-lengths/73647

        self.drop = nn.Dropout(dropout)


    def forward(self, src):

        # src = [batch, seq_len]
        batch_size = src.shape[0]
        seq_len = src.shape[1]

        # token_emb/pos_emb=[batch, seq_len, emb_dim]
        # create a position tensor of shape [batch, seq_len].
        # can be build in the following way:
        # [batch, 1] x [1, seq_len] = [batch, seq_len]
        pos_tensor = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size,1)
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
        for idx, layer in enumerate(self.conv_layers):
            # conv_input = [batch, hidd_dim, seq_len]
            # layer(conv_input) = [batch, hidd_dim*2, seq_len]
            # layer_output = [batch, hidd_dim, seq_len]
            layer_output = F.glu(self.drop(layer(conv_input)), dim =1)
            conved = (layer_output + conv_input) * self.scale
            # conved = [batch, hidd_dim, seq_len]
            conv_input = conved

        # final output
        conved = self.fc_hid2emb(conved.permute(0,2,1))  # [batch,  seq_len,emb_dim]
        combined_output = (conved + embedded) * self.scale # [batch, seq_len, emb_dim]

        return conved, combined_output

class Decoder_CNN(nn.Module):
        def __init__(self, output_dim, emb_dim, hidd_dim, n_layers, kernel_size, dropout, trg_pad_idx, device, max_length = 150):

            super().__init__()
            self.device = device
            # to control the varince over different seeds
            self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

            self.kernel_size = kernel_size
            self.device = device
            self.trg_pad_idx = trg_pad_idx
            self.hidd_dim = hidd_dim
            self.token_emb = nn.Embedding(output_dim, emb_dim)
            self.pos_emb = nn.Embedding(max_length, emb_dim)

            self.fc_emb2hid = nn.Linear(emb_dim, hidd_dim)
            self.fc_hid2emb = nn.Linear(hidd_dim, emb_dim)
            self.fc = nn.Linear(emb_dim, output_dim)
            self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels = hidd_dim,
                                          out_channels = 2*hidd_dim,
                                          kernel_size = kernel_size) for _ in range(n_layers)])
            self.drop = nn.Dropout(dropout)

        def attention(self, conved, emb_conved, enc_conved, enc_combined):
            # conved = [b,h,tl]
            # emb_conved = [b,trg_l,e]
            # enc_conved/enc_combined = [b,src_l,e]
            energy = torch.matmul(emb_conved, enc_conved.permute(0,2,1))  #[b,dec_l,src_l]
            attn = F.softmax(energy)  # [b, tl, sl]
            attn_encoding = torch.matmul(attn, enc_combined)  # [b, tl ,sl] [b, sl, e] --> [b, tl, e]
            attn_encoding = self.fc_emb2hid(attn_encoding)  # [b, tl, h]
            # residual connection of "attention block"
            attn_combined = (attn_encoding.permute(0,2,1) + conved) * self.scale  # [b,h,tl]

            return attn, attn_combined

        def forward(self, trg, enc_conved, enc_combined):
            # trg = [batch, trg_len]
            batch_size = trg.shape[0]
            trg_len = trg.shape[1]

            # embedded = [batch, trg_len, emb_dim] == [b, l, e]
            pos_tensor = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            embedded = self.drop(self.token_emb(trg) + self.pos_emb(pos_tensor))
            conv_input = self.fc_emb2hid(embedded).permute(0,2,1)   # [b,h,l]
            for layer in self.conv_layers:
                # let's first do padding
                # pad_tensor = [b,h,k-1]
                pad_tensor = torch.zeros(batch_size, self.hidd_dim, self.kernel_size-1).fill_(self.trg_pad_idx).to(self.device)
                # conv_input = [b,h,l+k-1]
                conved = F.glu(self.drop(layer(torch.cat((conv_input, pad_tensor), dim=2))), dim=1)  # dec_conved = [b,h,l]
                emb_conved = self.fc_hid2emb(conved.permute(0,2,1))  # [b,l, e]
                emb_conved = (emb_conved+ embedded) * self.scale # [b,l,e]
                attn, attn_combined = self.attention(conved, emb_conved, enc_conved, enc_combined)
                # final residual connection
                combined = (attn_combined + conv_input) * self.scale   # [b,h,l]
                conv_input  = combined

            output = self.fc(self.drop(self.fc_hid2emb(combined.permute(0,2,1))))      # [b,l,o]
            return output, attn

class Seq2Seq_CNN(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):

        # src = [batch, src_len]
        # trg = [batch, trg_len-1] ; since we don't want to give <eos> as input token
        enc_conved, enc_combined = self.encoder(src)
        #print('======='*5)
        #print('shape of enc_coved: ', enc_conved.shape)
        #print('shape of enc_combined: ', enc_combined.shape)
        output, attn = self.decoder(trg, enc_conved, enc_combined)
        #print ('shape of output_dec:  ', output.shape)
        #print('shape of attn:  ', attn.shape)
        return output
