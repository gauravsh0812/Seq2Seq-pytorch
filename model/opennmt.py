import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext


class Encoder(nn.Module):

    def __init__(self, input_channel, hid_dim, n_layers, kernel_size, dropout, device):
        super(Encoder, self).__init__()

        # kernel size must be odd
        assert kernel_size % 2==1: "kernle size must be odd."

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)
        self.hid_dim = hid_dim
        self.conv_layer1 = nn.Conv2d(input_channel, 64, kernel=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer2 = nn.Conv2d(64, 128, kernel=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer3 = nn.Conv2d(128, 256, kernel=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer4 = nn.Conv2d(256, 256, kernel=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer5 = nn.Conv2d(256, 512, kernel=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer6 = nn.Conv2d(512, 512, kernel=(3,3), stride=(1,1), padding =(1,1))
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.emb = nn.Embedding(512, 512)
        self.lstm = nn.LSTM(512, hid_dim,n_layers=2, dropout=0.3, bidirectional=True)
        self.fc_hidden = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # img = [batch, Cin, W, H]
        batch = img.shape[0]
        C_in = img.shape[1]

        enc_output = nn.Sequential(
                # src = [batch, Cin, w, h]
                # layer 1
                src = self.maxpool(F.relu(self.conv_layer1(src), inplace=True)),
                # layer 2
                src = self.maxpool(F.relu(self.conv_layer2(src), inplace=True)),
                # layer 3
                src = F.relu(self.batch_norm1(self.conv_layer3(src)), inplace=True),
                # layer 4
                src = self.maxpool(F.relu(self.conv_layer4(src), inplace=True),
                                    kernel_size=(1,2), stride=(1,2)),
                # layer 5
                src = self.maxpool(F.relu(self.batch_norm2(self.conv_layer5(src)), inplace=True),
                                    kernel_size=(2,1), stride=(2,1)),
                # layer 6
                # [batch, 512, W', H']
                src = F.relu(self.batch_norm2(self.conv_layer6(src)), inplace=True)
                )

        all_outputs = []
        for ROW in range(0, enc_output.shape(2)):
            # [batch, 512, W] since for each row,
            # it becomes a 2d matrix of [512, W] for all batches
            row = enc_output[:,:,ROW,:]
            row = row.permute(2,0,1)  # [W, batch, 512(enc_output)]
            pos_vec = torch.Tensor(row.shape[1]).long().fill_(ROW) # [batch]
            # self.emb(pos) ==> [batch, 512]
            lstm_input = torch.cat((self.emb(pos).unsqueeze(0), row), dim = 0) # [W+1, batch, 512]
            # output = [W+1, batch, hid_dimx2]
            # hidden/cell = [2x2, batch, hid_dim]
            # we want the fwd and bckwd directional final layer
            lstm_output, hidden, cell = self.lstm(embed)

            all_outputs.append(lstm_output.unsqueeze(0))

        final_encoder_output = torch.cat(all_outputs, dim =0)  #[H, W+1, BATCH, hid_dimx2]
        final_encoder_output = final_encoder_output.view(
                                            final_encoder_output.shape[0]*final_encoder_output.shape[1],
                                            final_encoder_output.shape[2], final_encoder_output.shape[3])

        # final hidden/cell layer
        for LAYER in [hidden, cell]:
            fwd_layer1, fwd_layer2 = LAYER[-1, :, :], LAYER[-3,:,:]  # [batch, hid_dim]
            bkd_layer1, bkd_layer2 = LAYER[-2, :, :], LAYER[-4,:,:]
            fwd_layer = self.fc_hidden(torch.cat((fwd_layer1, fwd_layer2), dim=1))  # [batch, hid_dim]
            bkd_layer = self.fc_hidden(torch.cat((bkd_layer1, bkd_layer2), dim=1))  # [batch, hid_dim]
            LAYER = torch.tanh(self.fc_hidden(torch.cat((fwd_layer, bkd_layer), dim=1)))  # [batc, hid_dim]

        return final_encoder_output, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.attn = torch.tanh(nn.Linear(hid_dim*4, hid_dim))
        self.attn_vector = nn.Linear(hid_dim, 1)

    def forward(self, final_encoder_output, hidden, cell):
        # final_encoder_output [H*W+1, batch, hid_dim*2]
        # hidden/cell = [batch, hid_dim]
        hidden = hidden.unsqueeze(0).repeat(final_encoder_output.shape[0], 1, 1)
        cell = cell.unsqueeze(0).repeat(final_encoder_output.shape[0], 1, 1)
        # hidden/cell = [H*W+1, batch, hid_dim]
        energy = self.attn(torch.cat((final_encoder_output, hidden , cell), dim=2))
        # energy = [H*W+1, batch, hid_dim]
        # attention only has to be over seq length, therefor what we want as output
        # is a tensor of dimension [batch, src_len] i.e. [batch, H*W+1]
        attn = self.attn_vector(energy).squeeze(2) # [batch, H*W+1]
        return F.softmax(attn)

class Decoder(nn.Module):
    def __init__(self, trg, emb_dim, hid_dim, output_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()

        seld.emb = nn.Embedding(input_dim, emb_dim)
        self.hid_dim = hid_dim
        self.attention = attention
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers=2, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hid_dim*3+emb_dim , output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell, final_encoder_output):
        # trg = [batch, trg_len]
        trg = trg.unsqueeze(0)  # [1, batch, trg_len]
        embed = self.drop(self.emb(trg))  # [1, batch, emb_dim]
        attn  = self.attention(final_encoder_output, hidden, cell).unsqueeze(1)  # [batch,1, H*W+1]
        # final_encoder_output [H*W+1, batch, hid_dim*2]
        final_encoder_output = final_encoder_output.permute(1,0,2)  # [batch, H*W+1, hid_dim*2]
        # final bmm dimension will going to be [batch, 1, hid_dim*2]
        wgt = torch.bmm(attn, final_encoder_output).permute(1,0,2)  # [1, batch, hid-dim*2]
        lstm_input = torch.cat((wgt, embed), dim=2)  # [1, batch, hid_dim*2 + emb_dim]
        lstm_output, hidden, cell = self.lstm(lstm_input, hidden.unsqueeze(0), cell.unsqueeze(0))
        # lstm_output = [1, batch, hid_dim]
        prediction_input = torch.cat((output, embed, wgt), dim=2)  #[1, batch, (hid_dim*3+emb_dim)]
        prediction = self.fc(prediction_input.squeeze(0))

        return prediction, hidden.squeeze(0), cell.squeeze(0)
