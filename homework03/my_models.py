import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, p=0.5):
        super().__init__()
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim
        )
        self.dropout = nn.Dropout(p)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=p
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        seq, h_0 = self.rnn(x)
        # seq.shape: [seq_len, batch_size, hid_dim]
        # h_0.shape: [1, batch_size, hid_dim]

        return seq, h_0


class CNN_Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, kernel_sizes, hid_dim, p=0.5):
        super().__init__()
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim
        )
        self.dropout = nn.Dropout(p)

        self.convs = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(
                    emb_dim * (i + 1),
                    emb_dim * (i + 2),
                    kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
            ) for i, kernel_size in enumerate(kernel_sizes[-1:])],
            nn.Conv1d(
                emb_dim * len(kernel_sizes),
                hid_dim,
                kernel_sizes[-1],
                padding=kernel_sizes[-1] // 2
            ),
            nn.AdaptiveMaxPool1d(1)
        )

        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.convs(x.permute(1, 2, 0))

        seq = x.permute(2, 0, 1)
        # seq.shape: [seq_len, batch_size, hid_dim]

        h_0 = self.maxpool(x).permute(2, 0, 1)
        # h_0.shape: [1, batch_size, hid_dim]

        return seq, h_0


class GRU_Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, p=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim
        )

        self.dropout = nn.Dropout(p)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=p
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, h_0, enc_seq=None):
        x = self.embedding(x.unsqueeze(0))
        x = self.dropout(x)
        output, h_n = self.rnn(x, h_0)
        pred = self.fc(output.squeeze(0))
        return pred, h_n.squeeze(0)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()

    def forward(self, enc_seq, h):
        scores = torch.bmm(
            enc_seq.permute(1, 0, 2),
            h[-1:].permute(1, 2, 0)
        ).squeeze(2)
        # scores.shape: [batch_size, seq_len]
        weights = F.softmax(scores, dim=1)
        # weights.shape: [batch_size, seq_len]
        attention_out = torch.sum(enc_seq * weights.permute(1, 0).unsqueeze(2), axis=0)
        # attention_out.shape: [batch_size, hid_dim]
        return attention_out


class GRU_Decoder_With_Attention(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, p=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim
        )

        self.dropout = nn.Dropout(p)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=p
        )
        self.attention = Attention(hid_dim, n_layers * hid_dim)
        self.fc = nn.Linear(2 * hid_dim, vocab_size)

    def forward(self, x, h_0, enc_seq):
        x = self.embedding(x.unsqueeze(0))
        x = self.dropout(x)
        output, h_n = self.rnn(x, h_0)
        attention_out = self.attention(enc_seq, h_n)
        output = torch.cat([output.squeeze(0), attention_out], dim=1)
        pred = self.fc(output)
        return pred, h_n


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_seq, hidden = self.encoder(src)
        if isinstance(self.encoder, CNN_Encoder):
            hidden = torch.cat([hidden] * self.decoder.n_layers, dim=0)

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, enc_seq)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs
