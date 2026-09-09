# model.py
# --------
# This is the main neural network.
# I built it myself from scratch (no pretrained transformers).
#
# Big picture:
#
#   Urdu sentence with <ans> tags
#            |
#            v
#        ENCODER  (reads the sentence)
#        2-layer BiLSTM
#            |
#            v
#        ATTENTION (looks at important words)
#            |
#            v
#        DECODER  (writes the question)
#        2-layer LSTM
#            |
#            v
#        Urdu question

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Encoder(nn.Module):
    """
    Encoder = the reader.
    It reads the input sentence from left to right AND right to left
    (that is why it is BiLSTM / bidirectional).
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()

        # embedding turns token id into a vector of numbers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=config.PAD_ID)

        # LSTM is a type of RNN that remembers sequence info
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # because bidirectional, hidden becomes 2x size
        # I map it back to normal hidden size for the decoder
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        # src shape: (batch_size, sentence_length)

        # 1) words/ids -> vectors
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # 2) pack padded sequence
        # this tells LSTM to ignore PAD tokens
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        # 3) run LSTM
        outputs, (hidden, cell) = self.lstm(packed)

        # 4) unpack back to normal tensor
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs has info for every input word
        # shape roughly: (batch, src_len, hidden*2)

        # 5) combine forward and backward states for each layer
        # BiLSTM stores them like: fwd0, bwd0, fwd1, bwd1 ...
        num_layers = hidden.size(0) // 2
        hidden_layers = []
        cell_layers = []

        for layer in range(num_layers):
            h_forward = hidden[2 * layer]
            h_backward = hidden[2 * layer + 1]
            c_forward = cell[2 * layer]
            c_backward = cell[2 * layer + 1]

            h_combined = torch.cat((h_forward, h_backward), dim=1)
            c_combined = torch.cat((c_forward, c_backward), dim=1)

            h_combined = torch.tanh(self.fc_hidden(h_combined))
            c_combined = torch.tanh(self.fc_cell(c_combined))

            hidden_layers.append(h_combined)
            cell_layers.append(c_combined)

        hidden = torch.stack(hidden_layers, dim=0)
        cell = torch.stack(cell_layers, dim=0)
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """
    Attention = "where should I look right now?"

    When decoder is writing one word of the question,
    attention checks all encoder words and gives higher weight
    to more useful words (often near the <ans> part).
    """

    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden, dec_hidden, bias=False)
        self.W_dec = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: what decoder currently remembers
        # encoder_outputs: all input word memories

        src_len = encoder_outputs.size(1)

        # copy decoder state for every input position
        dec_rep = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # score each input word
        energy = torch.tanh(self.W_enc(encoder_outputs) + self.W_dec(dec_rep))
        scores = self.v(energy).squeeze(2)

        # dont attend to PAD positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # turn scores into probabilities (add up to 1)
        attn_weights = F.softmax(scores, dim=1)

        # weighted sum = context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    """
    Decoder = the writer.
    It writes the question one token at a time.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden, num_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=config.PAD_ID)
        self.attention = BahdanauAttention(enc_hidden, hidden_size)

        # LSTM input is: current word embedding + attention context
        self.lstm = nn.LSTM(
            embed_size + enc_hidden,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # final layer predicts next token id
        self.fc_out = nn.Linear(hidden_size + enc_hidden + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, cell, encoder_outputs, mask):
        """
        Do only ONE step of writing.
        input_token = current word id
        output = scores for next word
        """
        # embed current token
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)

        # use top layer hidden for attention
        top_hidden = hidden[-1]
        context, attn_weights = self.attention(top_hidden, encoder_outputs, mask)

        # combine embedding + context and give to LSTM
        lstm_in = torch.cat((embedded, context), dim=1)
        lstm_in = lstm_in.unsqueeze(1)  # add time dimension
        output, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        output = output.squeeze(1)

        # predict next token
        combined = torch.cat((output, context, embedded), dim=1)
        prediction = self.fc_out(combined)
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """
    Full model = Encoder + Decoder together.
    """

    def __init__(self, vocab_size):
        super().__init__()

        # encoder is bidirectional so output size is hidden*2
        enc_hidden = config.HIDDEN_SIZE * 2

        self.encoder = Encoder(
            vocab_size,
            config.EMBED_SIZE,
            config.HIDDEN_SIZE,
            config.NUM_LAYERS,
            config.DROPOUT,
        )
        self.decoder = Decoder(
            vocab_size,
            config.EMBED_SIZE,
            config.HIDDEN_SIZE,
            enc_hidden,
            config.NUM_LAYERS,
            config.DROPOUT,
        )
        self.vocab_size = vocab_size

    def make_src_mask(self, src):
        # 1 means real token, 0 means pad
        mask = (src != config.PAD_ID).float()
        return mask

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        """
        Training forward pass.

        Teacher forcing meaning:
        sometimes I give the model the REAL next word
        instead of its own guess.
        This helps it learn faster.
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # encode whole input sentence
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_src_mask(src)

        # first decoder input is <s>
        input_token = tgt[:, 0]

        # store predictions for every time step
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size, device=src.device)

        # generate one token at a time
        for t in range(1, tgt_len):
            prediction, hidden, cell, _ = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, mask
            )
            outputs[:, t] = prediction

            # choose next input
            use_teacher = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)  # model's best guess

            if use_teacher:
                input_token = tgt[:, t]  # real word
            else:
                input_token = top1       # model guess

        return outputs
