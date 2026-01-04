import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.lang import SOS_TOKEN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 10


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    

class SoftAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua( keys)))
        scores = scores.squeeze(2).unsqueeze(1)  # (batch_size, seq_len)

        weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)
        context = torch.bmm(weights, keys)  # (batch_size, 1, hidden_size)

        return context, weights
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self). __init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = SoftAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_ratio=0.5):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, att_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(att_weights)

            if target_tensor is not None:
                # Teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
            
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)  # (batch_size, seq_len, encoder_seq_len)

        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)  # (batch_size, 1, hidden_size)
        context, att_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, att_weights


if __name__ == "__main__":
    # Simple test to check model dimensions
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    hidden_size = 8

    encoder = EncoderRNN(vocab_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, vocab_size).to(device)

    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor)

    decoder_outputs, decoder_hidden, attentions = decoder(encoder_outputs, encoder_hidden)

    print("Encoder outputs shape:", encoder_outputs.shape)  # Expected: (batch_size, seq_len, hidden_size)
    print("Decoder outputs shape:", decoder_outputs.shape)  # Expected: (batch_size, seq_len, vocab_size)
    print("Attention weights shape:", attentions.shape)     # Expected: (batch_size, seq_len, seq_len)