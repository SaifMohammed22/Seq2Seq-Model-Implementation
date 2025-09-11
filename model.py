import torch
import torch.nn as nn
import torch.nn.functional as F
from lang import SOS_TOKEN


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 10

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True) # Input with batch_first=True: [batch, seq_length, input_size] instead
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        # input_seq: [batch_size, seq_len]
        embedded = self.embedding(input)
        # embedded: [batch_size, seq_len, hidden_size]
        embedded = self.dropout(embedded)
        # output: [batch_size, seq_len, hidden_size]
        # hidden: [1, batch_size, hidden_size]
        output, hidden = self.lstm(embedded)
        return output,hidden



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.5):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, encoder_output, encoder_hidden):
        batch_size = encoder_output.size(0)

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output.unsqueeze(1))  # [batch_size, 1, output_size]

            _, topi = decoder_output.topk(1)
            decoder_input = topi.detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [batch_size, seq_len, output_size]
        # Apply log softmax to get log probabilities
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden
    

    def forward_step(self, input_token, hidden_state):
        # input_token: [batch_size, 1]
        # embedded: [batch_size, 1, hidden_size]
        embedded = self.dropout(self.embedding(input_token))
        # output: [batch_size, 1, hidden_size]
        output, hidden = self.lstm(embedded, hidden_state)
        # pred: [batch_size, hidden_size]
        pred = self.out(output.squeeze(1))
        return pred, hidden



if __name__ == "__main__":
    # Test parameters
    input_vocab_size = 1000   # English vocabulary size
    output_vocab_size = 800   # French vocabulary size  
    hidden_size = 256
    batch_size = 2
    seq_length = 5
    
    # Create models
    encoder = EncoderRNN(input_vocab_size, hidden_size)
    decoder = DecoderRNN(hidden_size, output_vocab_size)
    
    # Create dummy input (random word indices)
    dummy_input = torch.randint(0, input_vocab_size, (batch_size, seq_length))
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input: {dummy_input}")
    
    # Test encoder
    print("\n--- Testing Encoder ---")
    encoder_output, encoder_hidden = encoder(dummy_input)
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Encoder hidden shapes: h={encoder_hidden[0].shape}, c={encoder_hidden[1].shape}")
    
    # Test decoder
    print("\n--- Testing Decoder ---")
    decoder_outputs, decoder_hidden = decoder(encoder_output, encoder_hidden)
    print(f"Decoder outputs shape: {decoder_outputs.shape}")
    print(f"Decoder outputs range: [{decoder_outputs.min():.3f}, {decoder_outputs.max():.3f}]")
    
    # Test single forward step
    print("\n--- Testing Single Decoder Step ---")
    decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_TOKEN)
    step_output, step_hidden = decoder.forward_step(decoder_input, encoder_hidden)
    print(f"Single step output shape: {step_output.shape}")
    print(f"Single step output (first sample): {step_output[0][:5]}...")  # First 5 logits
    
    # Test prediction
    print("\n--- Testing Prediction ---")
    _, predicted_indices = step_output.topk(1)
    print(f"Predicted next words (indices): {predicted_indices.squeeze(1).detach()}")
    
    print("\n✅ Model test completed successfully!")
