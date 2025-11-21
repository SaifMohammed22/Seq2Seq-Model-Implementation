from lang import EOS_TOKEN
from model import device, MAX_LENGTH
from prep_data import prepareData
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


# Helper functions
def indexesFromSentence(lang, sentence):
    return [lang.word2idx[s] for s in sentence.split(" ")]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


# def tensorFromPair(pair, input_lang, output_lang):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, train_pairs, test_pairs = prepareData("eng", "fra", True)

    n = len(train_pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(train_pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        input_ids[idx,  :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids
    
    n_test = len(test_pairs)
    input_ids_test = np.zeros((n_test, MAX_LENGTH), dtype=np.int32)
    target_ids_test = np.zeros((n_test, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(test_pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        input_ids_test[idx,  :len(inp_ids)] = inp_ids
        target_ids_test[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))
    test_data = TensorDataset(torch.LongTensor(input_ids_test).to(device), torch.LongTensor(target_ids_test).to(device))

    train_sampler = RandomSampler(train_data)
    test_sampler = RandomSampler(test_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader, test_dataloader

#---------------------
# Training and testing functions

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)

        # Calculate loss
        # decoder_outputs: [batch_size, seq_len, vocab_size]
        # target_tensor: [batch_size, seq_len]
        
        loss = criterion(
            decoder_outputs.reshape(-1, decoder_outputs.size(-1)),  # [batch*seq, vocab]
            target_tensor.reshape(-1)  # [batch*seq]
        )

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, test_dataloader, encoder, decoder, epochs, lr=0.001, print_every=100):
    print_loss_total = 0 

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        test_loss = evaluate_epoch(test_dataloader, encoder, decoder, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            
            print(f"Epoch {epoch:3d}/{epochs} ({epoch/epochs*100:5.3f}) | Avg Training loss: {print_loss_avg} | Test loss: {test_loss:.4f}")
            print_loss_total = 0

    return encoder, decoder


def evaluate_epoch(dataloader, encoder, decoder, criterion):
    total_loss = 0

    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            # Forward pass
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)

            # Calculate loss
            loss = criterion(
                decoder_outputs.reshape(-1, decoder_outputs.size(-1)),  # [batch*seq, vocab]
                target_tensor.reshape(-1)  # [batch*seq]
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)

        loss = evaluate_epoch(None, encoder, decoder, nn.NLLLoss())

        _, topi = decoder_outputs.topk(1)
        decoder_id = topi.squeeze()

        decoder_words = []

        for idx in decoder_id:
            if idx.item() == EOS_TOKEN:
                decoder_words.append("<EOS>")
                break
            decoder_words.append(output_lang.idx2word[idx.item()])

        return decoder_words
    

def evaluateRandom(pairs, encoder, decoder, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")