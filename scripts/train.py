from src.data.lang import EOS_TOKEN, SOS_TOKEN
from src.models.model import device, MAX_LENGTH
from src.data.prep_data import prepareData
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.utils.utils import timeSince, showPlot
import time


def indexesFromSentence(lang, sentence):
    return [lang.word2idx[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size):
    input_lang, output_lang, train_pairs, test_pairs = prepareData('eng', 'fra', True)
    
    # Prepare training data
    n = len(train_pairs)
    import numpy as np
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(train_pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids
    
    # Prepare test data
    n_test = len(test_pairs)
    input_ids_test = np.zeros((n_test, MAX_LENGTH), dtype=np.int32)
    target_ids_test = np.zeros((n_test, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(test_pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        input_ids_test[idx, :len(inp_ids)] = inp_ids
        target_ids_test[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device),
        torch.LongTensor(target_ids).to(device)
    )
    test_data = TensorDataset(
        torch.LongTensor(input_ids_test).to(device),
        torch.LongTensor(target_ids_test).to(device)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader, test_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_epoch(dataloader, encoder, decoder, criterion):
    """Evaluate on test set"""
    total_loss = 0
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            # Use teacher forcing for evaluation (more stable loss measurement)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor, teacher_forcing_ratio=1.0)

            loss = criterion(
                decoder_outputs.reshape(-1, decoder_outputs.size(-1)),
                target_tensor.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn



def evaluateRandom(pairs, encoder, decoder, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def train(train_dataloader, test_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=10, plot_every=10):
    start = time.time()
    plot_losses = []
    test_plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    test_plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        test_loss = evaluate_epoch(test_dataloader, encoder, decoder, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        test_plot_loss_total += test_loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100:.0f}%) {print_loss_avg:.4f}')

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            test_plot_loss_avg = test_plot_loss_total / plot_every
            test_plot_losses.append(test_plot_loss_avg)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            test_plot_loss_total = 0

    showPlot(plot_losses)
    showPlot(test_plot_losses)