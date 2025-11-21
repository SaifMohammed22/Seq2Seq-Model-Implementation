from prep_data import prepareData
from model import EncoderRNN, DecoderRNN, device
from train import get_dataloader, train, evaluateRandom, evaluate_epoch
import torch
import torch.nn as nn
import os


def main():
    """Main function to run training and evaluation"""
    print(f"Using device: {device}")
    
    # Hyperparameters
    HIDDEN_SIZE = 256
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
        
    # Get dataloader
    print("Creating dataloader...")
    input_lang, output_lang, train_dataloader, test_dataloader = get_dataloader(BATCH_SIZE)
    
    # Initialize models
    print("Initializing models...")
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
    decoder = DecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(device)
    
    print(f"Encoder vocab size: {input_lang.n_words}")
    print(f"Decoder vocab size: {output_lang.n_words}")
    print(f"Training pairs: {len(train_dataloader.dataset)}")
    print(f"Test pairs: {len(test_dataloader.dataset)}")
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    trained_encoder, trained_decoder = train(train_dataloader, test_dataloader, encoder, decoder, EPOCHS, lr=LEARNING_RATE, print_every=10)
     
    # Save trained models
    os.makedirs("models", exist_ok=True)
    torch.save(trained_encoder.state_dict(), "models/trained_encoder.pth")
    torch.save(trained_decoder.state_dict(), "models/trained_decoder.pth")
    
    # Evaluate model on random examples
    print("\nEvaluating model on random examples:")
    evaluateRandom(test_dataloader.dataset, trained_encoder, trained_decoder, input_lang, output_lang, n=5)


if __name__ == "__main__":
    main()