from prep_data import prepareData
from model import EncoderRNN, DecoderRNN, device
from train import get_dataloader, train, evaluate, evaluateRandom
from lang import SOS_TOKEN, EOS_TOKEN


def main():
    """Main function to run training and evaluation"""
    print(f"Using device: {device}")
    
    # Hyperparameters
    HIDDEN_SIZE = 256
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Prepare data
    print("Preparing data...")
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    
    # Get dataloader
    print("Creating dataloader...")
    input_lang, output_lang, train_dataloader = get_dataloader(BATCH_SIZE)
    
    # Initialize models
    print("Initializing models...")
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
    decoder = DecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(device)
    
    print(f"Encoder vocab size: {input_lang.n_words}")
    print(f"Decoder vocab size: {output_lang.n_words}")
    print(f"Number of training pairs: {len(pairs)}")
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    trained_encoder, trained_decoder = train(train_dataloader, encoder, decoder, EPOCHS, lr=LEARNING_RATE, print_every=10)
    
    # Evaluate model on random examples
    print("\nEvaluating model on random examples:")
    evaluateRandom(pairs, trained_encoder, trained_decoder, input_lang, output_lang, n=5)

    # Interactive evaluation
    # print("\nInteractive translation (type 'quit' to exit):")
    # while True:
    #     try:
    #         sentence = input("English: ").strip().lower()
    #         if sentence == 'quit':
    #             break
            
    #         output_words = evaluate(trained_encoder, trained_decoder, sentence, input_lang, output_lang)
    #         output_sentence = ' '.join(output_words)
    #         print(f"French: {output_sentence}")
            
    #     except KeyError as e:
    #         print(f"Unknown word: {e}")
    #     except KeyboardInterrupt:
    #         print("\nExiting...")
    #         break
    #     print()

if __name__ == "__main__":
    main()