import time
import torch
from train import evaluate
from prep_data import prepareData
from model import EncoderRNN, DecoderRNN
from nltk.translate.bleu_score import corpus_bleu



def load_models(input_lang, output_lang, hidden_size=256, encoder_path="encoder.pth", decoder_path="decoder.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder, device


def benchmark_bleu_and_time(input_lang, output_lang, test_pairs, encoder, decoder, device):
    references = []
    hypotheses = []

    start_time = time.time()
    with torch.no_grad():
        for pair in test_pairs:
            # pair[0] is input, pair[1] is target
            input_sentence = pair[0]
            references = pair[1]
            output_words, _ = evaluate(encoder, decoder, input_lang, output_lang, input_sentence, device)
            output_sentence = " ".join(output_words)

            # Prepare for BLEU: list of tokens
            ref_tokens = references.split(" ")
            hyp_tokens = output_sentence.split(" ")

            references.append([ref_tokens])  # corpus_bleu expects list of list of refs
            hypotheses.append(hyp_tokens)

    total_time = time.time() - start_time
    sentences_per_sec = len(test_pairs) / total_time if total_time > 0 else 0.0

    bleu = corpus_bleu(references, hypotheses) * 100.0

    print(f"Sentences evaluated: {len(test_pairs)}")
    print(f"Corpus BLEU: {bleu:.2f}")
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Sentences per second: {sentences_per_sec:.2f}")


def main():
    """Run benchmarking on test set"""
    print("="*60)
    print("BENCHMARKING SEQ2SEQ MODEL")
    print("="*60)
    
    # Load test data (use SAME seed as training!)
    input_lang, output_lang, _, test_pairs = prepareData("eng", "fra", True, test_ratio=0.2, seed=42)
    
    print(f"\nTest set size: {len(test_pairs)} pairs")
    
    # Load trained models
    HIDDEN_SIZE = 256
    encoder, decoder = load_models(
        encoder_path="models/trained_encoder.pth",
        decoder_path="models/trained_decoder.pth",
        input_size=input_lang.n_words,
        hidden_size=HIDDEN_SIZE,
        output_size=output_lang.n_words
    )
    print("✅ Models loaded\n")
    
    # Benchmark
    bleu, sentences_per_sec = benchmark_bleu_and_time(
        test_pairs, encoder, decoder, input_lang, output_lang
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Corpus BLEU:      {bleu:.2f}")
    print(f"Inference speed:  {sentences_per_sec:.2f} sentences/sec")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
