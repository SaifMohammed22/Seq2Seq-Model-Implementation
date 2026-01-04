import time
import torch
from scripts.train import evaluate
from src.data.prep_data import prepareData
from src.models.model import EncoderRNN, AttnDecoderRNN, device
from nltk.translate.bleu_score import corpus_bleu


def benchmark_bleu_and_time(input_lang, output_lang, test_pairs, encoder, decoder, device):
    all_references = [] # Renamed to avoid conflict with string assignment
    hypotheses = []

    start_time = time.time()
    with torch.no_grad():
        for pair in test_pairs:
            input_sentence = pair[0]
            target_sentence = pair[1] # Store the target sentence in a new variable

            # Correct call to evaluate based on its definition: evaluate(encoder, decoder, sentence, input_lang, output_lang)
            output_words, _ = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
            output_sentence = " ".join(output_words)

            # Prepare for BLEU: list of tokens
            ref_tokens = target_sentence.split(" ")
            hyp_tokens = output_sentence.split(" ")

            all_references.append([ref_tokens])  # corpus_bleu expects list of list of refs
            hypotheses.append(hyp_tokens)

    total_time = time.time() - start_time
    sentences_per_sec = len(test_pairs) / total_time if total_time > 0 else 0.0

    bleu = corpus_bleu(all_references, hypotheses) * 100.0 # Use all_references here

    print(f"Sentences evaluated: {len(test_pairs)}")
    print(f"Corpus BLEU: {bleu:.2f}")
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Sentences per second: {sentences_per_sec:.2f}")

    return bleu, sentences_per_sec # Function needs to return these values


def main():
    """Run benchmarking on test set"""
    print("="*60)
    print("BENCHMARKING SEQ2SEQ MODEL")
    print("="*60)

    # Load test data (use SAME seed as training!)
    input_lang, output_lang, _, test_pairs = prepareData("eng", "fra", True, test_ratio=0.2, seed=42)

    print(f"\nTest set size: {len(test_pairs)} pairs")

    # Load trained models
    # Ensure HIDDEN_SIZE matches the one used during training (128 from cell 7_htl508zwas)
    TRAINING_HIDDEN_SIZE = 128

    encoder = EncoderRNN(input_lang.n_words, TRAINING_HIDDEN_SIZE).to(device)
    decoder = AttnDecoderRNN(TRAINING_HIDDEN_SIZE, output_lang.n_words).to(device) # Use AttnDecoderRNN

    # Load trained weights
    encoder.load_state_dict(torch.load("models/trained_encoder.pth"))
    decoder.load_state_dict(torch.load("models/trained_decoder.pth"))

    encoder.eval()
    decoder.eval()

    # Benchmark
    bleu, sentences_per_sec = benchmark_bleu_and_time(
        input_lang, output_lang, test_pairs, encoder, decoder, device # Corrected argument order
    )

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Corpus BLEU:      {bleu:.2f}")
    print(f"Inference speed:  {sentences_per_sec:.2f} sentences/sec")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
