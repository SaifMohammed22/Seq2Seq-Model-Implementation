from data_preprocessing import readLang, filterPairs
import random


def prepareData(lang1, lang2, reverse=False, test_ratio=0.2, seed=42):
    input_lang, output_lang, pairs = readLang(lang1, lang2, reverse=reverse)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    
    # Shuffle before splitting to avoid data leakage
    random.seed(seed)
    random.shuffle(pairs)
    
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words.")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    # Training pairs and Test pairs
    n_train = int((1 - test_ratio) * len(pairs))
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]
    
    return input_lang, output_lang, train_pairs, test_pairs


if __name__ == "__main__":
    input_lang, output_lang, train_pairs, test_pairs = prepareData("eng", "fra", True)
    print(f"Example pair: {train_pairs[0]}")