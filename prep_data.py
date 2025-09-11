from data_preprocessing import readLang, filterPairs
import random


# Prepare data for training
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLang(lang1, lang2, reverse=reverse)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words.")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
    

if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    random_pairs = random.choice(pairs)
    print(random_pairs)