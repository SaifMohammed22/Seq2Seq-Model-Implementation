from src.data.lang import Lang
from io import open
import unicodedata
import re


# Convert Unicode (U+0041) -> ASCII (65)
def UnicodeToAscii(s):
    "Decomposes characters into base + combining characters + Remove accents, etc + Return clean ASCII"
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = UnicodeToAscii(s.strip().lower())
    # Add space before punctuation for separation, so that they are treated as separate tokens
    s = re.sub(r"([.!?])", r" \1", s)
    # Remove all non-letter characters except for punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()


def readLang(lang1, lang2, reverse=False):
    print("Reading lines....")
    lines = open(f"data/{lang1}-{lang2}.txt").read().strip().split("\n")
    
    pairs = []
    for l in lines:
        parts = l.split("\t")
        if len(parts) == 2:
            pairs.append([normalizeString(s) for s in parts])

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# Clip dataset for simple training
MAX_LENGTH = 10

# English prefixes for filtering and narrowing dataset for simple training
eng_prefix = (
    # Original prefixes
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "we are", "we re",
    "they are", "they re",
    
    # Additional common patterns
    "i was", "i have", "i ve", "i had", "i d", "i will", "i ll",
    "you were", "you have", "you ve", "you had", "you d", "you will", "you ll",
    "he was", "he has", "he had", "he d", "he will", "he ll",
    "she was", "she has", "she had", "she d", "she will", "she ll",
    "we were", "we have", "we ve", "we had", "we d", "we will", "we ll",
    "they were", "they have", "they ve", "they had", "they d", "they will", "they ll",
    
    # Common sentence starters
    "it is", "it s", "it was", "it has",
    "this is", "this was",
    "that is", "that s", "that was",
    "there is", "there s", "there are", "there were",
    
    # Question words
    "what is", "what s", "what are", "what was",
    "where is", "where s", "where are", "where was",
    "when is", "when s", "when are", "when was",
    "who is", "who s", "who are", "who was",
    "why is", "why are", "why was",
    "how is", "how s", "how are", "how was",
    
    # Modal verbs
    "i can", "i could", "i should", "i would",
    "you can", "you could", "you should", "you would",
    "we can", "we could", "we should", "we would",
    "they can", "they could", "they should", "they would",
    
    # Negations
    "i don", "i didn", "i won", "i can t", "i couldn t",
    "you don", "you didn", "you won", "you can t",
    "we don", "we didn", "we won",
    "they don", "they didn", "they won",
)


def filterPair(p):
    return len(p[0].split(" ")) < MAX_LENGTH and \
           len(p[1].split(" ")) < MAX_LENGTH and \
           p[1].startswith(eng_prefix)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]




if __name__ == "__main__":
    # Test examples
    test_strings = [
        "Café résumé naïve",  # Unicode with accents
        "I'm 25 years old.",  # Numbers and contractions
        "¡Hola! ¿Cómo estás?", # Spanish punctuation
        "This    has   extra     spaces",  # Multiple spaces
        "email@domain.com & phone: 123-456-7890"  # Special chars
    ]
    
    _, _, pairs = readLang("eng", "fra")
    print(f"Total sentence pairs read: {len(pairs)}")
    print(f"First 5 pairs: {pairs[:5]}")
    print("Original -> Unicode to ASCII -> Normalized")
    print("-" * 60)
    
    for text in test_strings:
        ascii_text = UnicodeToAscii(text)
        normalized = normalizeString(text)
        print(f"'{text}'")
        print(f"  ASCII: '{ascii_text}'")
        print(f"  Normalized: '{normalized}'")
        print()