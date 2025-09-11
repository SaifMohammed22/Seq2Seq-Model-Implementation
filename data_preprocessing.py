from lang import Lang
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
    s = re.sub(r"([.!?])", r"\1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
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

eng_prefix = (
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "we are", "we re",
    "they are", "they re",
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
    
    print(readLang("eng", "fra"))
    print("Original -> Unicode to ASCII -> Normalized")
    print("-" * 60)
    
    for text in test_strings:
        ascii_text = UnicodeToAscii(text)
        normalized = normalizeString(text)
        print(f"'{text}'")
        print(f"  ASCII: '{ascii_text}'")
        print(f"  Normalized: '{normalized}'")
        print()