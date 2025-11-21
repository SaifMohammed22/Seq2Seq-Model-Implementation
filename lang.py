SOS_TOKEN = 0
EOS_TOKEN = 1 

class Lang():
    def __init__(self, lang_name):
        self.name = lang_name
        self.word2idx = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.idx2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1

        else:
            self.word2count[word] += 1


if __name__ == "__main__":
    lang = Lang("English")
    sentence = "Hello Hello world"
    lang.addSentence(sentence)
    print(lang.name)
    print(lang.word2idx)
    print(lang.word2count)
    print(lang.idx2word)



