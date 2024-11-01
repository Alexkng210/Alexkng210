import nltk

def tokenize_text(text):
    return nltk.word_tokenize(text)

def pos_tag(tokens):
    return nltk.pos_tag(tokens)
