import tensorflow as tf
import unicodedata
import re
import io


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentance(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[""]+', " ", w)
    
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    w = '<start> ' + w + ' <end>'

    return w

def create_dataset(path, num_examples):
    #if (raw_text == None):
    lines = io.open(path, encoding='UTF-8').read().split('\n')
    #else:
    #    lines = raw_text.split('\n')

    word_pairs = [[preprocess_sentance(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)
    

def tokenize (lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    targ, inp = create_dataset(path, num_examples)

    input_tensor, inp_tokenizer = tokenize(inp)
    target_tensor, targ_tokenizer = tokenize(targ)

    return input_tensor, target_tensor, inp_tokenizer, targ_tokenizer
