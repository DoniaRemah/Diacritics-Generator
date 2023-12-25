import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer
import tensorflow as tf


def extract_word_embeddings(tokenized_sentence):
    tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
    ids=[]
    # Convert tokens to IDs
    for word in tokenized_sentence:
        words_ids = tokenizer.encode(word)

    # Convert to TensorFlow tensor
    # words_ids_tensor_vectors = tf.constant([words_ids])
        ids.append(words_ids)
    return ids

    # for each sentence, a list of words, for each word, an embedding vector
trial=['او', 'قطع', 'الاول', 'يده', 'الخ', 'قال', 'الزر', '##كش', '##ي']
print(extract_word_embeddings(trial))
