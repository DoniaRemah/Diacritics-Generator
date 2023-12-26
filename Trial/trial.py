# import pandas as pd
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# import re
# from transformers import BertTokenizer
# import tensorflow as tf


# def extract_word_embeddings(tokenized_sentence):
#     tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
#     ids=[]
#     # Convert tokens to IDs
#     for word in tokenized_sentence:
#         words_ids = tokenizer.encode(word)

#     # Convert to TensorFlow tensor
#     # words_ids_tensor_vectors = tf.constant([words_ids])
#         ids.append(words_ids)
#     return ids

#     # for each sentence, a list of words, for each word, an embedding vector
# trial=['او', 'قطع', 'الاول', 'يده', 'الخ', 'قال', 'الزر', '##كش', '##ي']
# print(extract_word_embeddings(trial))

from transformers import BertModel, BertTokenizer
import torch

def trial_embbeding():
    # # Load pre-trained model and tokenizer
    model_name = 'asafaya/bert-base-arabic'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # # Tokenize input sentence
    # input_sentence = "قال احمد لا"
    # tokenized_input = tokenizer(input_sentence, return_tensors="pt")

    # print(tokenized_input)

    # # Forward pass through the model
    # with torch.no_grad():
    #     outputs = model(**tokenized_input)

    # # Extract embeddings from the output
    # word_embeddings = outputs.last_hidden_state

    # # word_embeddings now contains the embeddings for each token in the input sentence
    # print(word_embeddings)  # Shape: (batch_size, sequence_length, hidden_size)


    # sentence = "Hello, my dog is cute."

    # # Tokenize the sentence into words
    # tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))

    tokens = ['او', 'قطع', 'الاول', 'يده', 'الخ', 'قال', 'الزر', '##كش', '##ي']

    print("Tokens:", tokens)

    # Get contextual embeddings for each word
    word_embeddings = []
    for token in tokens:
        input_ids = tokenizer.encode(token, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids)
        # Extract embeddings for the first token (CLS token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        word_embeddings.append(cls_embedding)

    # Concatenate the word embeddings along the first dimension to get a tensor
    word_embeddings = torch.cat(word_embeddings, dim=0)

    # print("Word Embeddings:", word_embeddings)

    for i in range(len(tokens)):
        print("Token:", tokens[i])
        # print("Embedding:", word_embeddings[i])
        print("Embedding shape:", word_embeddings[i].shape)

trial_embbeding()