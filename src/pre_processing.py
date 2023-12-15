import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer
import globals  # Import the globals.py file


def extract_golden_output():
    pass



def word_tokenize():
    #Bert Tokenizer to generate word_vocabulary data already read in cleaned_sentences
    tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
    #loop over the cleaned sentences and tokenize them
    for sentence in globals.clean_sentences:
        sentence_tokens = tokenizer.tokenize(sentence) #Array of word tokens
        globals.tokenized_sentences.append(sentence_tokens) #List of Sentences of List of Word Tokens
        globals.word_vocabulary.update(sentence_tokens) #Set of vocabulary of word tokens
    
    print("Tokenized Sentences", globals.tokenized_sentences)
    with open('dataset/vocab.txt', 'w',encoding='utf-8') as vocab_file:
        for vocab_word in globals.word_vocabulary:
            vocab_file.write(vocab_word + '\n')

def tokenize():
    pass

def char_tokenize():
    pass

def clean_data():
    text = globals.sentences
    clean_sentences =[]
    for sentence in text:
        
        cleaned_sentence = re.sub(r'[^\u0600-\u06FF\u064B-\u065F\u0670\s]', '', sentence)
        clean_sentences.append(re.sub(r'[،؛]', '', cleaned_sentence))
        # \u0600-\u06FF: Arabic characters range.
        # \u064B-\u065F: Arabic diacritic marks range.
        # \u0670: Arabic vowel length mark.
        # \s: Whitespace characters.
        # 0-9: Numbers.
        # ،؛: Arabic symbols for comma (،) and semicolon (؛).

    globals.clean_sentences = clean_sentences
    
    #TODO: write the clean_sentences to general file
    with open('dataset/cleaned_train.txt', 'w',encoding='utf-8') as clean_train_file:
        for sentence in clean_sentences:
            clean_train_file.write(sentence)
    

def pre_processing():
    # clean_data()
    word_tokenize()
