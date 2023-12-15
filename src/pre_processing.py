import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer
import globals  # Import the globals.py file


def extract_golden_output(word: str):
    matches = re.finditer(r'([\u0621-\u064a])([\u064b-\u0652]*)', word)

    # List of tuples of characters and their diacritics
    result_list = []

    # Iterate over matches
    for match in matches:
        char = match.group(1)
        diacritics = match.group(2)
        # write to a file
        with open('dataset/test_diacritics.txt', 'a',encoding='utf-8') as diacritics_file:
            diacritics_file.write(char + '\t' + diacritics + '\n')
        # Create a list of lists with each character and its associated diacritic
        char_list = [(char, diacritic) for char, diacritic in zip(char, diacritics)]
        
        result_list.append(char_list)

    word_tuple = (word,result_list)
    # list of Lists of chars with their corresponding diacritics
    globals.golden_outputs_list.append(word_tuple)

def get_words_without_diacritics(sentence: str):
    words_without_diacritics = re.findall(r'[\u0600-\u06FF]+', re.sub(r'[\u064B-\u065F]', '', sentence))
    return words_without_diacritics

# This Function converts the sentence into tokens and creates the vocabulary of words
def word_tokenize():
    #Bert Tokenizer to generate word_vocabulary data already read in cleaned_sentences
    tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
    #loop over the cleaned sentences and tokenize them
    for sentence in globals.clean_sentences:
        # TODO: EXTRACT DIACRITIC FROM SENTENCE
        sentence_tokens = tokenizer.tokenize(sentence) #Array of word tokens
        globals.tokenized_sentences.append(sentence_tokens) #List of Sentences of List of Word Tokens
        globals.word_vocabulary.update(sentence_tokens) #Set of vocabulary of word tokens
    
    print("Tokenized Sentences", globals.tokenized_sentences[0])
    with open('dataset/vocab.txt', 'w',encoding='utf-8') as vocab_file:
        for vocab_word in globals.word_vocabulary:
            vocab_file.write(vocab_word + '\n')


def tokenize():
    # words_without_diacritics = get_words_without_diacritics(globals.clean_sentences)
    # word_tokenize()
    for sentence in globals.clean_sentences:
        for word in sentence.split():
            extract_golden_output(word)

    #write to file globals.golden_outputs_list
    with open('dataset/golden_outputs.txt', 'w',encoding='utf-8') as golden_outputs_file:
        for word_tuple in globals.golden_outputs_list:
            golden_outputs_file.write(word_tuple[0] + '\t' + word_tuple[1] + '\n')


def char_tokenize():
    pass

def clean_data():
    text = globals.unclean_sentences
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
    tokenize()
