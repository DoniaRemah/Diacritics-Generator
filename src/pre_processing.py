import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer, BertModel
import globals  # Import the globals.py file
import utils
import tensorflow as tf
import torch

tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
model = BertModel.from_pretrained('asafaya/bert-base-arabic')



def extract_golden_output():
    char_counter = 0
    for sentence in globals.clean_sentences:
        for word in sentence.split():
            matches = re.finditer(r'([\u0621-\u064a])([\u064b-\u0652]*)', word)

            # List of tuples of characters and their diacritics
            result_list = []
            # Iterate over matches
            for match in matches:
                char = match.group(1)
                diacritic = match.group(2)
                # Create a list of lists with each character and its associated diacritic
                char_list = (char_counter,globals.diacritics_ids.get(diacritic))
                char_counter += 1
                
                result_list.append(char_list)

            word_tuple = (word,result_list)
            # list of Lists of chars with their corresponding diacritics
            globals.golden_outputs_list.append(word_tuple)
    
    # utils.saveToTextFile('output/golden_outputs.txt', globals.golden_outputs_list)
    utils.SaveToPickle('output/golden_outputs.pickle', globals.golden_outputs_list)


def get_words_without_diacritics(sentences):
    words_without_diacritics = []
    for sentence in sentences:
        words_without_diacritics.append(re.findall(r'[\u0600-\u06FF]+', re.sub(r'[\u064B-\u065F]', '', sentence)))
    return words_without_diacritics


# this function takes the corpus and assign a vector to each char, preparing it for the models
# takes as input the tokenized words from the bert tokenizer

#NOTTTEEEEEEE : need to skip the [CLS] and [SEP] tokens elly homa <s> w </s> 

def assign_vector_to_char():
    char_counter = 0
    for sentence in globals.tokenized_sentences:
        sentence_vector_char = []
        # check for the [CLS] and [SEP] tokens and skip them
        for word in sentence:
            # check for the [CLS] and [SEP] tokens and skip them
            # if word == '[CLS]' or word == '[SEP]':
            #     continue
            result_list = []
            for char in word:
                char_list = (char_counter,globals.letters_vector.get(char))
                char_counter += 1
                result_list.append(char_list)

            word_tuple = (word,result_list)
            sentence_vector_char.append(word_tuple)
        globals.char_embeddings.append(sentence_vector_char)
    
    # utils.saveToTextFile('output/char_embeddings.txt', globals.char_embeddings)
    utils.SaveToPickle('output/char_embeddings.pickle', globals.char_embeddings)


# This Function converts the sentence into tokens and creates the vocabulary of words
def word_tokenize():
    #Bert Tokenizer to generate word_vocabulary data already read in cleaned_sentences
    global tokenizer
    #loop over the cleaned sentences and tokenize them
    for sentence in globals.clean_sentences:
        sentence_tokens = tokenizer.tokenize(sentence) #Array of word tokens
        #insert the [CLS] and [SEP] tokens to the beginning and end of the sentence
        # sentence_tokens.insert(0,'[CLS]')
        # sentence_tokens.append('[SEP]')
        globals.tokenized_sentences.append(sentence_tokens) #List of Sentences of List of Word Tokens
        globals.word_vocabulary.update(sentence_tokens) #Set of vocabulary of word tokens

    
    
    # utils.saveToTextFile('output/vocab.txt', globals.word_vocabulary)
    utils.SaveToPickle('output/vocab.pickle', globals.word_vocabulary)
    # utils.saveToTextFile('output/tokenized_sentences.txt', globals.tokenized_sentences)
    utils.SaveToPickle('output/tokenized_sentences.pickle', globals.tokenized_sentences)


def extract_word_embeddings():
    global tokenizer, model

    for tokenized_sentence in globals.tokenized_sentences:
        word_embeddings = []
        for token in tokenized_sentence:
            input_ids = tokenizer.encode(token, return_tensors="pt")
            with torch.no_grad():
                outputs = model(input_ids)
            # Extract embeddings for the first token (CLS token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            word_embeddings.append(cls_embedding)
        
        # Concatenate the word embeddings along the first dimension to get a tensor
        word_embeddings = torch.cat(word_embeddings, dim=0)
        globals.word_embeddings.append(word_embeddings)

    # utils.saveToTextFile('output/word_embeddings.txt', globals.word_embeddings)
    utils.SaveToPickle('output/word_embeddings.pickle', globals.word_embeddings)


def letter_to_vector():
    # Creating one-hot vector for each char
    letters_length = len(globals.letters)
    char_vector = np.zeros(shape=(letters_length, 1))
    for i, current_letter in enumerate(globals.letters):
        char_vector = np.zeros(shape=(letters_length, 1))
        char_vector[i] = 1
        globals.letters_vector[current_letter] = char_vector
    
    utils.SaveToPickle('output/letters_vector.pickle', globals.letters_vector)

def char_tokenize():
    for sentence in globals.words_without_diacritics:
        for word in sentence:
            for char in word:
                globals.letters.add(char)   

    # utils.saveToTextFile('output/char_vocabulary.txt', globals.letters)
    utils.SaveToPickle('output/char_vocabulary.pickle', globals.letters)

def tokenize():

    # //////////////////////////////////////STEP1: GETTING WORDS WITHOUT DIACRITICS//////////////////////////////////////////
    # list of list of elkalmat men 8ir tashkeel-> kol sentence, kol elklmat bt3tha
    words_without_diacritics = get_words_without_diacritics(globals.clean_sentences)
    # utils.saveToTextFile('output/words_without_diacritics.txt', words_without_diacritics)
    utils.SaveToPickle('output/words_without_diacritics.pickle', words_without_diacritics)
    globals.words_without_diacritics=utils.loadPickle('output/words_without_diacritics.pickle')

    # //////////////////////////////////////STEP2: TOKENIZING WORDS AND UPDATING VOCABULARY //////////////////////////////////////////
    # TODO: UNCOMMENT WHEN TESTING
    word_tokenize()
    extract_word_embeddings()

    # //////////////////////////////////////STEP3: Extracting Golden Output //////////////////////////////////////////
    # TODO: UNCOMMENT WHEN TESTING   
    extract_golden_output()

    # //////////////////////////////////////STEP4: Tokenizing Chars //////////////////////////////////////////
    
    letter_to_vector()
    char_tokenize()

    # //////////////////////////////////////STEP5: Assigning Vector to Char /////////////////////////////////////////
    assign_vector_to_char()
    # globals.char_embeddings=utils.loadPickle('output/char_embeddings.pickle')
    # utils.saveToTextFile('output/yarab.txt', globals.char_embeddings[0])
    



def clean_data():
    text = globals.unclean_sentences
    clean_sentences =[]
    for sentence in text:
        
        cleaned_sentence = re.sub(r'[^\u0600-\u06FF\u064B-\u065F\u0670\s]', '', sentence)
        clean_sentences.append(re.sub(r'[،؛؟]', '', cleaned_sentence))
        # \u0600-\u06FF: Arabic characters range.
        # \u064B-\u065F: Arabic diacritic marks range.
        # \u0670: Arabic vowel length mark.
        # \s: Whitespace characters.
        # 0-9: Numbers.
        # ،؛: Arabic symbols for comma (،) and semicolon (؛).

    globals.clean_sentences = clean_sentences


    ############################################ write the clean_sentences to file ############################################

    with open('output/cleaned_train.txt', 'w',encoding='utf-8') as cleaned_train_file:
        for sentence in globals.clean_sentences:
            cleaned_train_file.write(sentence)
        
        
            
    
    

def pre_processing():
    # clean_data()
    tokenize()

