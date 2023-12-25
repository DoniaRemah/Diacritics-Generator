import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers import BertTokenizer
import globals  # Import the globals.py file
import utils



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
        for char, diacritic in zip(char, diacritics):
            char_list = (char, diacritic) 
        
            result_list.append(char_list)

    word_tuple = (word,result_list)
    # list of Lists of chars with their corresponding diacritics
    globals.golden_outputs_list.append(word_tuple)

def get_words_without_diacritics(sentences):
    words_without_diacritics = []
    for sentence in sentences:
        words_without_diacritics.append(re.findall(r'[\u0600-\u06FF]+', re.sub(r'[\u064B-\u065F]', '', sentence)))
    return words_without_diacritics

# This Function converts the sentence into tokens and creates the vocabulary of words
def word_tokenize():
    #Bert Tokenizer to generate word_vocabulary data already read in cleaned_sentences
    tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')
    #loop over the cleaned sentences and tokenize them
    for sentence in globals.clean_sentences:
        sentence_tokens = tokenizer.tokenize(sentence) #Array of word tokens
        globals.tokenized_sentences.append(sentence_tokens) #List of Sentences of List of Word Tokens
        globals.word_vocabulary.update(sentence_tokens) #Set of vocabulary of word tokens
    
    utils.saveToTextFile('output/vocab.txt', globals.word_vocabulary)
    utils.SaveToPickle('output/vocab.pickle', globals.word_vocabulary)
    utils.saveToTextFile('output/tokenized_sentences.txt', globals.tokenized_sentences)
    utils.SaveToPickle('output/tokenized_sentences.pickle', globals.tokenized_sentences)


def char_tokenize():
    for sentence in globals.words_without_diacritics:
        for word in sentence:
            for char in word:
                globals.letters.add(char)   

    utils.saveToTextFile('output/char_vocabulary.txt', globals.letters)

def tokenize():

    # //////////////////////////////////////STEP1: GETTING WORDS WITHOUT DIACRITICS//////////////////////////////////////////
    ## list of list of elkalmat men 8ir tashkeel-> kol sentence, kol elklmat bt3tha
    # words_without_diacritics = get_words_without_diacritics(globals.clean_sentences)
    # utils.saveToTextFile('output/words_without_diacritics.txt', words_without_diacritics)
    # utils.SaveToPickle('output/words_without_diacritics.pickle', words_without_diacritics)
    # globals.words_without_diacritics=utils.loadPickle('output/words_without_diacritics.pickle')

    # //////////////////////////////////////STEP2: TOKENIZING WORDS AND UPDATING VOCABULARY //////////////////////////////////////////
    # TODO: UNCOMMENT WHEN TESTING
    # word_tokenize()

    # //////////////////////////////////////STEP3: Extracting Golden Output //////////////////////////////////////////
    # TODO: UNCOMMENT WHEN TESTING
    # for sentence in globals.clean_sentences:
    #     for word in sentence.split():
    #         extract_golden_output(word)       

    # //////////////////////////////////////STEP4: Tokenizing Chars //////////////////////////////////////////
    
    # letter_to_vector()
    # char_tokenize()
    pass

    


def letter_to_vector():
    # Creating one-hot vector for each char
    letters_length = len(globals.letters)
    char_vector = np.zeros(shape=(letters_length, 1))
    for i, current_letter in enumerate(globals.letters):
        char_vector = np.zeros(shape=(letters_length, 1))
        char_vector[i] = 1
        globals.letters_vector[current_letter] = char_vector
    
    utils.SaveToPickle('output/letters_vector.pickle', globals.letters_vector)
    utils.saveToTextFile('output/letters_vector.txt', globals.letters_vector)


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
