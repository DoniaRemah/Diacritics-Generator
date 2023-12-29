import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers import AutoTokenizer,AutoModel
import globals  # Import the globals.py file
import utils
import tensorflow as tf
import torch

# bert tokenizer
tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic')
model= AutoModel.from_pretrained('asafaya/bert-base-arabic')





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
            if word == '[CLS]' or word == '[SEP]':
                continue
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
    global model
    
    #loop over the cleaned sentences and tokenize them
    for sentence in globals.clean_sentences:
        result_list_of_lists=[]
        # sentence_tokens = tokenizer.tokenize(sentence, return_tensors="pt", truncation=True, max_length=400, padding=True) #Array of word token
        sentence_tokens = tokenizer(sentence, return_tensors='pt',truncation=True, max_length=400, padding=True)
        output = model(**sentence_tokens)
        # result=output.last_hidden_state.detach().numpy()
        result=output.last_hidden_state
        for i in range(result.shape[1]):
            list_of_embeddings=result[:, i].tolist()
            result_list_of_lists.append(list_of_embeddings[0])
        lenght=len(result_list_of_lists)
        length0=len(result_list_of_lists[0])
        globals.word_embeddings.append(result_list_of_lists)

        token_ids = sentence_tokens['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # utils.saveToTextFile('output/word_embeddings.txt', globals.word_embeddings)
        #insert the [CLS] and [SEP] tokens to the beginning and end of the sentence
        # sentence_tokens.insert(0,'[CLS]')
        # sentence_tokens.append('[SEP]')
        # Ensure all tokenized sentences have the same length by padding
        globals.tokenized_sentences.append(tokens) #List of Sentences of List of Word Tokens
        globals.word_vocabulary.update(tokens) #Set of vocabulary of word tokens

    # max_length = max(len(tokens) for tokens in globals.tokenized_sentences)
    # globals.padded_tokenized_sentences = [tokens + ['[PAD]'] * (max_length - len(tokens)) for tokens in globals.tokenized_sentences]
    
    # utils.saveToTextFile('output/vocab.txt', globals.word_vocabulary)
    utils.SaveToPickle('output/vocab.pickle', globals.word_vocabulary)
    # utils.saveToTextFile('output/tokenized_sentences.txt', globals.tokenized_sentences)
    utils.SaveToPickle('output/tokenized_sentences.pickle', globals.tokenized_sentences)
    utils.SaveToPickle('output/word_embeddings.pickle', globals.word_embeddings)
    # utils.SaveToPickle('output/padded_tokenized_sentences.pickle', globals.padded_tokenized_sentences)


def extract_word_embeddings():
    global tokenizer, model

    for tokenized_sentence in globals.tokenized_sentences:
        input_ids = tokenizer(tokenized_sentence, return_tensors="pt")["input_ids"]
        model_input={"input_ids":input_ids}
        outputs = model(**model_input)
        result=outputs.last_hidden_state.detach().numpy()
        globals.word_embeddings.append(result)

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
    for sentence in globals.tokenized_sentences:
        for word in sentence:
            for char in word:
                globals.letters.add(char)   

    # utils.saveToTextFile('output/char_vocabulary.txt', globals.letters)
    utils.SaveToPickle('output/char_vocabulary.pickle', globals.letters)

def tokenize():

    # # //////////////////////////////////////STEP1: GETTING WORDS WITHOUT DIACRITICS//////////////////////////////////////////
    # list of list of elkalmat men 8ir tashkeel-> kol sentence, kol elklmat bt3tha
    words_without_diacritics = get_words_without_diacritics(globals.clean_sentences)
    # utils.saveToTextFile('output/words_without_diacritics.txt', words_without_diacritics)
    utils.SaveToPickle('output/words_without_diacritics.pickle', words_without_diacritics)
    globals.words_without_diacritics=utils.loadPickle('output/words_without_diacritics.pickle')

    # # //////////////////////////////////////STEP2: TOKENIZING WORDS AND UPDATING VOCABULARY //////////////////////////////////////////
    # # TODO: UNCOMMENT WHEN TESTING
    word_tokenize()

    # la8enaha men hayatnaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    # extract_word_embeddings()

    # # //////////////////////////////////////STEP3: Extracting Golden Output //////////////////////////////////////////
    # # TODO: UNCOMMENT WHEN TESTING   
    extract_golden_output()

    # # //////////////////////////////////////STEP4: Tokenizing Chars //////////////////////////////////////////
    
    letter_to_vector()
    char_tokenize()

    # # //////////////////////////////////////STEP5: Assigning Vector to Char /////////////////////////////////////////
    assign_vector_to_char()
    # globals.word_embeddings=utils.loadPickle('output/word_embeddings.pickle')
    # utils.saveToTextFile('output/yarab.txt', globals.word_embeddings[0])
    # globals.tokenized_sentences=utils.loadPickle('output/tokenized_sentences.pickle')
    # utils.saveToTextFile('output/yarab.txt', globals.tokenized_sentences[0])
    # print(len(globals.word_embeddings[0][0]))

    

    



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

