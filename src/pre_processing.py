import numpy as np
import re
from transformers import AutoTokenizer,AutoModel
import globals  # Import the globals.py file
import utils
import tensorflow as tf

# Auto tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic')
model= AutoModel.from_pretrained('asafaya/bert-base-arabic')

def clean_data():
    text = globals.unclean_sentences
    # text = globals.test_sentences
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

    # with open('output/cleaned_train.txt', 'w',encoding='utf-8') as cleaned_train_file:
    #     for sentence in globals.clean_sentences:
    #         cleaned_train_file.write(sentence)
    # with open('output/cleaned_val.txt', 'w',encoding='utf-8') as cleaned_train_file:
    #     for sentence in globals.clean_sentences:
    #         cleaned_train_file.write(sentence)
    with open('output/cleaned_sentence.txt', 'w',encoding='utf-8') as cleaned_train_file:
        for sentence in globals.clean_sentences:
            cleaned_train_file.write(sentence)



# This Function converts the sentence into tokens, generates embeddings and creates the vocabulary of words
def word_tokenize():
    #Auto Tokenizer to generate word_vocabulary data already read in cleaned_sentences
    global tokenizer

    global model

    sentence_tokens = []

    count =0
    
    #loop over the cleaned sentences and tokenize them
    for sentence in globals.clean_sentences:
        sentence_word_embeddings=[]
        
        chunks =[]
        # tokenize the sentence
        sentence_tokens = tokenizer(sentence,return_tensors='pt') 

        # Check if tokenized sentence exceeds 400 tokens
        if sentence_tokens['input_ids'].shape[1] > 400:

            # Split into chunks of 400 tokens
            for i in range(0, sentence_tokens['input_ids'].shape[1], 400):
                chunk_end = min(i + 400, sentence_tokens['input_ids'].shape[1])  # Ensure last chunk is valid
                chunk = {key: value[:, i:chunk_end] for key, value in sentence_tokens.items()}  # Slice all tensors
                chunks.append(chunk)
        else:
            # No need to split if already within the limit
            chunks.append(sentence_tokens)

        output = []
        for chunk in chunks:
            output = model(**chunk)
            result=output.last_hidden_state
            for i in range(result.shape[1]):
                # Vector Embedding of one word
                word_embedding_vector=result[:, i].tolist()
                sentence_word_embeddings.append(word_embedding_vector[0])

        globals.word_embeddings.append(sentence_word_embeddings)

        token_ids = sentence_tokens['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        globals.tokenized_sentences.append(tokens) #List of Sentences of List of Word Tokens
        globals.word_vocabulary.update(tokens) #Set of vocabulary of word tokens
        print("sentence Number:",count)
        print("sentence Tokens Length:",len(tokens))
        count+=1
    
    #####################################Save outputs to file#################################################
    # utils.SaveToPickle('output/vocab_20000_25000.pickle', globals.word_vocabulary)
    # utils.SaveToPickle('output/tokenized_sentences_20000_25000_withoutHamza.pickle', globals.tokenized_sentences)
    # utils.SaveToPickle('output/word_embeddings_20000_25000.pickle', globals.word_embeddings)
    utils.SaveToPickle('output/vocab_sentence.pickle', globals.word_vocabulary)
    utils.SaveToPickle('output/tokenized_sentences_sentence_withoutHamza.pickle', globals.tokenized_sentences)
    utils.SaveToPickle('output/word_embeddings_sentence.pickle', globals.word_embeddings)


def extract_golden_output():
    char_counter = 0
    
    for sentence in globals.clean_sentences:
        word_tuples_list = []
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
            word_tuples_list.append(word_tuple)
            # list of Lists of chars with their corresponding diacritics
        globals.golden_outputs_list.append(word_tuples_list)
    
    utils.SaveToPickle('output/golden_outputs_val.pickle', globals.golden_outputs_list)

def letter_to_vector():
    # Creating one-hot vector for each char
    letters_length = len(globals.letters)
    char_vector = np.zeros(shape=(letters_length, 1))
    for i, current_letter in enumerate(globals.letters):
        char_vector = np.zeros(shape=(letters_length, 1))
        char_vector[i] = 1
        globals.letters_vector[current_letter] = char_vector
    
    utils.SaveToPickle('output/letters_vector.pickle', globals.letters_vector)

def get_words_without_diacritics(sentences):
    words_without_diacritics = []
    for sentence in sentences:
        words_without_diacritics.append(re.findall(r'[\u0600-\u06FF]+', re.sub(r'[\u064B-\u065F]', '', sentence)))
    return words_without_diacritics

# this function takes the corpus and assign a vector to each char, preparing it for the models
# takes as input the tokenized words from the bert tokenizer

#NOTTTEEEEEEE : need to skip the [CLS] and [SEP] tokens elly homa <s> w </s> 

def sub_tokenized_letters():

    words_without_diac = get_words_without_diacritics(globals.clean_sentences)

    for sent_index,sentence in enumerate(words_without_diac):
        current_tokenized_word_index = 0
        current_tokenized_sentence= globals.tokenized_sentences[sent_index]
        for word in sentence:
            word_length=len(word)
            processed_chars=0
            current_char_index=0
            current_tokenized_char_index=0
            while processed_chars< word_length:
                if current_tokenized_word_index >= len(current_tokenized_sentence):
                    print("Word index in sentence out of bounds")
                    break

                if current_tokenized_char_index >= len(current_tokenized_sentence[current_tokenized_word_index]):
                    current_tokenized_word_index += 1
                    current_tokenized_char_index = 0
                    continue

                current_tokenized_word = current_tokenized_sentence[current_tokenized_word_index]
                if current_tokenized_word == '[CLS]' or current_tokenized_word == '[SEP]':
                    current_tokenized_word_index += 1
                    continue
                if word[current_char_index] == current_tokenized_word[current_tokenized_char_index]:
                    current_char_index += 1
                    current_tokenized_char_index += 1
                    processed_chars += 1

                elif current_tokenized_word[current_tokenized_char_index] == '#':
                    current_tokenized_char_index += 1
                    continue
                elif word[current_char_index] != current_tokenized_word[current_tokenized_char_index]:

                    # Modify the char in the string
                    
                    word_list = list(globals.tokenized_sentences[sent_index][current_tokenized_word_index])
                    word_list[current_tokenized_char_index] =  word[current_char_index]
                    globals.tokenized_sentences[sent_index][current_tokenized_word_index] = ''.join(word_list)
                    current_char_index += 1
                    current_tokenized_char_index += 1
                    processed_chars += 1

            current_tokenized_word_index += 1
    utils.SaveToPickle('output/tokenized_sentences_sentence_withHamza.pickle',globals.tokenized_sentences)                



def assign_vector_to_char():
    char_counter = 0
    for sentence in globals.tokenized_sentences:
        sentence_vector_char = []
        for word in sentence:
            # check for the [CLS] and [SEP] tokens and skip them
            if word == '[CLS]' or word == '[SEP]':
                word_tuple=None
                sentence_vector_char.append(word_tuple)
                continue
            result_list = []
            for char in word:
                if char != '#':
                    char_list = (char_counter,globals.letters_vector.get(char))
                    char_counter += 1
                else:
                    char_list = (-1,globals.letters_vector.get(char))
                    
                result_list.append(char_list)
            word_tuple = (word,result_list)
            sentence_vector_char.append(word_tuple)
        globals.char_embeddings.append(sentence_vector_char)
    
    utils.SaveToPickle('output/char_embeddings_sentence.pickle', globals.char_embeddings)


def get_char_embeddings():

    letter_to_vector()
    sub_tokenized_letters()
    assign_vector_to_char()



def tokenize():

    # # ////////////////////////////////////// TOKENIZING WORDS, GENERATE WORD EMBEDDINGS AND UPDATING VOCABULARY //////////////////////////////////////////
    word_tokenize()

    print("finished tokenizing words")


    # # //////////////////////////////////////Extracting Golden Output //////////////////////////////////////////
    # not in test
    # extract_golden_output() 

    # print("finished extracting golden output")

    # # # //////////////////////////////////////One hot vector for every char in vocab //////////////////////////////////////////
    
    get_char_embeddings()

    print("finished tokenizing chars")

    # # # //////////////////////////////////////Assigning Vector to every char in the corpus /////////////////////////////////////////



def pre_processing():
    # clean_data()
    tokenize()

