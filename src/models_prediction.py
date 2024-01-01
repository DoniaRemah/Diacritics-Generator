import utils
import numpy as np

def load_data_for_prediction():

    # Load char embeddings
    globals.char_embeddings = utils.load_pickle_file("char_embeddings")
    # Load word embeddings
    globals.word_embeddings = utils.load_pickle_file("word_embeddings")
    

def extract_chars():

    for sentence in globals.char_embeddings:
        words_per_sentence = []
        chars_index_per_sentence=[]
        for word in sentence:
            chars_vectors_per_word = []
            chars_index_per_word = []
            chars_tuple_list = word[1]

            for char_tuple in chars_tuple_list:
                char_index_in_corpus = char_tuple[0]
                char_vector = char_tuple[1]

                chars_vectors_per_word.append(char_vector)
                chars_index_per_word.append(char_index_in_corpus)

            chars_index_per_sentence.append(chars_index_per_word)
            words_per_sentence.append(chars_vectors_per_word)

        globals.model_chars_index_per_corpus.append(chars_index_per_sentence)
        globals.model_char_embeddings.append(words_per_sentence)



def pad_word_embeddings():

    max_sentence_length=400
    # Pad words to the maximum length
    padded_word_embeddings = []
    for sentence in globals.word_embeddings:
        # remove the first and last token cls and sep
        sentence = sentence[1:-1]
        padded_sentence = np.pad(sentence, ((0, max_sentence_length - len(sentence)), (0, 0)), mode='constant')
        padded_word_embeddings.append(padded_sentence)

    globals.padded_word_embeddings = padded_word_embeddings


def pad_char_embeddings():

    # Define your padding vector
    char_padding_vector = np.zeros((37,1))

    max_sentence_length=400
   
    max_word_length=15

    padded_char_embeddings = []
    for sentence in globals.model_char_embeddings:
        new_sentence = []

        for word in sentence:

            # Pad the word to the max_word_length
            word = word + [char_padding_vector] * (max_word_length - len(word))
            new_sentence.append(word)
        
        empty_word = [char_padding_vector] * max_word_length
        new_sentence = new_sentence + [empty_word] * (max_sentence_length - len(sentence))
        padded_char_embeddings.append(new_sentence)

    globals.padded_char_embeddings = padded_char_embeddings

def predict():

    predictions = globals.our_model.predict([np.array(globals.padded_word_embeddings),np.array(globals.padded_char_embeddings)])
    print(predictions)