import utils
import numpy as np
import globals

def load_data_for_prediction():

    # Load char embeddings
    globals.char_embeddings = utils.load_pickle_file("char_embeddings")
    # Load word embeddings
    globals.word_embeddings = utils.load_pickle_file("word_embeddings")
    

def extract_chars():

    for sentence in globals.char_embeddings:
        words_per_sentence = []
        chars_index_per_sentence=[]
        for word in sentence[1:-1]:
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
    utils.SaveToPickle('output/competition/model/model_char_embeddings_test.pickle', globals.model_char_embeddings)
    utils.SaveToPickle('output/competition/model/model_chars_index_per_corpus_test.pickle', globals.model_chars_index_per_corpus)

def chunk_data():
    # Chunk word_embeddings and model_char_embeddings, where each chunk has max size 400

    # Chunk word_embeddings
    chunked_word_embeddings = []  
    chunked_model_indices = []   
    chunked_model_char_embeddings = []

    # Chunking Sentences
    for sentence in globals.word_embeddings:
        # remove the first and last token cls and sep
        sentence = sentence[1:-1]
        if len(sentence) > 400:
            for i in range(0, len(sentence), 400):
                chunk_end = min(i + 400, len(sentence))
                chunked_word_embeddings.append(sentence[i:chunk_end])
        else:
            chunked_word_embeddings.append(sentence)

    print("chunked word embeddings")

    # Chunking Labels
    for sentence in globals.model_chars_index_per_corpus:
        if len(sentence) > 400:
            for i in range(0, len(sentence), 400):
                chunk_end = min(i + 400, len(sentence))
                chunked_model_indices.append(sentence[i:chunk_end])
        else:
            chunked_model_indices.append(sentence)

    print("chunked model labels")


    # Chunking Char Embeddings
    for sentence in globals.model_char_embeddings:
        if len(sentence) > 400:
            for i in range(0, len(sentence), 400):
                chunk_end = min(i + 400, len(sentence))
                chunked_model_char_embeddings.append(sentence[i:chunk_end])
        else:
            chunked_model_char_embeddings.append(sentence)

    print("chunked model char embeddings")

    globals.chunked_word_embeddings = chunked_word_embeddings
    globals.chunked_char_embeddings = chunked_model_char_embeddings
    globals.chunked_model_indices = chunked_model_indices
    utils.SaveToPickle('output/competition/model/chunked_model_indices_test.pickle',globals.chunked_model_indices)



def pad_word_embeddings():

    max_sentence_length=400
    # Pad words to the maximum length
    padded_word_embeddings = []
    for sentence in globals.chunked_word_embeddings:
        padded_sentence = np.pad(sentence, ((0, max_sentence_length - len(sentence)), (0, 0)), mode='constant')
        padded_word_embeddings.append(padded_sentence)

    globals.padded_word_embeddings = padded_word_embeddings
    utils.SaveToPickle('output/competition/model/padded_word_embeddings_test.pickle', globals.padded_word_embeddings)


def pad_char_embeddings():

    # Define your padding vector
    char_padding_vector = np.zeros((37,1))

    max_sentence_length=400
   
    max_word_length=15

    padded_char_embeddings = []
    for sentence in globals.chunked_char_embeddings:
        new_sentence = []

        for word in sentence:


            # Pad the word to the max_word_length
            word = word + [char_padding_vector] * (max_word_length - len(word))
            new_sentence.append(word)
        
        empty_word = [char_padding_vector] * max_word_length
        new_sentence = new_sentence + [empty_word] * (max_sentence_length - len(sentence))
        padded_char_embeddings.append(new_sentence)

    globals.padded_char_embeddings = padded_char_embeddings
    utils.SaveToPickle('output/competition/model/padded_char_embeddings_test.pickle', globals.padded_char_embeddings)

def pad_indices():
    index_padding = -1
    max_sentence_length=400

    max_word_length=15

    padded_indices = []
    for sentence in globals.chunked_model_indices:
        new_sentence = []

        for word in sentence:
            # Pad the word to the max_word_length
            word = word + [index_padding] * (max_word_length - len(word))
            new_sentence.append(word)
        
        empty_word = [index_padding] * max_word_length
        new_sentence = new_sentence + [empty_word] * (max_sentence_length - len(sentence))
        padded_indices.append(new_sentence)

    globals.padded_indices = padded_indices
    utils.SaveToPickle('output/competition/model/padded_indices_test.pickle', globals.padded_indices)


def prepare_data():
    globals.word_embeddings = utils.loadPickle('output/competition/word_embeddings_test.pickle')
    print("finished loading word embeddings")
    globals.model_char_embeddings = utils.loadPickle('output/competition/model/model_char_embeddings_test.pickle')
    globals.model_chars_index_per_corpus = utils.loadPickle('output/competition/model/model_chars_index_per_corpus_test.pickle')
    chunk_data()
    print("finished chunking data")
    pad_word_embeddings()
    print("finished padding word embeddings")
    pad_char_embeddings()
    print("finished padding char embeddings")
    pad_indices()
    print("finished padding indices")

def load_padded_data():
    globals.padded_word_embeddings=utils.loadPickle('output/competition/model/padded_word_embeddings_test.pickle')
    globals.padded_char_embeddings=utils.loadPickle('output/competition/model/padded_char_embeddings_test.pickle')
    globals.padded_indices=utils.loadPickle('output/competition/model/padded_indices_test.pickle')

    globals.word_embeddings_numpy = np.array(globals.padded_word_embeddings)
    globals.char_embeddings_numpy = np.array(globals.padded_char_embeddings)
    
def predict():

    predictions = globals.our_model.predict([globals.word_embeddings_numpy,globals.char_embeddings_numpy])

    for sentence in predictions:
        predicted_labels_per_sentence = []
        for word in sentence:
            predicted_labels_per_word = []
            for char in word:
                # take softmax for char and get the index of the max value
                softmax = np.exp(char) / np.sum(np.exp(char), axis=0)
                predicted_label = np.argmax(softmax)
                predicted_labels_per_word.append(predicted_label)
            predicted_labels_per_sentence.append(predicted_labels_per_word)
        globals.predicted_labels.append(predicted_labels_per_sentence)

    print("prediction done")
    # print(predictions)

def save_output():
    # Save the output in csv file
    with open('output/competition/predicted_labels_test.csv', 'w', encoding='utf-8') as f:
        f.write('ID,label\n')
        for index1,sentence in enumerate(globals.padded_indices):
            for index2,word in enumerate(sentence):
                for index3,char in enumerate(word):
                    if char != -1:
                        f.write(str(char) + ',' + str(globals.predicted_labels[index1][index2][index3]) + '\n')
                    else:
                        continue