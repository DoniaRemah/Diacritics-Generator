from keras.models import Model
from keras.layers import LSTM, Input, concatenate,Bidirectional,Masking,Dense,Reshape
import numpy as np
from sklearn.svm import SVC
import globals
import utils
import tensorflow as tf
from keras.models import load_model
import re
def load_data_for_extraction():

    globals.char_embeddings = utils.loadPickle('output/char_embeddings_0_2224.pickle')
    print("finished loading char embeddings")
    globals.golden_outputs_list = utils.loadPickle('output/golden_outputs.pickle')
    print("finished loading golden outputs")
    globals.tokenized_sentences = utils.loadPickle('output/tokenized_sentences_0_2224.pickle')
    print("finished loading tokenized sentences")



def load_data_for_model_creation():

    globals.word_embeddings = utils.loadPickle('output/word_embeddings_0_2224.pickle')
    print("finished loading word embeddings")
    globals.model_char_embeddings = utils.loadPickle('output/model/model_char_embeddings.pickle')


def load_data_for_training():
    # globals.word_embeddings = utils.loadPickle('output/word_embeddings_0_2224.pickle')
    # print("finished loading word embeddings")
    # globals.model_char_embeddings = utils.loadPickle('output/model/model_char_embeddings.pickle')
    # print("finished loading model char embeddings")
    globals.word_embeddings_numpy = utils.loadPickle('output/model/word_embeddings_numpy_0_2224.pickle')
    globals.char_embeddings_numpy = utils.loadPickle('output/model/char_embeddings_numpy_0_2224.pickle')
    globals.model_labels = utils.loadPickle('output/model/model_labels.pickle')
    print("finished loading model labels")

def load_saved_model(model_name):
    name = "models/"+model_name
    globals.our_model = load_model(name)

def extract_char_embeddings_and_labels():
    for sentece_index, sentence in enumerate(globals.tokenized_sentences):
        chars_per_sentence_list = []
        chars_index_per_sentence_list = []
        labels_per_sentence_list = []

        for word_index, word in enumerate(sentence):
            chars_per_word_list = []
            chars_index_per_word_list = []
            labels_per_word_list = []
            if word == '[CLS]' or word == '[SEP]':
                continue    

            word,chars = globals.char_embeddings[sentece_index][word_index-1]
            _,char_labels = globals.golden_outputs_list[sentece_index][word_index-1]

            for char_index, char in enumerate(chars):
                char_index_in_corpus,char_vector = char
                the_char,label = char_labels[char_index]
                chars_per_word_list.append(char_vector)
                chars_index_per_word_list.append(char_index_in_corpus)
                labels_per_word_list.append(label)

            chars_per_sentence_list.append(chars_per_word_list)
            chars_index_per_sentence_list.append(chars_index_per_word_list)
            labels_per_sentence_list.append(labels_per_word_list)

        globals.model_char_embeddings.append(chars_per_sentence_list)
        globals.model_chars_index_per_corpus.append(chars_index_per_sentence_list)
        globals.model_labels.append(labels_per_sentence_list)
        print("EXTRACTION - Finished sentence number ", sentece_index)

    utils.SaveToPickle('output/model/model_char_embeddings.pickle', globals.model_char_embeddings)
    utils.SaveToPickle('output/model/model_chars_index_per_corpus.pickle', globals.model_chars_index_per_corpus)
    utils.SaveToPickle('output/model/model_labels.pickle', globals.model_labels)


def extract_char_embeddings_and_labels_align_labels():
    for sentece_index, sentence in enumerate(globals.golden_outputs_list[0:2225]):
        chars_per_sentence_list = []
        chars_index_per_sentence_list = []
        labels_per_sentence_list = []
        current_tokenized_sentence = globals.tokenized_sentences[sentece_index]
        current_tokenized_word_index=0

        # word in golden outputs
        for word_index, golden_word in enumerate(sentence):

            word_with_diacritic=golden_word[0]

            word_without_diacritics = re.sub(r'[\u064B-\u065F]', '', word_with_diacritic)

            chars_per_word_list = []
            chars_index_per_word_list = []
            labels_per_word_list = []

            word_char_count = len(golden_word[1])
            labelled_char_count = 0   
            current_golden_char_index = 0
            current_tokenized_char_index=0
            # loop over tokenized sentences with the same index as the golden outputs
            
            while labelled_char_count < word_char_count:

                current_tokenized_word=current_tokenized_sentence[current_tokenized_word_index]
                # Tokenized word is done
                if current_tokenized_word == '[CLS]' or current_tokenized_word == '[SEP]':
                    current_tokenized_word_index+=1
                    continue

                if current_tokenized_char_index == len(current_tokenized_sentence[current_tokenized_word_index]):
                    chars_per_sentence_list.append(chars_per_word_list)
                    chars_index_per_sentence_list.append(chars_index_per_word_list)
                    labels_per_sentence_list.append(labels_per_word_list)
                    chars_per_word_list = []
                    chars_index_per_word_list = []
                    labels_per_word_list = []
                    current_tokenized_word_index+=1
                    current_tokenized_char_index=0
                    continue
                
                if word_without_diacritics[current_golden_char_index] == current_tokenized_word[current_tokenized_char_index]:
                    char_index_in_corpus,char_vector = globals.char_embeddings[sentece_index][current_tokenized_word_index][1][current_tokenized_char_index]
                    _,label = golden_word[1][current_golden_char_index]
                    chars_index_per_word_list.append(char_index_in_corpus)
                    chars_per_word_list.append(char_vector)
                    labels_per_word_list.append(label)
                    current_golden_char_index+=1
                    current_tokenized_char_index+=1
                    labelled_char_count+=1
                    continue
                
                if current_tokenized_word[current_tokenized_char_index] == '#':
                    char_index_in_corpus,char_vector = globals.char_embeddings[sentece_index][current_tokenized_word_index][1][current_tokenized_char_index]
                    chars_per_word_list.append(char_vector)
                    chars_index_per_word_list.append(char_index_in_corpus)
                    labels_per_word_list.append(14)
                    current_tokenized_char_index+=1
                    continue
                
            # AFTER A WORD HAS ENDED 
            current_tokenized_word_index+=1    
            chars_per_sentence_list.append(chars_per_word_list)
            chars_index_per_sentence_list.append(chars_index_per_word_list)
            labels_per_sentence_list.append(labels_per_word_list)


        # AFTER A SENTENCE HAS ENDED
        globals.model_char_embeddings.append(chars_per_sentence_list)
        globals.model_chars_index_per_corpus.append(chars_index_per_sentence_list)
        globals.model_labels.append(labels_per_sentence_list)
        print("EXTRACTION - Finished sentence number ", sentece_index)

    utils.SaveToPickle('output/model/model_char_embeddings.pickle', globals.model_char_embeddings)
    utils.SaveToPickle('output/model/model_chars_index_per_corpus.pickle', globals.model_chars_index_per_corpus)
    utils.SaveToPickle('output/model/model_labels.pickle', globals.model_labels)





def create_model():

    # Determine the maximum sequence lengths for words and characters
    max_sentence_length = max(len(seq) for seq in globals.word_embeddings) -2

    max_word_length = max(max(len(word) for word in sentence) for sentence in globals.model_char_embeddings)

        # Pad words to the maximum length
    padded_word_embeddings = []
    for sentence in globals.word_embeddings:
        # remove the first and last token cls and sep
        sentence = sentence[1:-1]
        padded_sentence = np.pad(sentence, ((0, max_sentence_length - len(sentence)), (0, 0)), mode='constant')
        padded_word_embeddings.append(padded_sentence)

    # Define your padding vector
    char_padding_vector = np.zeros((37,1))

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


    globals.word_embeddings_numpy = np.array(padded_word_embeddings)
    globals.char_embeddings_numpy = np.array(padded_char_embeddings)

    utils.SaveToPickle('output/model/word_embeddings_numpy_0_2224.pickle', globals.word_embeddings_numpy)
    utils.SaveToPickle('output/model/char_embeddings_numpy_0_2224.pickle', globals.char_embeddings_numpy)

    # Shape information based on the precomputed embeddings
    num_sentences, max_sentence_length, word_embedding_dim = globals.word_embeddings_numpy.shape
    max_word_length, char_embedding_dim = globals.char_embeddings_numpy.shape[2:4]

    # LSTM units
    lstm_units = 64

    # Word LSTM model with return_sequences=True
    word_input = Input(shape=(max_sentence_length,word_embedding_dim),batch_size=num_sentences)
    word_input_masked = Masking(mask_value=0.0)(word_input)
    word_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(word_input_masked)

    # Character LSTM model with input directly from word-level LSTM
    char_input = Input(shape=(max_sentence_length,max_word_length ,char_embedding_dim),batch_size=num_sentences)
    char_input_masked = Masking(mask_value=0.0)(char_input)
    # # Reshape word_lstm to match the last dimension of char_input_masked
    # word_lstm_reshaped = Reshape((max_sentence_length, lstm_units * 2, 1))(word_lstm)
    # Extract the last time step of the output of the word_lstm
    # Extract the last time step of the output of the word_lstm

    # # Reshape char_input to match the last dimension of last_word_lstm_output
    # char_input_reshaped = Reshape((max_sentence_length, max_word_length * char_embedding_dim))(char_input_masked)
    word_lstm_expanded = tf.tile(tf.expand_dims(word_lstm, axis=-1), multiples=[1, 1, 1, 37])
    # # Concatenate the last_word_lstm_output with char_input_reshaped
    merged_input = concatenate([word_lstm_expanded, char_input], axis=2)

    # tensor_shape = (64, 64)
    reshaped_input = tf.reshape(merged_input, (num_sentences, 380, -1))
    # # Create a Keras tensor of zeros
    # zeros_tensor = Input(shape=tensor_shape, dtype=tf.float32,batch_size=0)

    # Bidirectional LSTM for character input without specifying initial_state
    char_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(reshaped_input)
    # # Concatenate along a new axis
    # merged_input = concatenate([word_lstm_reshaped, char_input_masked], axis=-1)


    # Linear classifiers for each diacritic
    num_diacritics = 15
    linear_classifiers = [Dense(1, activation='linear') for _ in range(num_diacritics)]
    linear_outputs = [classifier(char_lstm) for classifier in linear_classifiers]

    # Create the model
    combined_model = Model(inputs=[word_input, char_input], outputs=linear_outputs)

    # Compile the model
    combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Display the model summary
    combined_model.summary()
    globals.our_model = combined_model
    utils.save_model('initial_model',combined_model)


def training_model():

    # Define your padding vector
    label_padding = -1


    num_sentences, max_sentence_length, word_embedding_dim = globals.word_embeddings_numpy.shape
    max_word_length, char_embedding_dim = globals.char_embeddings_numpy.shape[2:4]

    padded_labels = []
    for sentence in globals.model_labels:
        new_sentence = []

        for word in sentence:
            # Pad the word to the max_word_length
            word = word + [label_padding] * (max_word_length - len(word))
            new_sentence.append(word)
        
        empty_word = [label_padding] * max_word_length
        new_sentence = new_sentence + [empty_word] * (max_sentence_length - len(sentence))
        padded_labels.append(new_sentence)

    labels_numpy = np.array(padded_labels)

   
    epochs = 10
    # batch_size = 32

    for epoch_index in range(epochs):
        for i in range(num_sentences):
            current_word_data = globals.word_embeddings_numpy[i]
            current_char_data = globals.char_embeddings_numpy[i]
            labels = labels_numpy[i]

            # Forward pass for the entire sentence
            globals.our_model.train_on_batch([current_word_data, current_char_data], labels)
            print("TRAINING - Finished sentence number ", i)
        folder_name='models/epoch_'+str(epoch_index)

        print("TRAINING - Finished epoch number ", epoch_index)

        # After the training loop, you can save or evaluate your models as needed
        utils.save_model(folder_name,globals.our_model)
    




    ########################## Evaluate the model ##############################
    # Load the model
    # combined_model = keras.models.load_model('combined_model')
    #     # Assuming you have a separate test dataset and labels
    # test_word_embeddings_list = [...]  # List of precomputed word embeddings for each test sentence
    # test_char_embeddings_list = [...]  # List of precomputed character embeddings for each test word
    # test_labels_list = [...]  # List of lists of labels for each diacritic in the test data

    # # Convert test data to NumPy arrays
    # test_word_embeddings = np.array(test_word_embeddings_list)
    # test_char_embeddings = np.array(test_char_embeddings_list)
    # test_labels_array = np.array(test_labels_list)

    # # Evaluate the model on the test data
    # evaluation_result = combined_model.evaluate([test_word_embeddings, test_char_embeddings], test_labels_array)

    # # Display evaluation results
    # print("Loss:", evaluation_result[0])
    # print("Accuracy:", evaluation_result[1])
