from keras.models import Model
from keras.layers import LSTM, Input, concatenate,Bidirectional,Masking,Dense,Reshape,Lambda
from keras import backend as K
import numpy as np
from sklearn.svm import SVC
import globals
import utils
import tensorflow as tf
from keras.models import load_model
import re
from keras.callbacks import ModelCheckpoint
from keras import losses

def load_data_for_extraction():

    globals.char_embeddings = utils.loadPickle('output/char_embeddings_5000_10000.pickle')
    print("finished loading char embeddings")
    globals.golden_outputs_list = utils.loadPickle('output/golden_outputs.pickle')
    print("finished loading golden outputs")
    globals.tokenized_sentences = utils.loadPickle('output/tokenized_sentences_5000_10000_withHamza.pickle')
    print("finished loading tokenized sentences")



def load_data_for_model_creation():

    globals.word_embeddings = utils.loadPickle('output/word_embeddings_5000_10000.pickle')
    print("finished loading word embeddings")
    globals.model_char_embeddings = utils.loadPickle('output/model/model_char_embeddings_5000_10000.pickle')

def load_data_for_training():
    # globals.word_embeddings = utils.loadPickle('output/word_embeddings_0_2224.pickle')
    # print("finished loading word embeddings")
    # globals.model_char_embeddings = utils.loadPickle('output/model/model_char_embeddings.pickle')
    # print("finished loading model char embeddings")
    globals.word_embeddings_numpy = utils.loadPickle('output/model/word_embeddings_numpy_0_2224.pickle')
    globals.char_embeddings_numpy = utils.loadPickle('output/model/char_embeddings_numpy_0_2224.pickle')
    globals.model_labels = utils.loadPickle('output/model/model_labels.pickle')
    print("finished loading model labels")

def load_saved_model(model_name,weight_name=""):
    name = "models/"+model_name
    globals.our_model = load_model(name)
    name = "models/weights/folder_5000_10000/"+weight_name
    globals.our_model.load_weights(name)

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
    for sentece_index, sentence in enumerate(globals.golden_outputs_list[5000:10000]):
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

    utils.SaveToPickle('output/model/model_char_embeddings_5000_10000.pickle', globals.model_char_embeddings)
    utils.SaveToPickle('output/model/model_chars_index_per_corpus_5000_10000.pickle', globals.model_chars_index_per_corpus)
    utils.SaveToPickle('output/model/model_labels_5000_10000.pickle', globals.model_labels)


def chunk_data():
    # Chunk word_embeddings and model_labels and model_char_embeddings, where each chunk has max size 400

    # Chunk word_embeddings
    chunked_word_embeddings = []
    chunked_model_labels = []   
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
    for sentence in globals.model_labels:
        if len(sentence) > 400:
            for i in range(0, len(sentence), 400):
                chunk_end = min(i + 400, len(sentence))
                chunked_model_labels.append(sentence[i:chunk_end])
        else:
            chunked_model_labels.append(sentence)

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
    globals.chunked_labels = chunked_model_labels
    utils.SaveToPickle('output/model/chunked_labels_5000_10000.pickle',globals.chunked_labels)
    globals.chunked_char_embeddings = chunked_model_char_embeddings


def pad_word_embeddings():

    max_sentence_length=400
    # Pad words to the maximum length
    padded_word_embeddings = []
    for sentence in globals.chunked_word_embeddings:
        padded_sentence = np.pad(sentence, ((0, max_sentence_length - len(sentence)), (0, 0)), mode='constant')
        padded_word_embeddings.append(padded_sentence)

    globals.padded_word_embeddings = padded_word_embeddings
    utils.SaveToPickle('output/model/padded_word_embeddings_5000_10000.pickle',globals.padded_word_embeddings)

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
    utils.SaveToPickle('output/model/padded_char_embeddings_5000_10000.pickle',globals.padded_char_embeddings)

def prepare_data():
    load_data_for_model_creation()
    print("finished loading data for model creation")
    globals.model_labels=utils.loadPickle('output/model/model_labels_5000_10000.pickle')
    chunk_data()
    print("finished chunking data")
    pad_word_embeddings()
    print("finished padding word embeddings")
    pad_char_embeddings()
    print("finished padding char embeddings")

def load_padded_data():
    globals.padded_word_embeddings=utils.loadPickle('output/model/padded_word_embeddings_5000_10000.pickle')
    globals.padded_char_embeddings=utils.loadPickle('output/model/padded_char_embeddings_5000_10000.pickle')
    globals.model_labels=utils.loadPickle('output/model/chunked_labels_5000_10000.pickle')

    globals.word_embeddings_numpy = np.array(globals.padded_word_embeddings)
    globals.char_embeddings_numpy = np.array(globals.padded_char_embeddings)
    # utils.SaveToPickle('output/model/word_embeddings_numpy_15000_20000.pickle', globals.word_embeddings_numpy)
    # utils.SaveToPickle('output/model/char_embeddings_numpy_15000_20000.pickle', globals.char_embeddings_numpy)


def create_model():

    max_sentence_length=400
    word_embedding_dim=768
    max_word_length=15
    char_embedding_dim=37
    # LSTM units
    lstm_units = 64

    # Word LSTM model with return_sequences=True
    word_input = Input(shape=(max_sentence_length,word_embedding_dim))

    # Input shape: Batch size, timesteps, features
    word_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(word_input)

    # Character LSTM model with input directly from word-level LSTM
    char_input = Input(shape=(max_sentence_length,max_word_length ,char_embedding_dim))

    word_lstm_expanded = tf.tile(tf.expand_dims(word_lstm, axis=-1), multiples=[1, 1, 1, 37])

    # Concatenate the last_word_lstm_output with char_input_reshaped
    merged_input = concatenate([word_lstm_expanded, char_input], axis=2)


    inferred_dimension = tf.reduce_prod(tf.shape(merged_input)[2:])  # Calculate the size of the third dimension
    reshaped_input = tf.reshape(merged_input, (tf.shape(merged_input)[0], max_sentence_length, inferred_dimension))

    # Input shape:Batch size, timesteps, features
    # Bidirectional LSTM for character input without specifying initial_state
    char_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(reshaped_input)


    num_diacritics = 15
    classifier_output = Dense(num_diacritics * max_word_length)(char_lstm)
    classifier_output = Reshape((max_sentence_length, max_word_length, num_diacritics))(classifier_output)

    # # Add a Lambda layer to perform argmax along the last dimension
    # classifier_argmax = Lambda(lambda x: K.argmax(x, axis=-1))(classifier_output)

    # Create the model  
    combined_model = Model(inputs=[word_input, char_input], outputs=classifier_output)

    # Compile the model
    combined_model.compile(optimizer='adam', loss = losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # Display the model summary
    combined_model.summary()
    globals.our_model = combined_model
    utils.save_model('initial_model',combined_model)



def training_model():

    # Define your padding vector
    label_padding = -1

    max_sentence_length=400

    max_word_length=15

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

    print("finished padding labels")


    # Mask the padded labels
    # labels_mask = (labels_numpy != label_padding)
    labels_mask = tf.math.not_equal(labels_numpy, -1)
    labels_mask = tf.cast(labels_mask, 'int64')
    labels_mask=np.array(labels_mask)

    # Apply the mask to the labels
    labels_mask_after_mask = (labels_numpy != label_padding)
    labels_numpy = np.multiply(labels_numpy, labels_mask_after_mask)
    labels_numpy = tf.cast(labels_numpy, 'int64')
    
   # Define a callback to save the model weights after each epoch
    checkpoint_callback = ModelCheckpoint(
        filepath='models/weights/folder_5000_10000/weights_epoch_{epoch:02d}.h5',  # Specify the filename format
        save_weights_only=True,  # Save only the weights, not the entire model
        verbose=1  # Display progress
    )

    epochs = 1
    batch_size = 50

    for i in range(0, len(globals.word_embeddings_numpy), 2500):
        batch_end = min(i + 2500, len(globals.word_embeddings_numpy))
        globals.our_model.fit([globals.word_embeddings_numpy[i:batch_end],globals.char_embeddings_numpy[i:batch_end] ], labels_numpy[i:batch_end], epochs=epochs, verbose=1,callbacks=[checkpoint_callback],batch_size=batch_size,sample_weight=labels_mask[i:batch_end])
        print("TRAINING - Finished batch number ", i)
    

def evaluating_model():
    # Define your padding vector
    label_padding = -1

    max_sentence_length=400

    max_word_length=15

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

    print("finished padding labels")


    # Mask the padded labels
    labels_mask = (labels_numpy != label_padding)

    # Apply the mask to the labels
    labels_numpy = np.multiply(labels_numpy, labels_mask)
    labels_numpy = tf.cast(labels_numpy, 'int64')

   # Evaluate the model on the test data
    # evaluation_result =  globals.our_model.evaluate([globals.word_embeddings_numpy, globals.char_embeddings_numpy], labels_numpy)
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

    # Save the output in csv file
    with open('output/predicted_labels_val.csv', 'w', encoding='utf-8') as f:
        f.write('label\n')
        for index1,sentence in enumerate(globals.predicted_labels):
            for index2,word in enumerate(sentence):
                for index3,char in enumerate(word):
                    f.write(str(char) +'\n')


    # # Display evaluation results
    # print("Loss:", evaluation_result[0])
    # print("Accuracy:", evaluation_result[1])




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
