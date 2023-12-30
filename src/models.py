from keras.models import Model
from keras.layers import LSTM, Dense, Input, concatenate
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import globals

def extract_char_embeddings_and_labels():
    for sentece_index, sentence in enumerate(globals.word_embeddings):
        chars_per_sentence_list = []
        chars_index_per_sentence_list = []
        labels_per_sentence_list = []
        globals.model_word_embeddings.append(sentence)

        for word_index, word in enumerate(sentence):
            chars_per_word_list = []
            chars_index_per_word_list = []
            labels_per_word_list = []
            word,chars = globals.char_embeddings[sentece_index][word_index]
            _,char_labels = globals.golden_outputs_list[sentece_index][word_index]

            for char_index, char in enumerate(chars):
                char_num,char_vector = char
                the_char,label = char_labels[char_index]
                chars_per_word_list.append(char_vector)
                chars_index_per_word_list.append(char_num)
                labels_per_word_list.append(label)

            chars_per_sentence_list.append(chars_per_word_list)
            chars_index_per_sentence_list.append(chars_index_per_word_list)
            labels_per_sentence_list.append(labels_per_word_list)

        globals.model_char_embeddings.append(chars_per_sentence_list)
        globals.model_chars_index_per_corpus.append(chars_index_per_sentence_list)
        globals.model_labels.append(labels_per_sentence_list)



def training_model():
    word_embeddings_numpy = np.array(globals.model_word_embeddings)
    char_embeddings_numpy = np.array(globals.model_char_embeddings)
    labels_numpy = np.array(globals.model_labels)

    # Shape information based on the precomputed embeddings
    num_sentences, max_word_length, word_embedding_dim = word_embeddings_numpy.shape
    max_word_length, char_embedding_dim = char_embeddings_numpy.shape[1:]

    # LSTM units
    lstm_units = 64

    # Word LSTM model with return_sequences=True
    word_input = Input(shape=(None, word_embedding_dim))
    word_lstm = LSTM(lstm_units, return_sequences=True)(word_input)

    # Character LSTM model with input directly from word-level LSTM
    char_input = Input(shape=(None, char_embedding_dim))
    merged_input = concatenate([word_lstm, char_input])
    char_lstm = LSTM(lstm_units, return_sequences=True)(merged_input)

    # SVM classifier for each diacritic
    num_diacritics = 15
    svm_classifiers = [SVC(kernel='linear') for _ in range(num_diacritics)]
    svm_outputs = [classifier(char_lstm) for classifier in svm_classifiers]

    # Create the model
    combined_model = Model(inputs=[word_input, char_input], outputs=svm_outputs)

    # Compile the model
    combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Display the model summary
    combined_model.summary()



    epochs = 10
    # batch_size = 32

    for epoch in range(epochs):
        for i in range(num_sentences):
            current_word_data = word_embeddings_numpy[i]
            current_char_data = char_embeddings_numpy[i]
            labels = labels_numpy[i]

            # Forward pass for the entire sentence
            combined_model.train_on_batch([current_word_data, current_char_data], labels)

    # After the training loop, you can save or evaluate your models as needed
    combined_model.save('combined_model')

    # Evaluate the model
    


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
