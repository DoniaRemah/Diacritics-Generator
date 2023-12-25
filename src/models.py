from keras.models import Model
from keras.layers import LSTM, Dense, Input, concatenate
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences

# # Assume you have precomputed word embeddings and character embeddings
# word_embeddings_list = [...]  # List of precomputed word embeddings for each sentence
# char_embeddings_list = [...]  # List of precomputed character embeddings for each word

# # Labels for each diacritic
# diacritic_labels_dict = {
#     "diacritic_1": 0,
#     "diacritic_2": 1,
#     # Add more diacritics as needed
# }

# # Invert the dictionary to map indices to diacritic labels
# index_to_diacritic = {v: k for k, v in diacritic_labels_dict.items()}

# # Assuming the embeddings are numpy arrays
# word_embeddings = np.array(word_embeddings_list)
# char_embeddings = np.array(char_embeddings_list)

# # Shape information based on the precomputed embeddings
# num_sentences, _, word_embedding_dim = word_embeddings.shape
# max_word_length, char_embedding_dim = char_embeddings.shape[1:]

# # Word LSTM model with return_sequences=True
# word_input = Input(shape=(None, word_embedding_dim))
# word_lstm = LSTM(lstm_units, return_sequences=True)(word_input)

# # Character LSTM model with input directly from word-level LSTM
# char_input = Input(shape=(None, char_embedding_dim))
# merged_input = concatenate([word_lstm, char_input])
# char_lstm = LSTM(lstm_units)(merged_input)

# # SVM classifier for each diacritic
# num_diacritics = len(diacritic_labels_dict)
# svm_classifiers = [SVC(kernel='linear') for _ in range(num_diacritics)]
# svm_outputs = [classifier(char_lstm) for classifier in svm_classifiers]

# # Create the model
# combined_model = Model(inputs=[word_input, char_input], outputs=svm_outputs)

# # Compile the model
# combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Display the model summary
# combined_model.summary()

# # Example training loop
# labels_per_sentence = [
#     [diacritic_labels_dict["label_for_char_{}_in_sentence_{}".format(j, i)] for j in range(len(char_embeddings_list[i]))]
#     for i in range(num_sentences)
# ]

# epochs = 10
# batch_size = 32

# for epoch in range(epochs):
#     for i in range(num_sentences):
#         current_word_data = word_embeddings[i]
#         current_char_data = char_embeddings[i]
#         labels = labels_per_sentence[i]

#         # Forward pass for each character in the word
#         for j in range(len(char_embeddings_list[i])):
#             char_input_data = current_char_data[j]
#             char_label = labels[j]

#             # Forward pass for the current character
#             combined_model.train_on_batch([current_word_data, char_input_data], [char_label])

# # After the training loop, you can save or evaluate your models as needed


def training_model():

    pass