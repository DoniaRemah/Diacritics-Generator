import utils
import globals

# load the data 
globals.word_embeddings=utils.loadPickle('output_from0to10000/word_embeddings_0_5000.pickle')
globals.char_embeddings=utils.loadPickle('output_from0to10000/char_embeddings_0_5000.pickle')

max_sentence_length = max(len(seq) for seq in globals.word_embeddings) -2

max_word_length = max(max(len(word) for word in sentence) for sentence in globals.char_embeddings)

print("max sentence length is: ", max_sentence_length)
print("max word length is: ", max_word_length)