import utils
import globals
import models_prediction

# load the data 
globals.word_embeddings=utils.loadPickle('output/word_embeddings_45000_50000.pickle')
globals.char_embeddings=utils.loadPickle('output/char_embeddings_45000_50000.pickle')

models_prediction.extract_chars()
max_sentence_length = max(len(seq) for seq in globals.word_embeddings) -2

max_word_length = max(max(len(word) for word in sentence) for sentence in globals.model_char_embeddings)

print("max sentence length is: ", max_sentence_length)
print("max word length is: ", max_word_length)

# max sentence length is:  740
# max word length is:  10

# max sentence length is:  1708
# max word length is:  10

# max sentence length is:  363
# max word length is:  10

# max sentence length is:  314
# max word length is:  11

# max sentence length is:  323
# max word length is:  10


# rana
# max sentence length is:  437
# max word length is:  11

# max sentence length is:  436
# max word length is:  10

# max sentence length is:  560
# max word length is:  11

#max sentence length is:  930
# max word length is:  11

# max sentence length is:  561
# max word length is:  10
