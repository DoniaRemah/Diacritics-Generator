test_sentences = []  # Sentences provided by the TAs with diacritics uncleaned, untokenized, List of Sentences
# cleaned_train_sentences = []
tokenized_words = [] # List of Word Tokens
tokenized_chars = [] # List of Char Tokens
golden_outputs = [] 
word_vocabulary = set()
tokenized_sentences = [] # List of Sentences of List of Word Tokens
tokenized_sentence_chars = [] 
unclean_sentences = [] # Sentences provided by the TAs with diacritics uncleaned, untokenized, List of Sentences
clean_sentences=[]
# a list of tuples, each one contains a word and a list of tuples of char and diacritic
golden_outputs_list = []
letters = set()
letters_vector = dict()
words_without_diacritics = []
diacritics_ids = dict()


# men gowa l barra
# tuple of (char, vector) this is for the char
# tuple (word, list of tuples of (char, vector)) this is for the word
# list of tuples of (word, list of tuples of (char, vector)) this is for the sentence
# list of list of tuples of (word, list of tuples of (char, vector)) this is for the whole dataset 
char_embeddings = []
word_embeddings = []