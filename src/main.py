import globals  # Import the globals.py file
import pre_processing as pre
import utils

def read_data():
    ################################################## Reading Unclean Data ##################################################
    # with open('dataset/train.txt', 'r',encoding='utf-8') as train_file:
    #     globals.unclean_sentences = train_file.readlines()
    
    ################################################## Reading Dev Data ##################################################
    # with open('dataset/val.txt', 'r',encoding='utf-8') as test_file:
    #     globals.test_sentences = test_file.readlines()


    # globals.letters_vector = utils.loadPickle('output/letters_vector.pickle')
    # globals.tokenized_sentences= utils.loadPickle('output/tokenized_sentences.pickle')
    # globals.letters = utils.loadPickle('Letters and Diacritics/arabic_letters.pickle')
    # globals.tokenized_sentence_chars= utils.loadPickle('output/tokenized_sentence_chars.pickle')
    
    # globals.char_vocabulary= utils.loadPickle('output/char_vocabulary.pickle')
    # utils.saveToTextFile("output/char_vocabulary.txt",globals.char_vocabulary)
    # globals.words_without_diacritics= utils.loadPickle("output/words_without_diacritics.pickle")
    ################################################## Reading Cleaned Data ##################################################
    # with open('output/cleaned_train.txt', 'r',encoding='utf-8') as cleaned_train_file:
    #     globals.clean_sentences = cleaned_train_file.readlines()

    # globals.clean_sentences = [sentence.strip("\n") for sentence in globals.clean_sentences]

    # utils.FromTextFileToPickle("output/char_vocabulary.txt","output/char_vocabulary.pickle")
    # globals.letters = eval(utils.loadPickle("output/char_vocabulary.pickle"))
    # ################################### READING DIACRITIC IDS
    # globals.diacritics_ids = utils.loadPickle('Letters and Diacritics/diacritic2id.pickle')

    # TODO: READ VOCAB  
    pass


def main():
    read_data()
    # pre.clean_data()
    pre.pre_processing()
    print("finished")
    

if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function

