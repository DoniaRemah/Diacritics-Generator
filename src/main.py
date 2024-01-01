import globals 
import pre_processing as pre
import utils

def read_data():
    ##################################################TRAINING DATA###########################################################
    ################################################## Reading Unclean Data ##################################################
    # with open('dataset/train.txt', 'r',encoding='utf-8') as train_file:
    #     globals.unclean_sentences = train_file.readlines()

    
    ################################################## Reading Cleaned Data ##################################################
    # with open('output/cleaned_train.txt', 'r',encoding='utf-8') as cleaned_train_file:
    #     globals.clean_sentences = cleaned_train_file.readlines()

    # globals.clean_sentences = [sentence.strip("\n") for sentence in globals.clean_sentences]


    # with open('output/cleaned_val.txt', 'r',encoding='utf-8') as cleaned_train_file:
    #     globals.clean_sentences = cleaned_train_file.readlines()

    # globals.clean_sentences = [sentence.strip("\n") for sentence in globals.clean_sentences]
    ################################################## Reading DEV DATA ##################################################
    # with open('dataset/val.txt', 'r',encoding='utf-8') as test_file:
    #     globals.test_sentences = test_file.readlines()

    ################################################## Reading TEST DATA ##################################################
    with open('output/competition/cleaned_test.txt', 'r',encoding='utf-8') as cleaned_test_file:
        globals.clean_sentences = cleaned_test_file.readlines()

    globals.clean_sentences = [sentence.strip("\n") for sentence in globals.clean_sentences]


    globals.letters={'#','س', 'ك', 'ن', 'ص', 'ة', 'ء', 'ؤ', 'ا', 'ع', 'ذ', 'ظ', 'ز', 'د', 'خ', 'ف', 'ش', 'ي', 'ض', 'ه', 'ر', 'ئ', 'إ', 'أ', 'غ', 'و', 'ب', 'ق', 'ى', 'ح', 'آ', 'ت', 'ج', 'م', 'ل', 'ط', 'ث'}
    

    # ################################### READING DIACRITIC IDS
    globals.diacritics_ids = utils.loadPickle('Letters and Diacritics/diacritic2id.pickle')



    ################################################## Reading TEST DATA ##################################################
    # with open('dataset/test_no_diacritics.txt', 'r',encoding='utf-8') as test_no_file:
    #     globals.unclean_sentences = test_no_file.readlines()

def main():
    read_data()
    # pre.clean_data()

    # globals.tokenized_sentences = utils.loadPickle('output/tokenized_sentences_0_2224.pickle')
    # # utils.saveToTextFile('tokenized_sentences_0_2224.txt',globals.tokenized_sentences)

    pre.pre_processing()


    print("finished preprocessing")
    

if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function

