import globals 
import pre_processing as pre
import utils

def read_data():


    ##################################################TRAINING DATA###########################################################
    ################################################# Reading Unclean Data ##################################################
    with open('dataset/sentence.txt', 'r',encoding='utf-8') as train_file:
        globals.unclean_sentences = train_file.readlines()

    globals.letters={'#','س', 'ك', 'ن', 'ص', 'ة', 'ء', 'ؤ', 'ا', 'ع', 'ذ', 'ظ', 'ز', 'د', 'خ', 'ف', 'ش', 'ي', 'ض', 'ه', 'ر', 'ئ', 'إ', 'أ', 'غ', 'و', 'ب', 'ق', 'ى', 'ح', 'آ', 'ت', 'ج', 'م', 'ل', 'ط', 'ث'}
    

    # ################################### READING DIACRITIC IDS
    globals.diacritics_ids = utils.loadPickle('Letters and Diacritics/diacritic2id.pickle')


def main():
    read_data()
    pre.clean_data()

    pre.pre_processing()


    print("finished preprocessing")
    

if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function

