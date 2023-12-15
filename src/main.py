import globals  # Import the globals.py file
import pre_processing as pre
import re
######TODO : uncomment the following lines
def read_data():
    # with open('dataset/train.txt', 'r',encoding='utf-8') as train_file:
    #     globals.training_sentences = train_file.readlines()
    # with open('dataset/val.txt', 'r',encoding='utf-8') as test_file:
    #     globals.test_sentences = test_file.readlines()
    #TODO: read the cleaned data from the general file
    with open('dataset/cleaned_train.txt', 'r',encoding='utf-8') as cleaned_train_file:
        globals.clean_sentences = cleaned_train_file.readlines()


def dia_trial():
    arabic_sentence_with_diacritics = "اللُّغَةُ الْعَرَبِيَّةُ جَمِيْلَةٌ وَغَنِيَّةٌ بِالتَّنَوُّعِ."

    # Extract diacritics using a regular expression
    matches = re.finditer(r'([\u0621-\u064a])([\u064b-\u0652]*)', arabic_sentence_with_diacritics)
    # Initialize a list to store the result
    result_list = []

    # Iterate over matches
    for match in matches:
        char = match.group(1)
        diacritics = match.group(2)
        print(char,diacritics)
        # write to a file
        with open('dataset/test_diacritics.txt', 'a',encoding='utf-8') as diacritics_file:
            diacritics_file.write(char + '\t' + diacritics + '\n')
        # Create a list of lists with each character and its associated diacritic
        # word_list = [[char, diacritic] for char, diacritic in zip(word, diacritics)]
        
        # result_list.append(word_list)
    
    # print(result_list)
def main():
    read_data()
    # pre.clean_data()
    pre.pre_processing()
    print("finished")
    # dia_trial()

    

if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function

