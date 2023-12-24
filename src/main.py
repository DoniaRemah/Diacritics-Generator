import globals  # Import the globals.py file
import pre_processing as pre
import re
######TODO : uncomment the following lines
def read_data():
    ################################################## Reading Unclean Data ##################################################
    # with open('dataset/train.txt', 'r',encoding='utf-8') as train_file:
    #     globals.training_sentences = train_file.readlines()
    
    ################################################## Reading Dev Data ##################################################
    # with open('dataset/val.txt', 'r',encoding='utf-8') as test_file:
    #     globals.test_sentences = test_file.readlines()
    
    ################################################## Reading Cleaned Data ##################################################
    with open('dataset/cleaned_train.txt', 'r',encoding='utf-8') as cleaned_train_file:
        globals.clean_sentences = cleaned_train_file.readlines()
    

def main():
    read_data()
    # pre.clean_data()
    pre.pre_processing()
    print("finished")
    

if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function

