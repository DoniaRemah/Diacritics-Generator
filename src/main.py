import globals  # Import the globals.py file

def read_data():
    with open('dataset/train.txt', 'r',encoding='utf-8') as train_file:
        globals.training_sentences = train_file.readlines()
    with open('dataset/val.txt', 'r',encoding='utf-8') as test_file:
        globals.test_sentences = test_file.readlines()

def main():
    read_data()
    

if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function

