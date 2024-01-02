import models
import tensorflow as tf
import os 
import models_prediction as mp
import diacritize as d

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU

def extract():
    mp.load_data_for_extraction()
    print("MAIN - finished loading data for extraction")
    mp.extract_chars()

def main():

    extract()
    mp.prepare_data()
    print("MAIN - finished preparing data")

    models.load_saved_model('initial_model','weights_epoch_01.h5')
    print("finished loading model and weights")

    mp.load_padded_data()
    mp.predict()
    # mp.save_output()

    # for one sentence
    prediction=mp.get_output()
    # read sentence from text file output/cleaned_sentence.txt
    with open('output/cleaned_sentence.txt', 'r',encoding='utf-8') as train_file:
        sentence = train_file.readline()

    diacritic_dict={0: 'َ', 1: 'ً', 2: 'ُ', 3: 'ٌ', 4: 'ِ', 5: 'ٍ', 6: 'ْ', 7: 'ّ', 8: 'َّ', 9: 'ًّ', 10: 'ُّ', 11: 'ٌّ', 12: 'ِّ', 13: 'ٍّ', 14: ''}
    output=d.add_diacritics(diacritic_dict,sentence,prediction)
    print(output)
    # save output to text 
    with open('output/output_sentence.txt', 'w',encoding='utf-8') as train_file:
        train_file.write(output)
    print("MAIN - OVERRRRRR")




if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function