import models
import tensorflow as tf
import os 
import models_prediction as mp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU

def extract():
    mp.load_data_for_extraction()
    print("MAIN - finished loading data for extraction")
    mp.extract_chars()

def main():

    # extract()
    # mp.prepare_data()
    # print("MAIN - finished preparing data")

    models.load_saved_model('initial_model','weights_epoch_01.h5')
    print("finished loading model and weights")

    mp.load_padded_data()
    mp.predict()
    mp.save_output()

    print("MAIN - OVERRRRRR")




if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function