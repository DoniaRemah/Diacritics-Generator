import models
import tensorflow as tf
import os 


os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU

def extract():
    models.load_data_for_extraction()
    print("MAIN - finished loading data for extraction")
    models.extract_char_embeddings_and_labels_align_labels()
    print("MAIN - finished extracting char embeddings and labels")
    
# def extract_and_create_model():
#     models.load_data_for_extraction()
#     print("MAIN - finished loading data for extraction")
#     models.extract_char_embeddings_and_labels_align_labels()
#     print("MAIN - finished extracting char embeddings and labels")
#     models.load_data_for_model_creation()
#     print("MAIN - finished loading data for model creation")
#     models.create_model()
#     print("MAIN - finished creating model")

# def create_model():
#     models.load_data_for_model_creation()
#     print("MAIN - finished loading data for model creation")
#     models.create_model()
#     print("MAIN - finished creating model")

def train_model(model_name,weights_name=""):
    # 0_2224/weights_epoch_09.h5 -> pass this
    models.load_saved_model(model_name)
    print("MAIN - finished loading model")
    models.load_data_for_training()
    print("MAIN - finished loading data for training")
    models.training_model()
    print("MAIN - finished training model")

def test_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Check if GPUs are available
    if gpus:
        try:
            # Set memory growth for the first GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

def test():
    models.load_data_for_model_creation()
    models.chunk_data()

def main():

    extract()
    models.prepare_data()
    models.create_model()
    # test()
    # extract_and_create_model()
    
    # train_model("initial_model")
    print("MAIN - OVERRRRRR")




if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function