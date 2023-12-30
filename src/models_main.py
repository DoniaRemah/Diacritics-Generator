import models

def create_model():
    models.load_data_for_extraction()
    print("MAIN - finished loading data for extraction")
    models.extract_char_embeddings_and_labels_align_labels()
    print("MAIN - finished extracting char embeddings and labels")
    models.load_data_for_model_creation()
    print("MAIN - finished loading data for model creation")
    models.create_model()
    print("MAIN - finished creating model")

def train_model(model_name):
    models.load_saved_model(model_name)
    print("MAIN - finished loading model")
    models.load_data_for_training()
    print("MAIN - finished loading data for training")
    models.training_model()
    print("MAIN - finished training model")


def main():
    create_model()
    print("MAIN - OVERRRRRR")




if __name__ == "__main__":  # If this file is run directly
    main()                  # Run the main() function