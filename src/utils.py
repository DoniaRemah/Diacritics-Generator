
import pickle
import os

def saveToTextFile(path, data):
    with open(path, 'w',encoding='utf-8') as file:
        file.write(str(data))


def FromTextFileToPickle(textPath, picklePath):
    with open(textPath, 'r',encoding='utf-8') as textFile:
        with open(picklePath, 'wb') as pickleFile:
            pickle.dump(textFile.read(), pickleFile)

def SaveToPickle(picklePath, data):
    with open(picklePath, 'wb') as pickleFile:
        pickle.dump(data, pickleFile)

def loadPickle(picklePath):
    with open(picklePath, 'rb') as pickleFile:
        return pickle.load(pickleFile)
    


def save_model(model_filename,combined_model):
    
        # Specify the directory and filename
    save_directory = 'models'

    # Ensure that the directory exists, create it if not
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the model
    combined_model.save(os.path.join(save_directory, model_filename))
