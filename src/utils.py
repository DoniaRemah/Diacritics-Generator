
import pickle

def saveToTextFile(path, data):
    with open(path, 'w',encoding='utf-8') as file:
        data.write(file)


def FromTextFileToPickle(textPath, picklePath):
    with open(textPath, 'r',encoding='utf-8') as textFile:
        with open(picklePath, 'wb') as pickleFile:
            pickle.dump(textFile.read(), pickleFile)

def FromPickleToData(picklePath):
    with open(picklePath, 'rb') as pickleFile:
        return pickle.load(pickleFile)