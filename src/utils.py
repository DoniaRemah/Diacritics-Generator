
import pickle

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
    