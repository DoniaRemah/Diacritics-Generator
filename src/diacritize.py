def add_diacritics(diacritic_dict, sentence, keys):
    diacritized_sentence = ""
    for char, key in zip(sentence, keys):
        diacritic = diacritic_dict.get(key, "")
        diacritized_sentence += char + diacritic
    return diacritized_sentence
