import globals  # Import the globals.py file
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re


def extract_golden_output():
    pass

def tokenize():
    pass

def wordTokenize():
    pass

def charTokenize():
    pass

def clean_data():
    text = globals.training_sentences
    clean_train_sentences =[]
    for sentence in text:
        
        cleaned_sentence = re.sub(r'[^\u0600-\u06FF\u064B-\u065F\u0670\s]', '', sentence)
        clean_train_sentences.append(re.sub(r'[،؛]', '', cleaned_sentence))
        # \u0600-\u06FF: Arabic characters range.
        # \u064B-\u065F: Arabic diacritic marks range.
        # \u0670: Arabic vowel length mark.
        # \s: Whitespace characters.
        # 0-9: Numbers.
        # ،؛: Arabic symbols for comma (،) and semicolon (؛).

    globals.cleaned_train_sentences = clean_train_sentences
        
    with open('dataset/cleaned_train.txt', 'w',encoding='utf-8') as clean_train_file:
        for sentence in clean_train_sentences:
            clean_train_file.write(sentence)
    

