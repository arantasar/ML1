import random, nltk

nltk.download('names')

male_names = nltk.corpus.names.words('male.txt')
female_names = nltk.corpus.names.words('female.txt')