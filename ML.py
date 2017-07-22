import random, nltk, datetime

nltk.download('names')

male_names = nltk.corpus.names.words('male.txt')
female_names = nltk.corpus.names.words('female.txt')

labeled_names = [(name, 'male') for name in male_names] # [f(item) for item in iterable], eg. [x+100 for x in [2, 3, 4]]
labeled_names += [(name, 'female') for name in female_names]

random.seed(datetime.datetime.now())
random.shuffle(labeled_names)

def get_features(name):
    return {
        'first_letter': name[0],
        'last_but_one_letter': name[-2],
        'last_letter': name[-1],
        'length': len(name)
    }

featuresets = [(get_features(name), gender) for (name, gender) in labeled_names]

bayes_classifier = nltk.NaiveBayesClassifier.train(featuresets)