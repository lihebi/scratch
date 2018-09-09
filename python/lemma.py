#!/usr/bin/env python3

# https://stackoverflow.com/questions/33064426/simple-stemming-and-lemmatization-in-python

from stanfordcorenlp import StanfordCoreNLP
import nltk
import json

def lemma_corenlp(sentence):
    stanfordcorenlp_jar = "/home/hebi/tmp/stanford-corenlp-full-2018-02-27/"
    nlp = StanfordCoreNLP(stanfordcorenlp_jar)
    props={
        # 'annotators': 'tokenize,ssplit,pos,lemma',
        'annotators': 'lemma',
        # 'pipelineLanguage':'en',
        # 'outputFormat':'xml'
        # 'outputFormat': 'json'
    }
    # nlp.parse(sentence)
    result_str = nlp.annotate(sentence, props)
    result_json = json.loads(result_str)
    # FIXME In case of many sentences, this should change
    # tokens = result_json['sentences'][0]['tokens']
    # lemmas = [token['lemma'] for token in tokens]
    # To this:
    lemmas = []
    for sent in result_json['sentences']:
        tmp = [token['lemma'] for token in sent['tokens']]
        lemmas.extend(tmp)
    nlp.close()
    return lemmas

def lemma_nltk(sentence):
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in nltk.word_tokenize(sentence)]

def lemma_nltk_stem(sentence):
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    return [stemmer.stem(t) for t in nltk.word_tokenize(sentence)]

if __name__ == '__main__':
    sentence = "Several women told me I have lying eyes."
    sentence = "Several women told me I have lying eyes. I thought it was true."
    # this is very precise, but quite slow
    lemma_corenlp(sentence)
    # Need nltk.download('punkt') and nltk.download('wordnet'). nltk
    # based ones are very fast, though it has pretty bad precision
    lemma_nltk(sentence)
    lemma_nltk_stem(sentence)
