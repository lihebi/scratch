import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import random
import json

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras import layers

import os, sys

sys.path.append('/home/hebi/github/reading/bert')
sys.path.append('/home/hebi/github/scratch/deeplearning/anti-rouge')

import tokenization
from vocab import Vocab, PAD_TOKEN


# vocab_file = '/home/hebi/Downloads/uncased_L-12_H-768_A-12/vocab.txt'
vocab_file = '/home/hebi/github/reading/cnn-dailymail/finished_files/vocab'

cnndm_dir = '/home/hebi/github/reading/cnn-dailymail/'
cnn_dir = os.path.join(cnndm_dir, 'data/cnn/stroies')
dm_dir = os.path.join(cnndm_dir, 'data/dailymail/stories')
cnn_tokenized_dir = os.path.join(cnndm_dir, 'cnn_stories_tokenized')
dm_tokenized_dir = os.path.join(cnndm_dir, 'dm_stories_tokenized')

def tokenize_cnndm():
    # using bert tokenizer
    tokenizer = FullTokenizer(vocab_file=vocab_file)
    tokens = tokenizer.tokenize('hello world! This is awesome.')
    for f in os.listdir(cnn_dir):
        file = os.path.join(cnn_dir, f)
        tokenized_dir = os.path.join(cnn_dir, )
    tokenizer.tokenize()

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def get_art_abs(story_file):
    lines = read_text_file(story_file)
    lines = [line.lower() for line in lines]
    # lines = [fix_missing_period(line) for line in lines]
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    article = ' '.join(article_lines)
    abstract = ' '.join(highlights)
    return article, abstract

def delete_words(summary, ratio):
    words = summary.split(' ')
    length = len(words)
    indices = set(random.sample(range(length),
                                int((1 - ratio) * length)))
    return ' '.join([words[i] for i in range(length)
                     if i not in indices])


def add_words(summary, ratio, vocab):
    words = summary.split(' ')
    length = len(words)
    indices = set([random.randint(0, length)
                   for _ in range(int((1 - ratio) * length))])
    res = []
    for i in range(length):
        if i in indices:
            res.append(vocab.random_word())
        res.append(words[i])
    return ' '.join(res)


def mutate_summary(summary, vocab):
    """I need to generate random mutation to the summary. Save it to a
    file so that I use the same generated data. For each summary, I
    generate several data:
        
    1. generate 10 random float numbers [0,1] as ratios
    2. for each ratio, do:
    2.1 deletion: select ratio percent of words to remove
    2.2 addition: add ratio percent of new words (from vocab.txt) to
    random places

    Issues:
    
    - should I add better, regularized noise, e.g. gaussian noise? How
      to do that?
    - should I check if the sentence is really modified?
    - should we use the text from original article?
    - should we treat sentences? should we maintain the sentence
      separator period?

    """
    ratios = [random.random() for _ in range(10)]
    res = []
    # add the original summary
    res.append([summary, 1.0, 'orig'])
    # the format: ((summary, score, mutation_method))
    for r in ratios:
        s = delete_words(summary, r)
        res.append((s, r, 'del'))
        s = add_words(summary, r, vocab)
        res.append((s, r, 'add'))
    return res

def preprocess_data():
    """
    1. load stories
    2. tokenize
    3. separate article and summary
    4. chunk and save

    This runs pretty slow
    """
    print('Doing nothing.')
    return 0
    vocab = Vocab(vocab_file, 200000)

    # 92,579 stories
    stories = os.listdir(cnn_tokenized_dir)
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    if not os.path.exists(hebi_dir):
        os.makedirs(hebi_dir)
    # hebi/xxxxxx/article.txt
    # hebi/xxxxxx/summary.json
    ct = 0
    for s in stories:
        ct += 1
        # if ct > 10:
        #     return
        # print('--', ct)
        if ct % 100 == 0:
            print ('--', ct*100)
        f = os.path.join(cnn_tokenized_dir, s)
        article, summary = get_art_abs(f)
        pairs = mutate_summary(summary, vocab)
        # write down to file
        d = os.path.join(hebi_dir, s)
        if not os.path.exists(d):
            os.makedirs(d)
        article_f = os.path.join(d, 'article.txt')
        summary_f = os.path.join(d, 'summary.json')
        with open(article_f, 'w') as fout:
            fout.write(article)
        with open(summary_f, 'w') as fout:
            json.dump(pairs, fout, indent=4)

def encode(sentence, vocab):
    """Using vocab encoding
    """
    words = sentence.split()
    return [vocab.word2id(w) for w in words]

def encode(sentence, vocab):
    """Using sentence encoding
    """
    words = sentence.split()
    return [vocab.word2id(w) for w in words]

def encode(sentence, vocab):
    """Using word embedding
    """
    words = sentence.split()
    return [vocab.word2id(w) for w in words]

def decode(ids, vocab):
    return ' '.join([vocab.id2word(i) for i in ids])

def prepare_data():
    """
    1. define a batch
    2. load a batch
    3. return as (article, summary) pairs?

    1. load all stories and summaries
    2. convert stories and summaries into vectors, according to vocab
    3. trunk or pad with MAX_LEN
    3. return (article, summary, score)
    """
    article_data = []
    summary_data = []
    score_data = []
    # 90,000 > 2,000
    vocab = Vocab(vocab_file, 200000)
    hebi_dir = os.path.join(cnndm_dir, 'hebi')
    hebi_sample_dir = os.path.join(cnndm_dir, 'hebi-sample')
    data_dir = hebi_sample_dir
    # data_dir = hebi_dir
    ct = 0
    for s in os.listdir(data_dir):
        ct+=1
        if ct % 100 == 0:
            print ('--', ct)
        article_f = os.path.join(data_dir, s, 'article.txt')
        summary_f = os.path.join(data_dir, s, 'summary.json')
        article_content = ' '.join(read_text_file(article_f))
        article_encoding = encode(article_content, vocab)
        with open(summary_f, 'r') as f:
            summaries = json.load(f)
            for summary,score,_ in summaries:
                article_data.append(article_encoding)
                summary_data.append(encode(summary, vocab))
                score_data.append(score)
    print('converting to numpy array ..')
    score_data = np.array(score_data)
    summary_data = np.array(summary_data)
    article_data = np.array(article_data)

    # plt.plot([len(v) for v in summary_data])
    # plt.hist([len(v) for v in article_data])

    print('padding ..')
    article_data_padded = pad_sequences(article_data,
                                        value=vocab.word2id(PAD_TOKEN),
                                        padding='post',
                                        maxlen=512)
    summary_data_padded = pad_sequences(summary_data,
                                        value=vocab.word2id(PAD_TOKEN),
                                        padding='post',
                                        maxlen=100)
    # x = np.array([np.concatenate((a,s)) for a,s in zip(article_data_padded, summary_data_padded)])
    print('concatenating ..')
    x = np.concatenate((article_data_padded, summary_data_padded),
                       axis=1)
    # np.concatenate((([1,2]), np.array([3])))
    # np.concatenate(([1,2], [3], [4,5,6]))
    y = score_data
    x.shape                     # (21000,768)
    y.shape                     # (21000,)
    # split x and y in training and testing
    split_at = len(x) // 10
    x_train = x[split_at:]
    x_val = x[:split_at]
    y_train = y[split_at:]
    y_val = y[:split_at]
    return (x_train, y_train), (x_val, y_val)

def build_model():
    """The model contains:
    """
    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64,
                           # input_shape=(768,),
                           activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    # model.add(layers.Dense(10, activation='softmax'))
    # output the score
    model.add(layers.Dense(1, activation='sigmoid'))

    # x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = GlobalMaxPooling1D()(x)
    # x = Dense(128, activation='relu')(x)
    
    return model

def main():
    """Steps:
    1. preprocess data:
      - tokenization (sentence tokenizer)
      - separate article and reference summary
      - chunk into train and test

    2. data generation: for each reference summary, do the following
    mutation operations: deletion, insertion, mutation. According to
    how much are changed, assign a score.
    
    3. sentence embedding: embed article and summary into sentence
    vectors. This is the first layer, the embedding layer. Then, do a
    padding to get the vector to the same and fixed dimension
    (e.g. summary 20, article 100). FIXME what to do for very long
    article? Then, fully connected layer directly to the final result.

    """

    (x_train, y_train), (x_val, y_val) = prepare_data()
    x_train.shape
    y_train.shape
    x_val.shape
    y_val.shape
    model = build_model()
    optimizer = tf.train.RMSPropOptimizer(0.001)
    # optimizer=tf.train.AdamOptimizer(0.01)
    model.compile(optimizer=optimizer,
                  # loss='binary_crossentropy',
                  loss='mse',
                  # metrics=['accuracy']
                  metrics=['mae'])
    model.fit(x_train, y_train,
              epochs=40, batch_size=32,
              validation_data=(x_val, y_val), verbose=1)
    model.summary()
    # results = model.evaluate(x_test, y_test)

def sentence_embedding():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    # Compute a representation for each message, showing various lengths supported.
    word = "Elephant"
    sentence = "I am a sentence for which I would like to get its embedding."
    paragraph = (
        "Universal Sentence Encoder embeddings also support short paragraphs. "
        "There is no hard limit on how long the paragraph is. Roughly, the longer "
        "the more 'diluted' the embedding will be.")
    messages = [word, sentence, paragraph]

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(messages))

        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            print("Message: {}".format(messages[i]))
            print("Embedding size: {}".format(len(message_embedding)))
            message_embedding_snippet = ", ".join(
                (str(x) for x in message_embedding[:3]))
            print("Embedding: [{}, ...]\n".format(message_embedding_snippet))


class MyEmbedding():
    def __init__(self):
        self.module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.embed_session = tf.Session()
        self.embed_session.run(tf.global_variables_initializer())
        self.embed_session.run(tf.tables_initializer())
    def embed(self, sentence):
        with tf.device('/cpu:0'):
            embedded = self.module(sentence)
        res = self.embed_session.run(embedded)
        return res

def myembed(sentence):
    myembed = MyEmbedding()
    myembed.embed(sentence)
    """Embed a string into 512 dim vector
    """
    sentence = ["The quick brown fox jumps over the lazy dog."]
    sentence = ["The quick brown fox is a jumping dog."]
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    # embeddings = embed(["The quick brown fox jumps over the lazy dog."])
    embed_session = tf.Session()
    embed_session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    with tf.device('/cpu:0'):
        embedded = embed(sentence)
    res = embed_session.run(embedded)
    return res

def test():
    myembed(["The quick brown fox jumps over the lazy dog."])
    with tf.device('/cpu:0'):
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        embeddings = embed(["The quick brown fox jumps over the lazy dog."])
        session = tf.Session()
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedded = session.run(embeddings)
        print (embedded)
    pass

def main():
    with tf.device('/cpu:0'):
        sentence_embedding()
