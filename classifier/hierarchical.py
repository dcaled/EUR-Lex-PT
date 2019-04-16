import io,json
from collections import defaultdict
import scipy.sparse

import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import keras
from keras import regularizers
from keras.layers import Input, Activation,Embedding, LSTM, Dense,Bidirectional,Dropout,TimeDistributed,GRU
from keras.models import Model
from keras import layers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from attention import AttentionWithContext

from sklearn.preprocessing import MultiLabelBinarizer


from keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf

from keras.engine.topology import Layer
import keras.backend as K
class Pentanh(Layer):

    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'pentanh'

    def call(self, inputs): return K.switch(K.greater(inputs,0), K.tanh(inputs), 0.25 * K.tanh(inputs))

    def get_config(self): return super(Pentanh, self).get_config()

    def compute_output_shape(self, input_shape): return input_shape

keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})

import string
pontuacao = string.punctuation+'\u2026'+'\xbb'+'\xab'+'\xba'+'“'+'’'+'”'+'‘'+'–'
phorbidden = list(pontuacao.replace('-','').replace('_',''))
phorbidden += list('0123456789')
def asphorbidden(token):
    return any(x in token for x in phorbidden)

def cleantext(txt):
    words = txt.split()
    new_text = ' '.join([w for w in words if not asphorbidden(w)]).lower()
    return new_text


def precisionatk(y_true,y_pred,k):
    precision_average = []
    idx =  (-y_pred).argsort(axis=-1)[:,:k]
    for i in range(idx.shape[0]):
        precision_sample = 0
        for j in idx[i,:]:
            if y_true[i,j] == 1:
                precision_sample += 1
        precision_sample = precision_sample / k
        precision_average.append(precision_sample)
    return np.mean(precision_average)


# import Levenshtein
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index)+1   # Adding again 1 because of reserved 0 index
    embedding_matrix = np.random.random((vocab_size, EMBEDDING_DIM))
    with io.open(filepath,encoding = 'utf8') as f:
        for i,line in enumerate(f):
            if i == 0:
                continue
            if '00\u2009% 0.048951 -0.002307 0.021459 0.016691 -0.043448 -0.063223 -0.026633 0.037860 0.042934' in line:
                continue
            if '00\u2009% 0.003048 -0.021540 -0.010371 -0.037366 -0.004355 -0.055332 0.045417 -0.053561 0.038612' in line:
                continue
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
            # else:
                #find the two words most similar


    return embedding_matrix


BATCH_SIZE=40
N_EPOCHS= 15
#
MAX_WORDS = 30000
MAX_DOC_LENGTH = 500
EMBEDDING_DIM = 300
#
N_SAMPLE_train = 10000000000000
N_SAMPLE_test = 10000000000000



###############################################################
###############################################################
###############################################################

if True:

    # create main dict with texts and the respective celex (as key)
    celex2text = {}
    # with io.open('../data/clean/clean_txt.json',encoding = 'utf8') as f:
    with io.open('../data/clean_txt_withallwords.json',encoding = 'utf8') as f:
        for i,line in enumerate(f):
            data = json.loads(line)
            celex2text[data['celex']] = data['txt']
    print("Textos lidos")


    y_train = defaultdict(list)
    celex_train = []
    with io.open('../data/train_val.txt') as f:
        for i,line in enumerate(f):
            if i >= N_SAMPLE_train:
                break

            celex,doms,mts,terms = line.split('|')
            celex = celex.strip()
            doms = doms.strip().split(',')
            mts = mts.strip().split(',')
            terms = terms.strip().split(',')
            y_train['dom'].append(doms)
            y_train['mts'].append(mts)
            y_train['terms'].append(terms)

            celex_train.append(celex)

    y_test = defaultdict(list)
    celex_test = []
    with io.open('../data/test.txt') as f:
        for i,line in enumerate(f):
            if i >= N_SAMPLE_test:
                break

            celex,doms,mts,terms = line.split('|')
            celex = celex.strip()
            doms = doms.strip().split(',')
            mts = mts.strip().split(',')
            terms = terms.strip().split(',')

            y_test['dom'].append(doms)
            y_test['mts'].append(mts)
            y_test['terms'].append(terms)

            celex_test.append(celex)

    y_bin_train = {}
    y_bin_test = {}
    for lvl in ['dom','mts','terms']:
        y = y_train[lvl] + y_test[lvl]
        mlb = MultiLabelBinarizer(sparse_output = True)
        y_bin = mlb.fit_transform(y)
        with io.open('../data/codes/codes_{}.txt'.format(lvl),'w') as f_out:
            for cl in mlb.classes_:
                f_out.write(cl+'\n')
        y_bin_train[lvl] = mlb.transform(y_train[lvl])
        y_bin_test[lvl] = mlb.transform(y_test[lvl])

        scipy.sparse.save_npz('../temp/y_{}_bin_train.npz'.format(lvl),y_bin_train[lvl])
        scipy.sparse.save_npz('../temp/y_{}_bin_test.npz'.format(lvl),y_bin_test[lvl])

        print(y_bin_train[lvl].shape,y_bin_test[lvl].shape)


    print("y construidos e gravados")

    corpus_train = [cleantext(celex2text[celex].replace('__SENT__',' ')) for celex in celex_train]
    corpus_test = [cleantext(celex2text[celex].replace('__SENT__',' ')) for celex in celex_test]

    # vectorizes to replace word by indexes followed by padding
    vectorizer = CountVectorizer(max_features = MAX_WORDS,lowercase = False,token_pattern = r"\S\S+")
    vectorizer.fit_transform(corpus_train)
    word_index = vectorizer.vocabulary_
    word_index = {tk:(ind+1) for tk,ind in word_index.items()}
    vocab_size = len(word_index)+1
    with io.open('../temp/word_index.tsv','w',encoding = 'utf8') as f_out:
        for word,ind in word_index.items():
            f_out.write('\t'.join([word.replace('\t',' '),str(ind)]))
            f_out.write('\n')



    # pad for train
    seq = [[word_index[tk] for tk in doc.split() if tk in word_index] for doc in corpus_train]
    X_train = pad_sequences(seq,maxlen = MAX_DOC_LENGTH,padding='pre',truncating='post')
    X_train = csr_matrix(np.array(X_train))
    scipy.sparse.save_npz('../temp/X_train.npz',X_train)
    print("X_train salvo")

    #pad for test
    seq = [[word_index[tk] for tk in doc.split() if tk in word_index] for doc in corpus_test]
    X_test = pad_sequences(seq,maxlen = MAX_DOC_LENGTH,padding='pre',truncating="post")
    X_test = csr_matrix(np.array(X_test))
    scipy.sparse.save_npz('../temp/X_test.npz',X_test)
    print("X_test salvo")

if True:

    word_index = {}
    with io.open('../temp/word_index.tsv',encoding = 'utf8') as f:
        for line in f:
            line = line.split('\t')
            if len(line) == 2:
                word_index[line[0]] = int(line[1])
    vocab_size = len(word_index) + 1

    # N_SAMPLE_train = 1000
    # N_SAMPLE_test = 100

    X_train = scipy.sparse.load_npz('../temp/X_train.npz')
    y_bin_train = {}
    for lvl in ['dom','mts','terms']:
        y_bin_train[lvl] = scipy.sparse.load_npz('../temp/y_{}_bin_train.npz'.format(lvl))
        # y_bin_train[lvl] = y_bin_train[lvl][np.arange(N_SAMPLE_train),:]

    X_test = scipy.sparse.load_npz('../temp/X_test.npz')
    y_bin_test = {}
    for lvl in ['dom','mts','terms']:
        y_bin_test[lvl] = scipy.sparse.load_npz('../temp/y_{}_bin_test.npz'.format(lvl))
        # y_bin_test[lvl] = y_bin_test[lvl][np.arange(N_SAMPLE_test),:]

    # X_train = X_train[np.arange(N_SAMPLE_train),:]
    # X_test = X_test[np.arange(N_SAMPLE_test),:]


    embedding_matrix = create_embedding_matrix('../data/embeddings/cbow_s300.txt',word_index,EMBEDDING_DIM)


    embedding_layer = Embedding(
        output_dim=EMBEDDING_DIM,
        input_dim=vocab_size,
        input_length=MAX_DOC_LENGTH,
        weights=[embedding_matrix],
        trainable=True,
        mask_zero=True)


    doc_input = Input(shape=(MAX_DOC_LENGTH,), dtype='int32',name = 'main_input')
    embedded_sequences = embedding_layer(doc_input)
    l_lstm = Bidirectional(LSTM(256, activation = 'pentanh',return_sequences=True))(embedded_sequences)
    l_lstm = Dropout(0.3)(l_lstm)
    l_att = AttentionWithContext()(l_lstm)
    l_att = Dense(512)(l_att)
    # l_att = Dense(512)(l_att)


    # middle_dom = Dense(16,activation = 'relu')(l_att)
    out_dom = Dense(y_bin_train['dom'].shape[1], activation='sigmoid', name='dense_out_dom')(l_att)

    merged_text_dom = keras.layers.concatenate([l_att, out_dom], axis= -1) #text,dom
    # middle_mts = Dense(64,activation = 'relu')(merged_text_dom)
    out_mts = Dense(y_bin_train['mts'].shape[1], activation='sigmoid', name='dense_out_mts')(merged_text_dom)


    merged_text_dom_mts = keras.layers.concatenate([l_att, out_dom, out_mts], axis= -1) #text,dom,mts
    # middle_terms = Dense(512,activation = 'relu')(merged_text_dom_mts)
    out_terms = Dense(y_bin_train['terms'].shape[1], activation='sigmoid', name='dense_out_terms')(merged_text_dom_mts)

    model = Model(inputs=[doc_input], outputs=[out_dom, out_mts, out_terms])
    # adam = keras.optimizers.Adam(lr=0.01,decay = 0.0005)
    # opt = keras.optimizers.RMSprop(lr=0.05,decay =0.004, clipnorm=1.0)
    # opt = keras.optimizers.Adam(lr=0.05,decay =0.004, clipnorm=1.0)
    model.compile(optimizer='adam',
                      loss={'dense_out_dom': 'binary_crossentropy', 'dense_out_mts': 'binary_crossentropy', 'dense_out_terms': 'binary_crossentropy'},
                      loss_weights={'dense_out_dom': 0.001, 'dense_out_mts': 0.01, 'dense_out_terms': 1})
    model.summary()

    # es = EarlyStopping(monitor='val_dense_out_terms_loss', mode='min', min_delta=0, verbose = 1,patience = 20)
    mc = ModelCheckpoint('models/hierarchical_v6.{epoch:02d}.h5', monitor='val_dense_out_terms_loss', mode='min', verbose=1, save_best_only=False)

    history = model.fit({'main_input': X_train},
                            {'dense_out_dom': y_bin_train['dom'],
                            'dense_out_mts': y_bin_train['mts'],
                            'dense_out_terms': y_bin_train['terms']},
                            validation_data = ({'main_input': X_test},
                                {'dense_out_dom': y_bin_test['dom'],
                                'dense_out_mts': y_bin_test['mts'],
                                'dense_out_terms': y_bin_test['terms']}
                                ),
                                epochs=N_EPOCHS,
                                verbose = 2,
                                batch_size=BATCH_SIZE,
                                callbacks = [mc]
                                )

    y_pred = model.predict(X_test)

    for i,lvl in enumerate(['dom','mts','terms']):
        for k in [1,3,5]:
            print(lvl,"P@k={}".format(k),precisionatk(y_bin_test[lvl],y_pred[i],k))



    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'], loc='upper right')
    plt.title("main loss")
    plt.savefig('plots/main_loss.png')

    plt.clf()
    plt.plot(history.history['dense_out_dom_loss'])
    plt.plot(history.history['val_dense_out_dom_loss'])
    plt.legend(['train', 'test'], loc='upper right')
    plt.title("dom loss")
    plt.savefig('plots/dom_loss.png')

    plt.clf()
    plt.plot(history.history['dense_out_mts_loss'])
    plt.plot(history.history['val_dense_out_mts_loss'])
    plt.legend(['train', 'test'], loc='upper right')
    plt.title("mts loss")
    plt.savefig('plots/mts_loss.png')

    plt.clf()
    plt.plot(history.history['dense_out_terms_loss'])
    plt.plot(history.history['val_dense_out_terms_loss'])
    plt.legend(['train', 'test'], loc='upper right')
    plt.title("terms loss")
    plt.savefig('plots/terms_loss.png')
