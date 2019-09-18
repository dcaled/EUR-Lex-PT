#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io,sys,json
import string
from collections import defaultdict

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.engine.topology import Layer
from keras import layers
from keras.layers import Input,Embedding,LSTM,Dense,Bidirectional,Dropout
from keras.models import load_model,Model,Sequential
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

import matplotlib.pyplot as plt

from attention import AttentionWithContext

import tensorflow as tf


class Pentanh(Layer):

    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = "pentanh"

    def call(self, inputs): return K.switch(K.greater(inputs,0), K.tanh(inputs), 0.25 * K.tanh(inputs))

    def get_config(self): return super(Pentanh, self).get_config()

    def compute_output_shape(self, input_shape): return input_shape

keras.utils.generic_utils.get_custom_objects().update({"pentanh": Pentanh()})


pontuacao = string.punctuation+'\u2026'+'\xbb'+'\xab'+'\xba'+'“'+'’'+'”'+'‘'+'–'
phorbidden = list(pontuacao.replace('-','').replace('_',''))
phorbidden += list("0123456789")
def asphorbidden(token):
    return any(x in token for x in phorbidden)

def cleantext(txt):
    words = txt.split()
    new_text = " ".join([w for w in words if not asphorbidden(w)]).lower()
    return new_text


def save_results(y_true,y_pred,idx2code,filepath_results):
    """Save to disk the ten EuroVoc labels with the higher probability."""
    idx = (-y_pred).argsort(axis=-1)[:,:10]
    with io.open(filepath_results,"w") as f_out:
        for doc in idx:
            eurovoc_labels = []
            for lbl in doc:
                eurovoc_labels+=[idx2code[lbl]]

            f_out.write(" ".join(eurovoc_labels)+"\n")


def load_codes(filepath_codes):
    idx2code = dict()
    for lvl in ["dom","mts","terms"]:
        idx2code[lvl] = dict()
        with io.open("{}codes_{}.txt".format(filepath_codes,lvl),"r") as f:
            for cnt, line in enumerate(f):
                idx2code[lvl][cnt] = line.strip()
    return idx2code
           


def precisionatk(y_true,y_pred,k):
    precision_average = []
    idx = (-y_pred).argsort(axis=-1)[:,:k]
    print(idx)
    for i in range(idx.shape[0]):
        precision_sample = 0
        for j in idx[i,:]:
            if y_true[i,j] == 1:
                precision_sample += 1
        precision_sample = precision_sample / k
        precision_average.append(precision_sample)
    return np.mean(precision_average)


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index)+1   # Adding again 1 because of reserved 0 index
    embedding_matrix = np.random.random((vocab_size, embedding_dim))
    with io.open(filepath,encoding = "utf8") as f:
        for i,line in enumerate(f):
            if i == 0:
                continue
            if "00\u2009% 0.048951 -0.002307 0.021459 0.016691 -0.043448 -0.063223 -0.026633 0.037860 0.042934" in line:
                continue
            if "00\u2009% 0.003048 -0.021540 -0.010371 -0.037366 -0.004355 -0.055332 0.045417 -0.053561 0.038612" in line:
                continue
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
            # else:
                #find the two words most similar
    return embedding_matrix


def load_eurlex_pt(filepath_eurlex_pt):
    """Create main dict with texts and the respective celex (as key)."""
    celex2text = {}
    with io.open(filepath_eurlex_pt,encoding="utf8") as f:
        for i,line in enumerate(f):
            data = json.loads(line)
            celex2text[data["celex"]] = data["txt"]
    return celex2text


def load_split(filepath_split, N_SAMPLE):
    """Loads the celex keys and the labels for the 3 hierarchical levels."""
    y = defaultdict(list)
    celexs = []
    with io.open(filepath_split) as f:
        for i,line in enumerate(f):
            if i >= N_SAMPLE:
                break

            celex,doms,mts,terms = line.split("|")
            celex = celex.strip()
            doms = doms.strip().split(",")
            mts = mts.strip().split(",")
            terms = terms.strip().split(",")
            y["dom"].append(doms)
            y["mts"].append(mts)
            y["terms"].append(terms)

            celexs.append(celex)
    return celexs,y


def label_binarizer(y_train,y_test,filepath_codes):
    """Performs the binarization of the EuroVoc labels, creating a sequencial index. These indexes are stored in
    disk files. Then, new binaries labels are created and also persisted to the disk."""
    y_bin_train = {}
    y_bin_test = {}
    for lvl in ["dom","mts","terms"]:
        y = y_train[lvl] + y_test[lvl]
        mlb = MultiLabelBinarizer(sparse_output = True)
        y_bin = mlb.fit_transform(y)
        with io.open("{}codes_{}.txt".format(filepath_codes,lvl),"w") as f_out:
            for cl in mlb.classes_:
                f_out.write(cl+"\n")
        y_bin_train[lvl] = mlb.transform(y_train[lvl])
        y_bin_test[lvl] = mlb.transform(y_test[lvl])

        scipy.sparse.save_npz("../temp/y_{}_bin_train.npz".format(lvl),y_bin_train[lvl])
        scipy.sparse.save_npz("../temp/y_{}_bin_test.npz".format(lvl),y_bin_test[lvl])

        print(lvl, y_bin_train[lvl].shape,y_bin_test[lvl].shape)


def create_word_index(corpus_train,max_features):
    """ Creates word indexes for the words in the training corpus."""
    vectorizer = CountVectorizer(max_features=max_features,lowercase=False,token_pattern=r"\S\S+")
    vectorizer.fit_transform(corpus_train)
    word_index = vectorizer.vocabulary_
    word_index = {tk:(ind+1) for tk,ind in word_index.items()}
    vocab_size = len(word_index)+1
    with io.open("../temp/word_index.tsv","w",encoding = "utf8") as f_out:
        for word,ind in word_index.items():
            f_out.write("\t".join([word.replace("\t"," "),str(ind)]))
            f_out.write("\n")
    return word_index


def load_word_index():
    word_index = {}
    with io.open("../temp/word_index.tsv",encoding="utf8") as f:
        for line in f:
            line = line.split("\t")
            if len(line) == 2:
                word_index[line[0]] = int(line[1])
    return word_index


def create_X(corpus,word_index,maxlen,filename):
    """ Replaces word by indexes followed by padding. The corpus features are saved to the disk."""
    seq = [[word_index[tk] for tk in doc.split() if tk in word_index] for doc in corpus]
    X = pad_sequences(seq,maxlen=maxlen,padding="pre",truncating="post")
    X = csr_matrix(np.array(X))
    scipy.sparse.save_npz("../temp/{}.npz".format(filename),X)
    

def load_X_y(split, N_SAMPLE=None):
    """ Loads both features and binary labels from the disk."""
    X = scipy.sparse.load_npz("../temp/X_{}.npz".format(split))
    y_bin = {}
    for lvl in ["dom","mts","terms"]:
        y_bin[lvl] = scipy.sparse.load_npz("../temp/y_{}_bin_{}.npz".format(lvl,split))

    if N_SAMPLE:
        X = X[np.arange(N_SAMPLE),:]
        for lvl in ["dom","mts","terms"]:
            y_bin[lvl] = y_bin[lvl][np.arange(N_SAMPLE),:]

    return X, y_bin




def main():

    BATCH_SIZE=40
    N_EPOCHS= 10
    #
    MAX_WORDS = 30000
    MAX_DOC_LENGTH = 500
    EMBEDDING_DIM = 300
    
    #Set the size of the samples.
    N_SAMPLE_train = 5
    N_SAMPLE_test = 5


    filepath_eurlex_pt = "../data/clean_txt_withallwords.json"
    filepath_train_split = "../data/stratification/train_val.txt"
    filepath_test_split = "../data/stratification/test.txt"
    filepath_embeddings = "../data/embeddings/cbow_s300.txt"
    filepath_codes = "../data/codes/"
    filepath_results = "../results/"

    preprocessing = False
    run_model = True

    ###############################################################
    ###############################################################
    ###############################################################


    #Preprocessing.
    #You should execute run this code in the first run or if you change the training and testing splits.
    if preprocessing:

        celex2text = load_eurlex_pt(filepath_eurlex_pt)
        print("Corpus loaded.")

        celex_train,y_train = load_split(filepath_train_split,N_SAMPLE_train)
        celex_test,y_test = load_split(filepath_test_split,N_SAMPLE_test)
        print("Splits loaded.")    

        print("Creating sparse representation for the labels.")
        label_binarizer(y_train,y_test,filepath_codes)
        print("ys created and saved to disk.")

        corpus_train = [cleantext(celex2text[celex].replace("__SENT__"," ")) for celex in celex_train]
        corpus_test = [cleantext(celex2text[celex].replace("__SENT__"," ")) for celex in celex_test]

        word_index = create_word_index(corpus_train,MAX_WORDS)
        print("Word index created.")

        create_X(corpus_train,word_index,MAX_DOC_LENGTH,"X_train")
        print("X_train saved.")

        create_X(corpus_test,word_index,MAX_DOC_LENGTH,"X_test")
        print("X_test saved.")


    #Trains the model on X_train data. Uses X_test as the validation.
    #The trained models are stored at ../models. 
    #Plots on the model losses are stored at ../plots.
    if run_model:

        word_index = load_word_index()
        vocab_size = len(word_index) + 1

        X_train, y_bin_train = load_X_y("train", N_SAMPLE_train)
        X_test, y_bin_test = load_X_y("test", N_SAMPLE_test)

        embedding_matrix = create_embedding_matrix(filepath_embeddings,word_index,EMBEDDING_DIM)

        embedding_layer = Embedding(
            output_dim=EMBEDDING_DIM,
            input_dim=vocab_size,
            input_length=MAX_DOC_LENGTH,
            weights=[embedding_matrix],
            trainable=True,
            mask_zero=True)

        doc_input = Input(shape=(MAX_DOC_LENGTH,), dtype="int32",name = "main_input")
        embedded_sequences = embedding_layer(doc_input)
        l_lstm = Bidirectional(LSTM(256, activation = "pentanh",return_sequences=True))(embedded_sequences)
        l_lstm = Dropout(0.3)(l_lstm)
        l_att = AttentionWithContext()(l_lstm)
        l_att = Dense(512)(l_att)

        out_dom = Dense(y_bin_train["dom"].shape[1], activation="sigmoid", name="dense_out_dom")(l_att)

        merged_text_dom = keras.layers.concatenate([l_att, out_dom], axis= -1) #text,dom
        out_mts = Dense(y_bin_train["mts"].shape[1], activation="sigmoid", name="dense_out_mts")(merged_text_dom)

        merged_text_dom_mts = keras.layers.concatenate([l_att, out_dom, out_mts], axis= -1) #text,dom,mts
        out_terms = Dense(y_bin_train["terms"].shape[1], activation="sigmoid", name="dense_out_terms")(merged_text_dom_mts)

        model = Model(inputs=[doc_input],outputs=[out_dom, out_mts, out_terms])
        model.compile(optimizer="adam",
                      loss={
                          "dense_out_dom": "binary_crossentropy", 
                          "dense_out_mts": "binary_crossentropy", 
                          "dense_out_terms": "binary_crossentropy"},
                      loss_weights={
                          "dense_out_dom": 0.001, 
                          "dense_out_mts": 0.01, 
                          "dense_out_terms": 1})
        model.summary()

        #es = EarlyStopping(monitor="val_dense_out_terms_loss", mode="min", min_delta=0, verbose=1, patience=20)
        mc = ModelCheckpoint("../models/hierarchical.{epoch:02d}.h5", 
                             monitor="val_dense_out_terms_loss", 
                             mode="min", 
                             verbose=1, 
                             save_best_only=False)

        history = model.fit(x = {"main_input": X_train},
                            y = {"dense_out_dom": y_bin_train["dom"],
                                "dense_out_mts": y_bin_train["mts"],
                                "dense_out_terms": y_bin_train["terms"]},
                            validation_data = ({"main_input": X_test},
                                {"dense_out_dom": y_bin_test["dom"],
                                "dense_out_mts": y_bin_test["mts"],
                                "dense_out_terms": y_bin_test["terms"]}
                                ),
                            epochs=N_EPOCHS,
                            verbose=2,
                            batch_size=BATCH_SIZE,
                            callbacks = [mc]
                            #callbacks = [es,mc]
                            )

        y_pred = model.predict(X_test)



        idx2code = load_codes(filepath_codes)
        #Save EuroVoc predicted labels to disk.
        for i,lvl in enumerate(["dom","mts","terms"]):
            filepath_lvl_results = "{}{}.txt".format(filepath_results,lvl)
            save_results(y_bin_test[lvl],y_pred[i],idx2code[lvl],filepath_lvl_results)
        

        #Computes the precision at k for k=[1,3,5].
        for i,lvl in enumerate(["dom","mts","terms"]):
            for k in [1,3,5]:
                print(lvl,"P@k={}".format(k),precisionatk(y_bin_test[lvl],y_pred[i],k))


        #Plot main loss.
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.legend(["train", "test"], loc="upper right")
        plt.title("main loss")
        plt.savefig("../plots/main_loss.png")

        #Plot domains loss.
        plt.clf()
        plt.plot(history.history["dense_out_dom_loss"])
        plt.plot(history.history["val_dense_out_dom_loss"])
        plt.legend(["train", "test"], loc="upper right")
        plt.title("dom loss")
        plt.savefig("../plots/dom_loss.png")

        #Plot microthesauri loss.
        plt.clf()
        plt.plot(history.history["dense_out_mts_loss"])
        plt.plot(history.history["val_dense_out_mts_loss"])
        plt.legend(["train", "test"], loc="upper right")
        plt.title("mts loss")
        plt.savefig("../plots/mts_loss.png")

        #Plot descriptors loss.
        plt.clf()
        plt.plot(history.history["dense_out_terms_loss"])
        plt.plot(history.history["val_dense_out_terms_loss"])
        plt.legend(["train", "test"], loc="upper right")
        plt.title("terms loss")
        plt.savefig("../plots/terms_loss.png")


if __name__== "__main__":
    main()
