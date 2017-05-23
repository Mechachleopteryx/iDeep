import sys
import os
import numpy
#sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/Keras-0.3.1-py2.7.egg')
from keras.models import Sequential, model_from_config
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
#from keras.optimizers import kl_divergence
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import gzip
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib 
from scipy import sparse
import pdb
from math import  sqrt
from sklearn.metrics import roc_curve, auc
import theano
import subprocess as sp
import scipy.stats as stats
import argparse

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def merge_seperate_network(X_train1, X_train2, Y_train):
    left_hid = 128
    right_hid = 64
    left = get_rnn_fea(X_train1, sec_num_hidden = left_hid)
    right = get_rnn_fea(X_train2, sec_num_hidden = right_hid)
    
    model = Sequential()
    model.add(Merge([left, right], mode='concat'))
    total_hid = left_hid + right_hid
    
    model.add(Dense(total_hid, 2))
    model.add(Dropout(0.3))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd) #'rmsprop')
    
    model.fit([X_train1, X_train2], Y_train, batch_size=100, nb_epoch=100, verbose=0)
    
    return model

def get_blend_data(j, clf, skf, X_test, X_dev, Y_dev, blend_train, blend_test):
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)
            
            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]
            
            clf.fit(X_train, Y_train)
            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            #blend_train[cv_index, j] = clf.predict(X_cv)
            #blend_test_j[:, i] = clf.predict(X_test)
            blend_train[cv_index, j] = clf.predict_proba(X_cv)[:,1]
            blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)
    
        print 'Y_dev.shape = %s' % (Y_dev.shape)

def centrality_scores(X, alpha=0.85, max_iter=100, tol=1e-10):
    """Power iteration computation of the principal eigenvector

    This method is also known as Google PageRank and the implementation
    is based on the one from the NetworkX project (BSD licensed too)
    with copyrights by:

      Aric Hagberg <hagberg@lanl.gov>
      Dan Schult <dschult@colgate.edu>
      Pieter Swart <swart@lanl.gov>
    """
    X = sparse.csc_matrix(X)
    n = X.shape[0]
    X = X.copy()
    incoming_counts = np.asarray(X.sum(axis=1)).ravel()

    print("Normalizing the graph")
    for i in incoming_counts.nonzero()[0]:
        X.data[X.indptr[i]:X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
    dangle = np.asarray(np.where(X.sum(axis=1) == 0, 1.0 / n, 0)).ravel()

    scores = np.ones(n, dtype=np.float32) / n  # initial guess
    for i in range(max_iter):
        print("power iteration #%d" % i)
        prev_scores = scores
        scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                  + (1 - alpha) * prev_scores.sum() / n)
        # check convergence: normalized l_inf norm
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        print("error: %0.6f" % err)
        if err < n * tol:
            return scores

    return scores

def poweig(A, x0, maxiter = 100, ztol= 1.0e-5, mode= 0, teststeps=1):
    """
    Performs iterative power method for dominant eigenvalue.
     A  - input matrix.
     x0 - initial estimate vector.
     maxiter - maximum iterations
     ztol - zero comparison.
     mode:
       0 - divide by last nonzero element.
       1 - unitize.
    Return value:
     eigenvalue, eigenvector
    """
    m    = len(A)
    xi   = x0[:] 
 
    for n in range(maxiter):
       # matrix vector multiplication.
       xim1 = xi[:]
       for i in range(m):
           xi[i] = 0.0
           for j in range(m):
             xi[i] += A[i][j] * xim1[j]
       print n, xi
       if mode == 0:
          vlen = sqrt(sum([xi[k]**2 for k in range(m)]))
          xi = [xi[k] /vlen for k in range(m)]
       elif mode == 1:
          for k in range(m-1, -1, -1):
             c = abs(xi[k])
             if c > 1.0e-5:
                xi = [xi[k] /c for k in range(m)]
                break
       # early termination test.
       if n % teststeps == 0:
          S = sum([xi[k]-xim1[k] for k in range(m)])
          if abs(S) < ztol:
             break
       #print n, xi
    # Compute Rayleigh quotient.
    numer = sum([xi[k] * xim1[k] for k in range(m)])
    denom = sum([xim1[k]**2 for k in range(m)])
    xlambda = numer/denom
    return xlambda, xi

def get_larget_eign(cov_mat):
    evals, evecs = np.linalg.eigh(cov_mat)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    #evals = evals[idx]
    return evecs[0]

def get_meta_predictor(eg_array):
    normed, scl=preprocess_data(eg_array)
    covariance = np.cov(normed.T)
    #weights = centrality_scores(covariance)
    '''
    evals, evecs = np.linalg.eigh(cov_mat)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    '''
    '''
    x0 = np.array([1] * normed.shape[1])
    #pdb.set_trace()
    ramda, weights = poweig(covariance, x0)
    '''
    weights = get_larget_eign(covariance)
    weights = np.array(weights)
    weights = weights/weights.sum()
    #pdb.set_trace()
    ensemble_score = np.dot(eg_array, weights)
    #ensemble_score = [ 0 if x<0 else x for x in ensemble_score]
    return ensemble_score
    
    
def load_labels(path, kmer=True, rg=True, clip=True, rna=True, go=True):
    """
        Load column labels for data matrices.
    """

    labels = dict()
    if go: labels["X_GO"]   = gzip.open(os.path.join(path,
                        "matrix_GeneOntology.tab.gz")).readline().split("\t")
    if kmer: labels["X_KMER"] = gzip.open(os.path.join(path,
                        "matrix_RNAkmers.tab.gz")).readline().split("\t")
    if rg: labels["X_RG"]   = gzip.open(os.path.join(path,
                        "matrix_RegionType.tab.gz")).readline().split("\t")
    if clip: labels["X_CLIP"] = gzip.open(os.path.join(path,
                        "matrix_Cobinding.tab.gz")).readline().split("\t")
    if rna: labels["X_RNA"]  = gzip.open(os.path.join(path,
                        "matrix_RNAfold.tab.gz")).readline().split("\t")
    return labels

def read_seq(seq_file):
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq_array)                    
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = get_RNA_seq_concolutional_array(seq)
            seq_list.append(seq_array) 
    
    return np.array(seq_list)

def read_oli_feature(seq_file):
    trids4 = get_4_trids()
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_4_nucleotide_composition(trids4, seq)
                    seq_list.append(seq_array)                    
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = get_4_nucleotide_composition(trids4, seq)
            seq_list.append(seq_array) 
    
    return np.array(seq_list)    

def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return  nucle_com

def get_4_nucleotide_composition(tris, seq, pythoncount = True):
    #pdb.set_trace()
    seq_len = len(seq)
    seq = seq.upper().replace('T', 'U')
    tri_feature = []
    
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num)/seq_len)
    else:
        k = len(tris[0])
        tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1- k):
            kmer = seq[x:x+k]
            if kmer in tris:
                ind = tris.index(kmer)
                tmp_fea[ind] = tmp_fea[ind] + 1
        tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return tri_feature


def load_data(path, kmer=False, rg=True, clip=True, rna=True, go=False, motif= True, seq = True, oli = False, test = False):
    """
        Load data matrices from the specified folder.
    """

    data = dict()
    if go:   data["X_GO"]   = np.loadtxt(gzip.open(os.path.join(path,
                                            "matrix_GeneOntology.tab.gz")),
                                            skiprows=1)
    if kmer: data["X_KMER"] = np.loadtxt(gzip.open(os.path.join(path,
                                            "matrix_RNAkmers.tab.gz")),
                                            skiprows=1)
    if rg:   data["X_RG"]   = np.loadtxt(gzip.open(os.path.join(path,
                                            "matrix_RegionType.tab.gz")),
                                            skiprows=1)
    if clip: data["X_CLIP"] = np.loadtxt(gzip.open(os.path.join(path,
                                            "matrix_Cobinding.tab.gz")),
                                            skiprows=1)
    if rna:  data["X_RNA"]  = np.loadtxt(gzip.open(os.path.join(path,
                                            "matrix_RNAfold.tab.gz")),
                                            skiprows=1)
    if motif: data["motif"] = np.loadtxt(gzip.open(os.path.join(path, 'motif_fea.gz'))
                                         , skiprows=1, usecols=range(1,103))
    if seq: data["seq"] = read_seq(os.path.join(path, 'sequences.fa.gz'))
    if oli: data["oli"] = read_oli_feature(os.path.join(path, 'sequences.fa.gz'))
    if test:
        data["Y"] = []
    else:    
        data["Y"] = np.loadtxt(gzip.open(os.path.join(path,
                                            "matrix_Response.tab.gz")),
                                            skiprows=1)
    #data["Y"] = data["Y"].reshape((len(data["Y"]), 1))

    return data

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement[base] for base in seq]
    return complseq

def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))
    
def get_hg19_sequence():
    chr_focus = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
     'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14',
     'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21',
     'chr22', 'chrX', 'chrY', 'chrM']
    
    sequences = {}
    dir1 = 'hg19_seq/'
    #for chr_name in chr_foc
    for chr_name in chr_focus:
        file_name = chr_name + '.fa.gz'
        if not os.path.exists(dir1 + file_name):
            print 'download genome sequence file'
            cli_str = 'rsync -avzP rsync://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/' + chr_name + '.fa.gz ' + dir1
            fex = os.popen(cli_str, 'r')
            fex.close()
        
        print 'file %s' %file_name
        fp = gzip.open(dir1 + file_name, 'r')
        sequence = ''
        for line in fp:
            if line[0] == '>':
                name = line.split()[0]
            else:
                sequence = sequence + line.split()[0]
        sequences[chr_name] =  sequence 
        fp.close()
    
    return sequences

def get_seq_for_RNA_bed(RNA_bed_file, whole_seq):
    print RNA_bed_file
    fp = gzip.open(RNA_bed_file, 'r')
    fasta_file = RNA_bed_file.split('.')[0] + '.fa.gz'
    fw = gzip.open(fasta_file, 'w')
    
    for line in fp:
        if 'tract' in line:
            continue
        values = line.split()
        chr_n = values[0]
        start = int(values[1])
        end = int(values[2])
        #gene_name = values[4]
        strand = values[3]
        seq = whole_seq[chr_n]
        extract_seq = seq[start-50: start + 51]
        extract_seq = extract_seq.upper()
        if strand == '-':
            extract_seq =  reverse_complement(extract_seq)
        
        #fw.write('>%s\t%s\t%s\t%s\t%s\n' %(gene_name, chr_n, start, end, strand))
        fw.write('%s\n'%extract_seq)
            
    fp.close()
    fw.close()

def preprocess_data(X, scaler=None, stand = False):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler    

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def get_2d_cnn_network():
    nb_conv = 4
    nb_pool = 2
    model = Sequential()
    model.add(Convolution2D(64, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, 107, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    
    return model

def get_cnn_network():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    print 'configure cnn network'
    nbfilter = 102
    #forward_lstm = LSTM(input_dim=nbfilter, output_dim=nbfilter, return_sequences=True)
    #backward_lstm = LSTM(input_dim=nbfilter, output_dim=nbfilter, return_sequences=True)
    #brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)
    #brnn = Merge([forward_lstm, backward_lstm], mode='concat', concat_axis=-1)

    model = Sequential()
    model.add(Convolution1D(input_dim=4,input_length=107,
                            nb_filter=nbfilter,
                            filter_length=7,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))
    
    model.add(MaxPooling1D(pool_length=3))
    
    model.add(Dropout(0.5))

    #model.add(brnn)

    #model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(nbfilter, activation='relu'))
    #model.add(Activation('relu'))
    #model.add(PReLU())
    #model.add(BatchNormalization(mode=2))
    #model.add(Dense(64))
    model.add(Dropout(0.25))
    
    #model.fit(X_train, y_train)
    
    return model

def get_rnn_fea(train, sec_num_hidden = 128, num_hidden = 128):
    print 'configure network for', train.shape
    model = Sequential()

    #model.add(Dense(num_hidden, input_dim=train.shape[1], activation='relu'))
    model.add(Dense(num_hidden, input_shape=(train.shape[1],), activation='relu'))
    model.add(PReLU())
    model.add(BatchNormalization(mode=2))
    model.add(Dropout(0.5))
    model.add(Dense(num_hidden, input_dim=num_hidden, activation='relu'))
    #model.add(Dense(num_hidden, input_shape=(num_hidden,), activation='relu'))
    model.add(PReLU())
    model.add(BatchNormalization(mode=2))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    '''
    model.add(Dense(sec_num_hidden, input_shape=(num_hidden,), activation='relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    '''
    return model


def get_features():
    all_weights = []
    for layer in model.layers:
       w = layer.get_weights()
       all_weights.append(w)
       
    return all_weights

def run_network(model, total_hid, training, testing, y, validation, val_y, protein=None):
    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print 'model training'
    checkpointer = ModelCheckpoint(filepath="models/" + protein + "_bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=50, nb_epoch=30, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])
    
    #model.fit(training, y, batch_size=50, nb_epoch=20, verbose=0)
    
    predictions = model.predict_proba(testing)[:,1]
    return predictions, model

def run_randomforest_classifier(data, labels, test):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(data, labels)
    #pdb.set_trace()
    pred_prob = clf.predict_proba(test)[:,1]
    return pred_prob, clf  

def run_svm_classifier(data, labels, test):
    C_range = 10.0 ** np.arange(-1, 2)
    param_grid = dict(C=C_range.tolist())
    svr = svm.SVC(probability =True, kernel = 'linear')
    grid = GridSearchCV(svr, param_grid)
    grid.fit(data, labels)
    
    clf = grid.best_estimator_
    pred_prob = clf.predict_proba(test)[:,1]
    return pred_prob, clf
    
def calculate_auc(net, hid, train, test, true_y, train_y, rf = False, validation = None, val_y = None, protein = None):
    #print 'running network' 
    if rf:
        print 'running oli'
        #pdb.set_trace()
        predict, model = run_svm_classifier(train, train_y, test)
    else:
        predict, model = run_network(net, hid, train, test, train_y, validation, val_y, protein = protein)
        #
        
        #get_feature = theano.function([model.layers[0].input],model.layers[7].get_output(train=False),allow_input_downcast=True)
        #train = get_feature(train)
        #test = get_feature(test)
        #plt.imshow(test,cmap = cm.Greys_r)
        #plt.show()
        '''
        #pdb.set_trace()
        real_labels = []
        for val in train_y:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
        predict, model = run_randomforest_classifier(train, real_labels, test)
        '''
    
    auc = roc_auc_score(true_y, predict)
    
    print "Test AUC: ", auc
    return auc, predict

def run_individual_network(protein, kmer=True, rg=True, clip=True, rna=True, go=False, motif = True, seq = True, oli = False, fw = None):
    training_data = load_data("datasets/clip/%s/5000/training_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, motif=motif, 
                              seq = seq, oli = oli)
    print 'training', len(training_data)
    go_hid = 512
    kmer_hid = 512
    rg_hid = 128
    clip_hid = 256
    rna_hid=64
    motif_hid = 64
    seq_hid = 102
    oli_hid = 64
    train_Y = training_data["Y"]
    #pdb.set_trace()
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    if go:
        go_data, go_scaler = preprocess_data(training_data["X_GO"])
        go_train = go_data[training_indice]
        go_validation = go_data[validation_indice]
        go_net = get_rnn_fea(go_train, sec_num_hidden = go_hid, num_hidden = go_hid*4)
        go_data = []
    if kmer:
        kmer_data, kmer_scaler = preprocess_data(training_data["X_KMER"])
        kmer_train = kmer_data[training_indice]
        kmer_validation = kmer_data[validation_indice]        
        kmer_net = get_rnn_fea(kmer_train, sec_num_hidden = kmer_hid, num_hidden = kmer_hid*4)
        kmer_data = []
    if rg:
        rg_data, rg_scaler = preprocess_data(training_data["X_RG"])
        rg_train = rg_data[training_indice]
        rg_validation = rg_data[validation_indice]
        rg_net = get_rnn_fea(rg_train, sec_num_hidden = rg_hid, num_hidden = rg_hid*2)
        rg_data = []
    if clip:
        clip_data, clip_scaler = preprocess_data(training_data["X_CLIP"])
        clip_train = clip_data[training_indice]
        clip_validation = clip_data[validation_indice]
        clip_net = get_rnn_fea(clip_train, sec_num_hidden = clip_hid, num_hidden = clip_hid*3)
        clip_data = []
    if rna:
        rna_data, rna_scaler = preprocess_data(training_data["X_RNA"])
        rna_train = rna_data[training_indice]
        rna_validation = rna_data[validation_indice]        
        rna_net = get_rnn_fea(rna_train, sec_num_hidden = rna_hid, num_hidden = rna_hid*2)
        rna_data = []
    if motif:
        motif_data, motif_scaler = preprocess_data(training_data["motif"], stand = True)
        motif_train = motif_data[training_indice]
        motif_validation = motif_data[validation_indice] 
        motif_net = get_rnn_fea(motif_train, sec_num_hidden = motif_hid, num_hidden = motif_hid*2)
        #seq_net =  get_cnn_network()
        motif_data = []
    if seq:
        seq_data = training_data["seq"]
        seq_train = seq_data[training_indice]
        seq_validation = seq_data[validation_indice] 
        seq_net =  get_cnn_network()
        seq_data = []
    if oli:
        oli_data, oli_scaler = preprocess_data(training_data["oli"], stand = True)
        oli_train = oli_data[training_indice]
        oli_validation = oli_data[validation_indice] 
        oli_net = get_rnn_fea(oli_train, sec_num_hidden = oli_hid, num_hidden = oli_hid*2)
        #seq_net =  get_cnn_network()
        #oli_data = []
                   
    rf = False
    if oli:
        rf = True
    if not rf:   
        #all_label =  training_data["Y"]   
        y, encoder = preprocess_labels(training_label)
        val_y, encoder = preprocess_labels(validation_label, encoder = encoder)
    else:
        y = training_label
        val_y = validation_label
    
    training_data.clear()
    
    
    test_data = load_data("datasets/clip/%s/5000/test_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, motif=motif, 
                          seq = seq, oli = oli)
    
    true_y = test_data["Y"].copy()
    
    print 'predicting'
    
    eg_array = []
    if go:
        go_test, go_scaler = preprocess_data(test_data["X_GO"], scaler=go_scaler)
        #testing.append(go_test)
        go_auc, go_predict = calculate_auc(go_net, go_hid, go_train, go_test, true_y, y, validation = go_validation, 
                                           val_y = val_y, protein = protein,  rf= True)
        eg_array.append(go_predict)
        go_train = []
        go_test = []
    if kmer:
        kmer_test, kmer_scaler = preprocess_data(test_data["X_KMER"], scaler=kmer_scaler)
        kmer_auc, kmer_predict = calculate_auc(kmer_net, kmer_hid, kmer_train, kmer_test, true_y, y, validation = kmer_validation, 
                                           val_y = val_y, protein = protein,  rf= rf)
        kmer_train = []
        kmer_test = []
        eg_array.append(kmer_predict)
    if rg:
        rg_test, rg_scaler = preprocess_data(test_data["X_RG"], scaler=rg_scaler)
        rg_auc, rg_predict = calculate_auc(rg_net, rg_hid, rg_train, rg_test, true_y, y, validation = rg_validation, 
                                           val_y = val_y, protein = protein,  rf= rf)
        rg_train  = []
        rg_test = []
        eg_array.append(rg_predict)
    if clip:
        clip_test, clip_scaler = preprocess_data(test_data["X_CLIP"], scaler=clip_scaler)
        clip_auc, clip_predict = calculate_auc(clip_net, clip_hid, clip_train, clip_test, true_y, y, validation = clip_validation, 
                                               val_y = val_y, protein = protein,  rf= rf)
        clip_train  = []
        clip_test = [] 
        eg_array.append(clip_predict)
    if rna:
        rna_test, rna_scaler = preprocess_data(test_data["X_RNA"], scaler=rna_scaler, stand = True)
        rna_auc, rna_predict = calculate_auc(rna_net, rna_hid, rna_train, rna_test, true_y, y, validation = rna_validation, 
                                             val_y = val_y, protein = protein,  rf= rf)
        rna_train  = []
        rna_test = []        
        eg_array.append(rna_predict)
    if motif:
        motif_test, motif_scaler = preprocess_data(test_data["motif"], scaler=motif_scaler, stand = True)
        motif_auc, motif_predict = calculate_auc(motif_net, motif_hid, motif_train, motif_test, true_y, y, validation = motif_validation,
                                              val_y = val_y, protein = protein,  rf= rf)
        motif_train = []
        motif_test = []
        eg_array.append(motif_predict)
    if seq:
        seq_test = test_data["seq"]
        seq_auc, seq_predict = calculate_auc(seq_net, seq_hid, seq_train, seq_test, true_y, y, validation = seq_validation,
                                              val_y = val_y, protein = protein,  rf= rf)
        seq_train = []
        seq_test = []
        eg_array.append(seq_predict)
    if oli:
        oli_test, oli_scaler = preprocess_data(test_data["oli"], scaler=oli_scaler, stand = True)
        oli_auc, oli_predict = calculate_auc(oli_net, oli_hid, oli_data, oli_test, true_y, train_Y,
                                              val_y = val_y, protein = protein,  rf= rf)
        oli_train = []
        oli_test = []
        eg_array.append(oli_predict)
                
    test_data.clear()
    if 1:
        if oli:
            print oli_auc
        if seq:
            print seq_auc
        #print seq_auc, motif_auc, rg_auc, clip_auc, 
    else:
    	eg_array = np.array(eg_array).T
    	print eg_array.shape 

        
    	print str(rg_auc) + '\t' +  str(clip_auc) + '\t' +  str(rna_auc) +'\t'  + str(motif_auc) +'\t'  + str(seq_auc)
    	fw.write(str(rg_auc) + '\t' +  str(clip_auc) + '\t' +  str(rna_auc) + '\t'  + str(motif_auc) + '\t'  + str(seq_auc) +'\n')
    
    	mylabel = "\t".join(map(str, true_y))
    	#myprob1 = "\t".join(map(str, kmer_predict))
    	myprob2 = "\t".join(map(str, rg_predict))
    	myprob3 = "\t".join(map(str, clip_predict))
    	myprob4 = "\t".join(map(str, rna_predict))
    	myprob5 = "\t".join(map(str, motif_predict))
        myprob6 = "\t".join(map(str, seq_predict))
        
    	fw.write(mylabel + '\n')
    	#fw.write(myprob1 + '\n')
    	fw.write(myprob2 + '\n')
    	fw.write(myprob3 + '\n')
    	fw.write(myprob4 + '\n')
    	fw.write(myprob5 + '\n')
        fw.write(myprob6 + '\n')
        

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label        
        
def merge_seperate_network_with_multiple_features(protein, kmer=False, rg=True, clip=True, rna=True, go=False, motif = True, seq = True, fw = None):
    training_data = load_data("datasets/clip/%s/5000/training_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, motif=motif, seq = seq)
    print 'training', len(training_data)
    go_hid = 512
    kmer_hid = 512
    rg_hid = 128
    clip_hid = 256
    rna_hid=64
    cnn_hid = 64
    motif_hid = 64
    seq_hid = 102
    training_indice, training_label, validation_indice, validation_label = split_training_validation(training_data["Y"])
    #x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)
    if go:
        go_data, go_scaler = preprocess_data(training_data["X_GO"])
        go_train = go_data[training_indice]
        go_validation = go_data[validation_indice]
        go_net = get_rnn_fea(go_train, sec_num_hidden = go_hid, num_hidden = go_hid*4)
        go_data = []
        training_data["X_GO"] = []
    if kmer:
        kmer_data, kmer_scaler = preprocess_data(training_data["X_KMER"])
        kmer_train = kmer_data[training_indice]
        kmer_validation = kmer_data[validation_indice]
        kmer_net = get_rnn_fea(kmer_train, sec_num_hidden = kmer_hid, num_hidden = kmer_hid*4)
        kmer_data = []
        training_data["X_KMER"] = []
    if rg:
        rg_data, rg_scaler = preprocess_data(training_data["X_RG"])
        rg_train = rg_data[training_indice]
        rg_validation = rg_data[validation_indice]
        rg_net = get_rnn_fea(rg_train, sec_num_hidden = rg_hid, num_hidden = rg_hid*2)
        rg_data = []
        training_data["X_RG"] = []
    if clip:
        clip_data, clip_scaler = preprocess_data(training_data["X_CLIP"])
        clip_train = clip_data[training_indice]
        clip_validation = clip_data[validation_indice]
        clip_net = get_rnn_fea(clip_train, sec_num_hidden = clip_hid, num_hidden = clip_hid*3)
        clip_data = []
        training_data["X_CLIP"] = []
    if rna:
        rna_data, rna_scaler = preprocess_data(training_data["X_RNA"], stand = True)
        rna_train = rna_data[training_indice]
        rna_validation = rna_data[validation_indice]        
        rna_net = get_rnn_fea(rna_train, sec_num_hidden = rna_hid, num_hidden = rna_hid*2)
        rna_data = []
        training_data["X_RNA"] = []
    if motif:
        motif_data, motif_scaler = preprocess_data(training_data["motif"], stand = True)
        motif_train = motif_data[training_indice]
        motif_validation = motif_data[validation_indice]
        motif_net =  get_rnn_fea(motif_train, sec_num_hidden = motif_hid, num_hidden = motif_hid*2) #get_cnn_network()
        motif_data = []
        training_data["motif"] = []
    if seq:
        seq_data = training_data["seq"]
        seq_train = seq_data[training_indice]
        seq_validation = seq_data[validation_indice] 
        seq_net =  get_cnn_network()
        seq_data = []         
        
    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder)
    training_data.clear()
    
    model = Sequential()
    training_net=[]
    training =[]
    validation = []
    total_hid =0
    if go:
        training_net.append(go_net)
        training.append(go_train)
        validation.append(go_validation)
        total_hid = total_hid + go_hid
        go_train = []
        go_validation = []
    if kmer:
        training_net.append(kmer_net)
        training.append(kmer_train)
        validation.append(kmer_validation)
        total_hid = total_hid + kmer_hid
        kmer_train = []
        kmer_validation = []
    if rg:
        training_net.append(rg_net)
        training.append(rg_train)
        validation.append(rg_validation)
        total_hid = total_hid + rg_hid
        rg_train = []
        rg_validation = []
    if clip:
        training_net.append(clip_net)
        training.append(clip_train)
        validation.append(clip_validation)
        total_hid = total_hid + clip_hid
        clip_train = []
        clip_validation = []
    if rna:
        training_net.append(rna_net)
        training.append(rna_train)
        validation.append(rna_validation)
        total_hid = total_hid + rna_hid
        rna_train = []
        rna_validation = []
    if motif:
        training_net.append(motif_net)
        training.append(motif_train)
        validation.append(motif_validation)
        total_hid = total_hid + motif_hid
        motif_train = []
        motif_validation = []
    if seq:
        training_net.append(seq_net)
        training.append(seq_train)
        validation.append(seq_validation)
        total_hid = total_hid + seq_hid
        seq_train = []
        seq_validation = []        
        
    model.add(Merge(training_net, mode='concat'))
    #model.add(Dense(total_hid, input_shape=(total_hid,)))
    #model.add(Activation('relu'))
    #model.add(PReLU())
    #model.add(BatchNormalization(mode=2))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    
    #checkpointer = ModelCheckpoint(filepath="models/bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    #validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']), callbacks=[checkpointer,earlystopper]
    print 'model training'
    model.fit(training, y, batch_size=100, nb_epoch=20, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])
    
    training = []
    validation = []
    
    test_data = load_data("datasets/clip/%s/5000/test_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, motif=motif, seq = seq)
    
    true_y = test_data["Y"].copy()
    
    print 'predicting'
    testing = []
    if go:
        go_test, go_scaler = preprocess_data(test_data["X_GO"], scaler=go_scaler)
        testing.append(go_test)
    if kmer:
        kmer_test, kmer_scaler = preprocess_data(test_data["X_KMER"], scaler=kmer_scaler)
        testing.append(kmer_test)
    if rg:
        rg_test, rg_scaler = preprocess_data(test_data["X_RG"], scaler=rg_scaler)
        testing.append(rg_test)
    if clip:
        clip_test, clip_scaler = preprocess_data(test_data["X_CLIP"], scaler=clip_scaler)
        testing.append(clip_test)
    if rna:
        rna_test, rna_scaler = preprocess_data(test_data["X_RNA"], scaler=rna_scaler, stand = True)
        testing.append(rna_test)
    if motif:
        motif_test, motif_scaler = preprocess_data(test_data["motif"], scaler=motif_scaler, stand = True)
        testing.append(motif_test)
    if seq:
        seq_test = test_data["seq"]
        testing.append(seq_test)
        
    '''
    pdb.set_trace()
    get_feature = theano.function([model.layers[0].input],model.layers[8].get_output(train=False),allow_input_downcast=True)
    feature_plot = get_feature(test)
    plt.imshow(feature_plot,cmap = cm.Greys_r)
    plt.show()
    '''    
    predictions = model.predict_proba(testing)
    #pdb.set_trace()
    auc = roc_auc_score(true_y, predictions[:, 1])
    print "Test AUC: ", auc    
    fw.write(str(auc) + '\n')
    mylabel = "\t".join(map(str, true_y))
    myprob = "\t".join(map(str, predictions[:, 1]))
    fw.write(mylabel + '\n')
    fw.write(myprob + '\n')
    
    return model


def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )

def read_protein_name(filename='proteinnames'):
    protein_dict = {}
    with open(filename, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            key_name = values[0][1:-1]
            protein_dict[key_name] = values[1]
    return protein_dict
    
def read_result_file(filename = 'result_file_seq_wohle_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            if index % 3 == 0:
                protein = values[0].split('_')[0]
            if index % 3 != 0:
                results.setdefault(protein, []).append(values)
                
                
            index = index + 1
    
    return results

def plot_parameter_bar(menMeans, xlabel):
    methodlabel = ['region type', 'clip-cobinding', 'structure', 'motif', 'CNN sequence', 'iDeep']
    
    #xval = [5, 10, 20, 30, 40, 50, 60,70]#, 80, 90]
    width = 0.10
    ind = np.arange(len(menMeans[0]))
    fig, ax = plt.subplots(figsize=(12,12))
    #pdb.set_trace()
    #plt.plot(xval,menMeans)
    rects1 = plt.barh(ind, menMeans[0], width, color='r')
    rects2 = plt.barh(ind +width, menMeans[1], width, color='g')
    rects3 = plt.barh(ind +2*width, menMeans[2], width, color='y')
    rects4 = plt.barh(ind+3*width, menMeans[3], width, color='b')
    rects5 = plt.barh(ind+4*width, menMeans[4], width, color='m')
    rects6 = plt.barh(ind+5*width, menMeans[5], width, color='c')
    #plt.title('stem cell circRNA vs other circRNA')
    ax.set_xlabel('AUC', fontsize=20)
    #plt.xlabel('Number of trees', fontsize=20)
    #ax.set_ylim([0.6, 0.75])
    ax.set_yticks(ind)
    ax.set_yticklabels(xlabel )
    #plt.margins(0.1)
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0],  rects6[0]), ('region type', 'clip-cobinding', 'structure', 'motif', 'CNN sequence', 'iDeep'), 
              loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True)
    plt.tight_layout()
    
    plt.show()

def plot_confusion_matrix(results, title='Confusion matrix'):
    '''plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()
    '''
    unique_conditions = ['region type', 'clip-cobinding', 'structure', 'motif', 'CNN sequence', 'kmer']
    confusion = []
    for i in range(len(results)):
        tmp = []
        for j in range(len(results)):
            rval, pval = stats.pearsonr(results[i], results[j])
            tmp.append(abs(rval))
        confusion.append(tmp)
    #pdb.set_trace()
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #res = ax.imshow(np.array(norm_conf), cmap=confusion_matrix.jet, interpolation='nearest')
    res = ax.imshow(np.array(confusion), cmap=plt.cm.jet, 
                interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks(np.arange(len(unique_conditions)), unique_conditions)
    plt.yticks(np.arange(len(unique_conditions)), unique_conditions)
    for i, cas in enumerate(confusion ):
        for j, c in enumerate(cas):
            if c>0:
                plt.text(j-.2, i+.2, round(c, 2), fontsize=14)
    plt.title('Correlation between different modalities')
    
    plt.show() 

    #plot_confusion_matrix(df_confusion)

def plot_scatter(new_results):
    #resuts =[]
    region = new_results[0]
    cnn = new_results[4]
    motif = new_results[3]
    inds = range(len(motif))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #pdb.set_trace()
    p1 = plt.plot(inds, cnn, marker='x', color='r')
    p2 = plt.plot(region, inds, marker='o', color='y')
    #plt.legend( (p1, p2), ('stem cell circRNAs', 'other circRNAs'), 1)
    #plt.xlim(-2, 6)
    #plt.xlim(0,14000)
    
    #plt.xlim(0,0.2)
    plt.xlabel('Sequence using CNN')
    plt.ylabel('k-mer using DBN')
    #plt.legend( (p1[0], p2[0]), ('circularRNA', 'other lncRNAs'), 2)
    plt.show()
    
def read_individual_auc(filename = 'result_file_all_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            pro = values[0].split('_')[0]
            results[int(pro)] = values[1:-1]    
    
    return results

def read_ideep_auc(filename = 'result_mix_auc_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            #pdb.set_trace()
            pro = values[0].split('_')[0]
            results[int(pro)] = values[1]  
    
    return results

def plot_ideep_indi_comp():
    proteins = read_protein_name()
    ideep_resut = read_ideep_auc(filename='result_mix_auc_new')
    #pdb.set_trace()
    indi_result = read_individual_auc()
    keys = indi_result.keys()
    keys.sort()
    
    new_results = []
    names = []
    for key in keys:
        str_key = str(key)
        names.append(proteins[str_key])
        tmp = []
        for val in indi_result[key]:
            tmp.append(float(val))
        #for val in ideep_resut[key]:
        tmp.append(float(ideep_resut[key]))
        #tmp = indi_result[key] + ideep_resut[key]
        new_results.append(tmp)
    pdb.set_trace()
    new_results = map(list, zip(*new_results))
    #plot_confusion_matrix(new_results)
    plot_parameter_bar(new_results, names)
            
def plot_figure():
    protein_dict = read_protein_name()
    results = read_result_file()
    
    Figure = plt.figure(figsize=(12, 15))
    
    for key, values in results.iteritems():
        protein = protein_dict[key]
        #pdb.set_trace()
        labels = [int(float(val)) for val in values[0]]
        probability = [float(val) for val in values[1]]
        plot_roc_curve(labels, probability, protein)
    #plot_roc_curve(labels[1], probability[1], '')
    #plot_roc_curve(labels[2], probability[2], '')
    
    #title_type = 'stem cell circRNAs vs other circRNAs'
    title_type = 'ROC'
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(title_type)
    plt.legend(loc="lower right")
    plt.savefig('roc1.eps', format='eps') 
    #plt.show() 

#test cnn    
def run_cnn():
    X_train = []
    seqs =50*['CGUACACGGUGGAUGCCCUGGCAGUCAAGGCGAUGAAGGACGUGCUAAUCUGCGAUAAGCGUCGGUAAGGUGAUAUGAACCGUUUAACCGGCGAUUUCCGC', 'GGCGGCCGUAGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAACGCCGUAGCGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGG']
    for seq in seqs:
        tmp_train = get_RNA_seq_concolutional_array(seq)
        X_train.append(tmp_train)
    #pdb.set_trace()
    model = get_cnn_network()
    y_train = np.array([0, 1]*50).T
    y_train, encoder = preprocess_labels(y_train)
    
    print len(y_train)
    #y_train = np.array([0, 1])
    model.add(Dense(input_dim=64, output_dim=2))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
    #(455024, 1000, 4)
    #pdb.set_trace()
    model.fit(np.array(X_train), y_train, batch_size=50, nb_epoch=50, verbose=0)
    pdb.set_trace()

def run_get_sequence():
    whole_seq = get_hg19_sequence()
    data_dir = '/home/panxy/eclipse/ideep/datasets/clip/'
    protein_dirs = os.listdir(data_dir)
    for protein in protein_dirs:
        new_dir = data_dir + protein + '/5000/'
        for beddir in os.listdir(new_dir):
            RNA_bed_file = new_dir + beddir + '/positions.bedGraph.gz'
            #pdb.set_trace()
            get_seq_for_RNA_bed(RNA_bed_file, whole_seq)
            
def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = gzip.open(fasta_file, 'r')
    name = ''
    name_list = []
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[2:] #discarding the initial >
            name_list.append(name)
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper().replace('U', 'T')
    fp.close()
    
    return seq_dict, name_list

def remove_some_files():
    data_dir = '/home/panxy/eclipse/ideep/datasets/clip/'
    protein_dirs = os.listdir(data_dir)
    for protein in protein_dirs:
        new_dir = data_dir + protein + '/5000/'
        for beddir in os.listdir(new_dir):
            print beddir
            #pdb.set_trace()
            path = new_dir + beddir
            fas_name = os.path.join(path, 'matrix_GeneOntology.tab.gz')
            os.remove(fas_name)
            
def get_binding_motif_fea():
    data_dir = '/home/panxy/eclipse/ideep/datasets/clip/'
    protein_dirs = os.listdir(data_dir)
    for protein in protein_dirs:
        new_dir = data_dir + protein + '/5000/'
        for beddir in os.listdir(new_dir):
            print beddir
            #pdb.set_trace()
            path = new_dir + beddir
            fas_name = os.path.join(path, 'sequences.fa')
            fw = open(fas_name, 'w')
            seq_file = os.path.join(path, 'sequences.fa.gz')
            seq_dict, name_list = read_fasta_file(seq_file)
            
            for name in name_list:
                values = name.rstrip().split(';')
                posi = values[0].split(',')
                coor = posi[0] + ':' + posi[2] + '-' + posi[3]
                fw.write('>' + coor + '\n')
                fw.write(seq_dict[name] + '\n') 
            fw.close()
            
            #pdb.set_trace()
            clistr = './get_RNA_motif_fea.sh ' + path + '>/dev/null 2>&1'
            f_cli = os.popen(clistr, 'r')
            f_cli.close()

def run_predict():
    data_dir = 'datasets/clip'
    fw = open('result_file', 'w')
    for protein in os.listdir(data_dir):
        print protein
        fw.write(protein + '\t')
        model = merge_seperate_network_with_multiple_features(protein, kmer=False, rg=True, clip=True, rna=True, motif = True, seq = True, fw = fw)
    fw.close()

def calculate_perofrmance(inputfile='../comp_result'):
    result = []
    with open(inputfile) as fp:
        for line in fp:
            values = line.rstrip().split('&')
            result.append([float(values[0]), float(values[1])])
    res_array = numpy.array(result)
    print np.mean(res_array, axis=0)
    print np.std(res_array, axis=0)



def train_ideep(data_dir, model_dir, rg=True, clip=True, rna=True, motif = False, seq = True, batch_size=100, nb_epoch=20):
    training_data = load_data(data_dir, rg=rg, clip=clip, rna=rna, motif=motif, seq = seq)
    print 'training', len(training_data)
    rg_hid = 128
    clip_hid = 256
    rna_hid=64
    cnn_hid = 64
    motif_hid = 64
    seq_hid = 102
    training_indice, training_label, validation_indice, validation_label = split_training_validation(training_data["Y"])
    if rg:
        rg_data, rg_scaler = preprocess_data(training_data["X_RG"])
        joblib.dump(rg_scaler, os.path.join(model_dir,'rg_scaler.pkl')) 
        rg_train = rg_data[training_indice]
        rg_validation = rg_data[validation_indice]
        rg_net = get_rnn_fea(rg_train, sec_num_hidden = rg_hid, num_hidden = rg_hid*2)
        rg_data = []
        training_data["X_RG"] = []
    if clip:
        clip_data, clip_scaler = preprocess_data(training_data["X_CLIP"])
        joblib.dump(rg_scaler, os.path.join(model_dir,'clip_scaler.pkl')) 
        clip_train = clip_data[training_indice]
        clip_validation = clip_data[validation_indice]
        clip_net = get_rnn_fea(clip_train, sec_num_hidden = clip_hid, num_hidden = clip_hid*3)
        clip_data = []
        training_data["X_CLIP"] = []
    if rna:
        rna_data, rna_scaler = preprocess_data(training_data["X_RNA"], stand = True)
        joblib.dump(rg_scaler, os.path.join(model_dir,'rna_scaler.pkl')) 
        rna_train = rna_data[training_indice]
        rna_validation = rna_data[validation_indice]        
        rna_net = get_rnn_fea(rna_train, sec_num_hidden = rna_hid, num_hidden = rna_hid*2)
        rna_data = []
        training_data["X_RNA"] = []
    if motif:
        motif_data, motif_scaler = preprocess_data(training_data["motif"], stand = True)
        joblib.dump(rg_scaler, os.path.join(model_dir,'motif_scaler.pkl'))
        motif_train = motif_data[training_indice]
        motif_validation = motif_data[validation_indice]
        motif_net =  get_rnn_fea(motif_train, sec_num_hidden = motif_hid, num_hidden = motif_hid*2) #get_cnn_network()
        motif_data = []
        training_data["motif"] = []
    if seq:
        seq_data = training_data["seq"]
        seq_train = seq_data[training_indice]
        seq_validation = seq_data[validation_indice] 
        seq_net =  get_cnn_network()
        seq_data = []         
        
    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder)
    training_data.clear()
    
    model = Sequential()
    training_net=[]
    training =[]
    validation = []
    total_hid =0
    if rg:
        training_net.append(rg_net)
        training.append(rg_train)
        validation.append(rg_validation)
        total_hid = total_hid + rg_hid
        rg_train = []
        rg_validation = []
    if clip:
        training_net.append(clip_net)
        training.append(clip_train)
        validation.append(clip_validation)
        total_hid = total_hid + clip_hid
        clip_train = []
        clip_validation = []
    if rna:
        training_net.append(rna_net)
        training.append(rna_train)
        validation.append(rna_validation)
        total_hid = total_hid + rna_hid
        rna_train = []
        rna_validation = []
    if motif:
        training_net.append(motif_net)
        training.append(motif_train)
        validation.append(motif_validation)
        total_hid = total_hid + motif_hid
        motif_train = []
        motif_validation = []
    if seq:
        training_net.append(seq_net)
        training.append(seq_train)
        validation.append(seq_validation)
        total_hid = total_hid + seq_hid
        seq_train = []
        seq_validation = []        
        
    model.add(Merge(training_net, mode='concat'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    
    #checkpointer = ModelCheckpoint(filepath="models/bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    print 'model training'
    model.fit(training, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])
    
    joblib.dump(rg_scaler, os.path.join(model_dir,'model.pkl'))
    
    return model

def test_ideep(data_dir, model_dir, outfile = 'prediction.txt', rg=True, clip=True, rna=True, motif=False, seq = True):
    test_data = load_data(data_dir, rg=rg, clip=clip, rna=rna, motif=motif, seq = seq, test = True)
    
    #true_y = test_data["Y"].copy()
    
    print 'predicting'
    testing = []
    if rg:
        rg_scaler = joblib.load( os.path.join(model_dir,'rg_scaler.pkl'))
        rg_test, rg_scaler = preprocess_data(test_data["X_RG"], scaler=rg_scaler)
        testing.append(rg_test)
    if clip:
        clip_scaler = joblib.load( os.path.join(model_dir,'clip_scaler.pkl'))
        clip_test, clip_scaler = preprocess_data(test_data["X_CLIP"], scaler=clip_scaler)
        testing.append(clip_test)
    if rna:
        rna_scaler = joblib.load( os.path.join(model_dir,'rna_scaler.pkl'))
        rna_test, rna_scaler = preprocess_data(test_data["X_RNA"], scaler=rna_scaler, stand = True)
        testing.append(rna_test)
    if motif:
        motif_scaler = joblib.load( os.path.join(model_dir,'motif_scaler.pkl'))
        motif_test, motif_scaler = preprocess_data(test_data["motif"], scaler=motif_scaler, stand = True)
        testing.append(motif_test)
    if seq:
        seq_test = test_data["seq"]
        testing.append(seq_test)
    
    model = joblib.load( os.path.join(model_dir,'model.pkl'))       
    predictions = model.predict_proba(testing)
    #pdb.set_trace()
    #auc = roc_auc_score(true_y, predictions[:, 1])
    #print "Test AUC: ", auc    
    #fw.write(str(auc) + '\n')
    #mylabel = "\t".join(map(str, true_y))
    fw = open(outfile, 'w')
    myprob = "\n".join(map(str, predictions[:, 1]))
    #fw.write(mylabel + '\n')
    fw.write(myprob)
    fw.close()


def run_ideep(args):
    data_dir = parser.data_dir
    out_file = parser.out_file
    train = parser.train
    model_dir = parser.model_dir
    predict = parser.predict
    seq = parser.seq
    region_type = parser.region_type
    cobinding = parser.cobinding
    structure = parser.structure
    motif = parser.motif
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    
    if train:
        print 'model training'
        train_ideep(data_dir, model_dir, rg=cobinding, clip=cobinding, rna=structure, motif = motif, seq = seq, batch_size= batch_size, n_epochs = n_epochs)
    else:
        print 'model prediction'
        test_ideep(data_dir, model_dir, outfile = outfile, rg=cobinding, clip=cobinding, rna=structure, motif = motif, seq = seq)
    
    

def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, metavar='<data_directory>', help='Under this directory, you should have feature file: sequences.fa.gz, \
    matrix_Response.tab.gz, matrix_RegionType.tab.gz, matrix_RNAfold.tab.gzmatrix_Cobinding.tab.gz, motif_fea.gz, and label file matrix_Response.tab.gz with 0 and 1 ')
    parser.add_argument('--train', type=bool, default=True, help='use this option for training model')
    parser.add_argument('--model_dir', type=str, default='models', help='The directory to save the trained models for future prediction')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of testing data')
    parser.add_argument('--seq', type=bool, default=True, help='The sequences feature for Convolutional neural network')
    parser.add_argument('--region_type', type=bool, default=True, help='The modularity of region type (types (exon, intron, 5UTR, 3UTR, CDS)')
    parser.add_argument('--cobinding', type=bool, default=True, help='The modularity of cobinding')
    parser.add_argument('--structure', type=bool, default=True, help='The modularity of structure that is probability of RNA secondary structure')
    parser.add_argument('--motif', type=bool, default=False, help='The modularity of motif scores')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--n_epochs', type=int, default=20, help='The number of training epochs (default value: 20)')
    args = parser.parse_args()
    return args

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_ideep(args)
    #run_predict()

