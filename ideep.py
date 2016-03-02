import sys
import os
from keras.models import Sequential, model_from_config
from keras.layers.core import Dense, Dropout, Activation, AutoEncoder, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import containers, normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.optimizers import kl_divergence
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
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from scipy import sparse
import pdb
from math import  sqrt
from sklearn.metrics import roc_curve, auc
import theano

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

def load_data(path, kmer=True, rg=True, clip=True, rna=True, go=True, seq= True):
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
    if seq: data["seq"] = read_seq(os.path.join(path, 'sequences.fa.gz'))
                                   
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

def preprocess_data(X, scaler=None, stand = True):
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
    nbfilter = 32
    model = Sequential()
    model.add(Convolution1D(input_dim=4,input_length=107,
                            nb_filter=nbfilter,
                            filter_length=6,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))
    
    model.add(MaxPooling1D(pool_length=3))
    
    #model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(nbfilter, activation='relu'))
    #model.add(Activation('relu'))
    #model.add(PReLU())
    #model.add(BatchNormalization())
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
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_hidden, input_dim=num_hidden, activation='relu'))
    #model.add(Dense(num_hidden, input_shape=(num_hidden,), activation='relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
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
    #checkpointer = ModelCheckpoint(filepath="models/" + protein + "_bestmodel.hdf5", verbose=0, save_best_only=True)
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

def calculate_auc(net, hid, train, test, true_y, train_y, rf = False, validation = None, val_y = None, protein = None):
    #print 'running network' 
    if rf:
        predict, model = run_randomforest_classifier(train, train_y, test)
    else:
        predict, model = run_network(net, hid, train, test, train_y, validation, val_y, protein = protein)
        #
        
        get_feature = theano.function([model.layers[0].input],model.layers[7].get_output(train=False),allow_input_downcast=True)
        #train = get_feature(train)
        test = get_feature(test)
        plt.imshow(test,cmap = cm.Greys_r)
        plt.show()
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

def run_individual_network(protein, kmer=True, rg=True, clip=True, rna=True, go=True, seq = False, fw = None):
    training_data = load_data("../datasets/clip/%s/5000/training_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, seq=seq)
    print 'training', len(training_data)
    go_hid = 512
    kmer_hid = 512
    rg_hid = 128
    clip_hid = 256
    rna_hid=64
    seq_hid = 64
    training_indice, training_label, validation_indice, validation_label = split_training_validation(training_data["Y"])
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
    if seq:
        seq_data = training_data["seq"]
        seq_train = seq_data[training_indice]
        seq_validation = seq_data[validation_indice] 
        seq_net =  get_cnn_network()
        seq_data = []
    
    rf = False
    if not rf:   
        #all_label =  training_data["Y"]   
        y, encoder = preprocess_labels(training_label)
        val_y, encoder = preprocess_labels(validation_label, encoder = encoder)
    else:
        y = training_label
        val_y = validation_label
    
    training_data.clear()
    
    
    test_data = load_data("../datasets/clip/%s/5000/test_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, seq=seq)
    
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
        rna_test, rna_scaler = preprocess_data(test_data["X_RNA"], scaler=rna_scaler)
        rna_auc, rna_predict = calculate_auc(rna_net, rna_hid, rna_train, rna_test, true_y, y, validation = rna_validation, 
                                             val_y = val_y, protein = protein,  rf= rf)
        rna_train  = []
        rna_test = []        
        eg_array.append(rna_predict)
    if seq:
        seq_test = test_data["seq"]
        seq_auc, seq_predict = calculate_auc(seq_net, seq_hid, seq_train, seq_test, true_y, y, validation = seq_validation,
                                              val_y = val_y, protein = protein,  rf= rf)
        seq_train = []
        seq_test = []
        eg_array.append(seq_predict)
        
    test_data.clear()
    if seq:
        print seq_auc
    else:
        eg_array = np.array(eg_array).T
        print eg_array.shape 
        #weight_score = get_meta_predictor(eg_array)
        weight_score = eg_array.mean(axis=1)
        
        weight_auc = roc_auc_score(true_y, weight_score)
        
        print str(kmer_auc) + '\t' +  str(rg_auc) + '\t' +  str(clip_auc) + '\t' +  str(rna_auc) +'\t'  + str(weight_auc)
        fw.write(str(kmer_auc) + '\t' +  str(rg_auc) + '\t' +  str(clip_auc) + '\t' +  str(rna_auc) + '\t' + str(weight_auc) +'\n')
        
        mylabel = "\t".join(map(str, true_y))
        myprob = "\t".join(map(str, weight_score))
        myprob1 = "\t".join(map(str, kmer_predict))
        myprob2 = "\t".join(map(str, rg_predict))
        myprob3 = "\t".join(map(str, clip_predict))
        myprob4 = "\t".join(map(str, rna_predict))
    
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
        fw.write(myprob1 + '\n')
        fw.write(myprob2 + '\n')
        fw.write(myprob3 + '\n')
        fw.write(myprob4 + '\n')
    

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
        
def merge_seperate_network_with_multiple_features(protein, kmer=False, rg=True, clip=True, rna=True, go=False, seq = False, fw = None):
    training_data = load_data("../datasets/clip/%s/5000/training_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, seq=seq)
    print 'training', len(training_data)
    go_hid = 512
    kmer_hid = 512
    rg_hid = 128
    clip_hid = 256
    rna_hid=64
    cnn_hid = 64
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
        rna_data, rna_scaler = preprocess_data(training_data["X_RNA"])
        rna_train = rna_data[training_indice]
        rna_validation = rna_data[validation_indice]        
        rna_net = get_rnn_fea(rna_train, sec_num_hidden = rna_hid, num_hidden = rna_hid*2)
        rna_data = []
        training_data["X_RNA"] = []
    if seq:
        seq_train = training_data["seq"]
        seq_net =  get_cnn_network()
        seq_train = []
        
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
    if seq:
        training_net.append(seq_net)
        training.append(seq_train)
        total_hid = total_hid + cnn_hid
        seq_train = []
        
    model.add(Merge(training_net, mode='concat'))
    
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
    
    test_data = load_data("../datasets/clip/%s/5000/test_sample_0" % protein, kmer=kmer, rg=rg, clip=clip, rna=rna, go=go, seq=seq)
    
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
        rna_test, rna_scaler = preprocess_data(test_data["X_RNA"], scaler=rna_scaler)
        testing.append(rna_test)
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
    
def read_result_file(filename = 'result_file_mix_100'):
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
    methodlabel = ['k-mer', 'region type', 'clip-cobinding', 'structure', 'iDeep']
    
    #xval = [5, 10, 20, 30, 40, 50, 60,70]#, 80, 90]
    width = 0.15
    ind = np.arange(len(menMeans[0]))
    fig, ax = plt.subplots(figsize=(12,12))
    #pdb.set_trace()
    #plt.plot(xval,menMeans)
    rects1 = plt.bar(ind, menMeans[0], width, color='r')
    rects2 = plt.bar(ind +width, menMeans[1], width, color='g')
    rects3 = plt.bar(ind +2*width, menMeans[2], width, color='y')
    rects4 = plt.bar(ind+3*width, menMeans[3], width, color='b')
    rects5 = plt.bar(ind+4*width, menMeans[4], width, color='k')
    #plt.title('stem cell circRNA vs other circRNA')
    ax.set_ylabel('AUC', fontsize=20)
    #plt.xlabel('Number of trees', fontsize=20)
    #ax.set_ylim([0.6, 0.75])
    ax.set_xticks(ind)
    ax.set_xticklabels(xlabel, rotation=90 )
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('k-mer', 'region type', 'clip-cobinding', 'structure', 'iDeep'))
    plt.tight_layout()
    
    plt.show()

def read_individual_auc(filename = 'result_file_all'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            pro = values[0].split('_')[0]
            results[int(pro)] = values[1:-1]    
    
    return results

def read_ideep_auc(filename = 'result_mix_auc'):
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
    ideep_resut = read_ideep_auc(filename='result_mix_auc')
    indi_result = read_individual_auc()
    keys = indi_result.keys()
    keys.sort()
    #pdb.set_trace()
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
    #pdb.set_trace()
    new_results = map(list, zip(*new_results))
    
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
     


def run_predict():
    data_dir = '/home/panxy/eclipse/ideep/datasets/clip'
    fw = open('result_file_seq_cnn', 'w')
    for protein in os.listdir(data_dir):
        print protein
        fw.write(protein + '\t')
        model = merge_seperate_network_with_multiple_features(protein, kmer=True, rg=True, clip=True, rna=True, go=False, seq = False, fw = fw)
    fw.close()
         
if __name__ == "__main__":
    #training_indice, training_label, validation_indice, validation_label = split_training_validation(labels)
    #pdb.set_trace()
    run_predict()
    #run_cnn()
    #run_get_sequence()
    #plot_figure()
    #plot_ideep_indi_comp()
    
