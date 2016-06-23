#!/bin/sh

# for positives
# file with names of the PWMs (correspond to the name of the PWM in the folder ./pwms_folder)
#PATH_TO_FEATURE_ORDER=./example/feature_order.txt 
PATH_TO_FEATURE_ORDER=/home/panxy/eclipse/ideep/ideep/selected_motif
#.fasta file
#PATH_TO_FASTA_FILE=/home/panxy/eclipse/primescore/example/pos.fa
PATH_TO_FASTA_DIR=$1
#echo "uncompress seq"
#zcat $PATH_TO_FASTA_DIR/sequences.fa.gz > $PATH_TO_FASTA_DIR/sequences.fa
PATH_TO_FASTA_FILE=$PATH_TO_FASTA_DIR/sequences.fa
# path to save results
#PATH_TO_SAVE=/home/panxy/eclipse/primescore/results/pos.FT.txt
PATH_TO_SAVE=$PATH_TO_FASTA_DIR/motif_fea.gz
# python script
PATH_TO_SCRIPT=/home/panxy/eclipse/primescore/src/make_feature_table.py
# folder where files with PWMs are
PATH_TO_SINGLEOTNS=./pwms
# path to compiled Cluster-buster tool
PATH_TO_CBUST=/home/panxy/eclipse/primescore/cbust_folder/cbust
# command to run the script
python ${PATH_TO_SCRIPT} -f ${PATH_TO_FASTA_FILE} -o ${PATH_TO_FEATURE_ORDER} -m ${PATH_TO_SINGLEOTNS}/ -s ${PATH_TO_SAVE} -c ${PATH_TO_CBUST}

rm -rf $PATH_TO_FASTA_DIR/sequences.fa
#echo "compress generated fea"
#gzip motif_fea
#mv motif_fea.gz $PATH_TO_FASTA_DIR/motif_fea.gz

# for negatives
# file with names of the PWMs (correspond to the name of the PWM in the folder ./pwms_folder)
#PATH_TO_FEATURE_ORDER=./example/feature_order.txt
#.fasta file
#PATH_TO_FASTA_FILE=./example/neg.fa
# path to save results
#PATH_TO_SAVE=./results/neg.FT.txt
# python script
#PATH_TO_SCRIPT=./src/make_feature_table.py
# folder where files with PWMs are
#PATH_TO_SINGLEOTNS=./pwms
# path to compiled Cluster-buster tool
#PATH_TO_CBUST=./cbust_folder/cbust
# command to run the script
#python ${PATH_TO_SCRIPT} -f ${PATH_TO_FASTA_FILE} -o ${PATH_TO_FEATURE_ORDER} -m ${PATH_TO_SINGLEOTNS}/ -s ${PATH_TO_SAVE} -c ${PATH_TO_CBUST}
