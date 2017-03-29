# iDeep

we proposed a deep learning based framework, iDeep, to fuse heterogeneous data for predicting RNA-protein interaction sites. The deep learning framework can
not only learn the hidden feature patterns from individual source of data, but also extracted the shared representation across them. In addition, the convolutional neural network in iDeep can automatically identify binding motifs. To validate our proposed method over other methods,
we perform experiments on large-scale CLIP-seq datasets. The comprehensive results indicated the huge advantage of iDeep, which performs much better than the state-of-the-art methods. 
 <br><br>
Dependency <br>
<a href=https://github.com/fchollet/keras/>keras 1.0.0 library</a> <br>
<a href=https://github.com/scikit-learn/scikit-learn>sklearn</a> <br>

Content <br>
./datasets: the training and testing dataset with extracted features, label and sequence. <br>
./cbust_folder: Cluster-buster tool is used to generate motif features. <br>
./pwms_folder: 102 PWMs from CISBP-RNA (Position Weight Matrix). <br>
./predicted_motifs: detected binding motifs for individual proteins from iDeep. and it also includes the report file ame.html from AME in MEME suite, it reporte the enrichment score for the predicted motifs. <br>
./ideep.py: the python code, it can be ran to reproduce our results. <br>
./make_feature_table.py: it is modified based on <a href=https://github.com/aertslab/primescore>primescore</a>. <br>

<b>Reference</b> <br>
Xiaoyong Pan and Hong-Bin Shen. RNA-protein binding motifs mining with a new hybrid deep learning based cross-domain knowledge integration approach. 201718:136. DOI: 10.1186/s12859-017-1561-8

Contact: Xiaoyong Pan (xypan172436atgmail.com)
