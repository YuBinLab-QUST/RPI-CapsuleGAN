# RPI-CapsuleGAN
RPI-CapsuleGAN: predicting RNA-protein interactions through an interpretable generative adversarial capsule network with the convolutional block attention module
###RPI-CapsuleGAN uses the following dependencies:

 * Python 3.6
 * numpy
 * scipy
 * scikit-learn
 * pandas
 * tensorflow 
 * keras

###Guiding principles: 

**The dataset file contains seven datasets, among which RPI488, RPI369, RPI2241, RPI1807, RPI1446, NPInter v3.0, NPInter227.

**Feature extraction：
 * feature-RNA is the implementation of kGap descriptors, pseSSC for RNA.
 * feature-protein is the implementation of GTPC, RPSSM and EDPSSM for protein.
 * Sequence_Struct_RNA_protein is the implementation of sequence information and secondary structure for protein and RNA.


**Feature_selection:
 * ALL_select is the implementation of all feature selection methods used in this work, among which SE, MI, LinearSVC, Chi2, LASSO, TSVD, MDS, OMP,EN.


**Classifier:
 * RPI-CapsuleGAN_model.py is the implementation of our model in this work.
 * classical classifier is the implementation of classical classifierS compared in this work, among which ET, KNN, MLP, NB, RF, SVM.
 * deep learning classifier is the implementation of deep learning classifiers compared in this work, among which Capsule, CNN, DCGAN, DNN, GAN, GRU.



