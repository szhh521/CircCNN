# CircCNN
Python code for Predicting the Back-Splicing Sites for Circular RNA Formation by Cross-Validation and Convolutional Neural Networks

# Code Describe

In this study, we propose CircCNN using convolution neural network and batch normalization for predicting pre-mRNA back-splicing sites. Experimental results on three datasets show that CircCNN outperforms other baseline models. Moreover, PWM features extract by CircCNN are converted as motifs. Further analysis reveals that some of motifs found by CircCNN match known motifs involved in gene expression regulation, the distribution of motif and special short sequence is important for pre-mRNA back-splicing. In general, the findings in this study provide a new direction for exploring pre-mRNA back-splicing for CircRNA formation.

For details of this work, users can refer to our paper "Predicting the Back-Splicing Sites for Circular RNA Formation by Cross-Validation and Convolutional Neural Networks" (Zhen Shen, Lin Yuan, et al. 2022).

# Software Requests
Python 3.6

Keras 2.1.5

Tensorflow-GPU 1.8.0

meme 5.1.1

weblogo 3.7.8

# Model Train
python CircCNN.py './data/human' './data/human/circcnn_result.txt' 1024 7 10

Model parameter description

#sys.argv[1] datapath: model load experiment data from this path

#sys.argv[2] resultpath: store model prediction result

#sys.argv[3] batch_size: CircCNN batch_size

#sys.argv[4] cv_fold_num: cross-validiation fold number

#sys.argv[5] model_run_num: the number of model re-run


# Get motif
python CircCNN_get_motif.py './motif/human' './motif/human/motif_result' 7

Model parameter description

#sys.argv[1] datapath: model load test data and motif_model from this path

#sys.argv[2] resultpath: store motif_related file

#sys.argv[3] cv_fold_num: cross-validiation fold number


# Acknowledgments
J Wang, L Wang. Deep learning of the back-splicing code for circular RNA formation, Bioinformatics 2019;35:5235-5242.
@szhh521
