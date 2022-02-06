# CircCNN
Python code for Predicting the Back-Splicing Sites for Circular RNA Formation by Cross-Validation and Convolutional Neural Networks

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
