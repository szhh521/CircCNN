
#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import os
import sys
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Flatten, GRU,Convolution1D,MaxPooling1D,Input,BatchNormalization
from keras.layers import Concatenate,Dot,Merge,Multiply,RepeatVector,AveragePooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 12345
np.random.seed(seed)

key_seq = {"A":[1.0,0.0,0.0,0.0],"C":[0.0,1.0,0.0,0.0],"G":[0.0,0.0,1.0,0.0],"U":[0.0,0.0,0.0,1.0]}

def data(file_path):
    #load train seq data
    print ('Loading seq data...')
    train_data = file_path + '/all_train'
    train_label = file_path + '/all_train_label'
    test_data = file_path + '/all_test'
    test_label = file_path + '/test_label'
    data_train = []
    #train data encoding
    for line in open(train_data):
        line = line[:-1].strip()
        temp = []
        temp.extend(np.array(key_seq.get(k)).astype(np.float) for k in line)
        data_train.append(temp)
    label_train = [int(k[:-1]) for k in open(train_label)]
    #test data encoding
    for line in open(test_data):
        line = line[:-1].strip()
        temp = []
        temp.extend(np.array(key_seq.get(k)).astype(np.float) for k in line)
        data_train.append(temp)
    label_test = [int(k[:-1]) for k in open(test_label)]
    label = label_train + label_test
    data_train = np.array(data_train)
    label_train = np.array(label)
    return data_train,label_train

def main_model():
    print('Building model...')
    ip1 = Input(shape=[100,4])#SA input
    ip2 = Input(shape=[100,4])#SD input
    #SA input data feature learning
    conv1 = Convolution1D(filters=256, kernel_size=12, strides=1,activation='relu',padding='valid',name='conv1')(ip1)
    d1 = Dropout(0)(conv1)
    conv2 = Convolution1D(filters=128, kernel_size=30, strides=2,activation='relu',padding='same',name='conv2')(d1)
    dro1 = Dropout(0.7)(conv2)
    mp1 = MaxPooling1D(5,5)(dro1)
    d2 = Dropout(0.7)(mp1)
    l_fla1=Flatten()(d2)
    #SD input data feature learning
    conv3 = Convolution1D(filters=256, kernel_size=12, strides=1,activation='relu',padding='valid',name='conv3')(ip2)
    d3 = Dropout(0)(conv3)
    conv4 = Convolution1D(filters=128, kernel_size=30, strides=2,activation='relu',padding='same',name='conv4')(d3)
    dro2 = Dropout(0.7)(conv4)
    mp2 = MaxPooling1D(5,5)(dro2)
    d4 = Dropout(0.7)(mp2)
    l_fla2 = Flatten()(d4)
    convs = []
    convs.append(l_fla1)
    convs.append(l_fla2)
    l_concat = Concatenate(axis=1,name='cvout')(convs)#
    bn1 = BatchNormalization()(l_concat)
    l_sigmoid = Dense(1, activation='sigmoid')(bn1)
    model = Model(inputs=[ip1,ip2],outputs=l_sigmoid)
    return model

def model_eval(file_path,history_callback,model,model_path,test_1,test_2,label_test,batchsize,cvind,index):
    print ('Testing model...')
    model.load_weights(model_path)
    tresults = model.evaluate([test_1,test_2], label_test,batch_size=batchsize)
    print (tresults)
    y_predn = model.predict([test_1,test_2], batch_size=batchsize, verbose=1)
    y_new = label_test

    print ('Calculating metrics...')
    t_auc = metrics.roc_auc_score(y_new, y_predn)
    t_acc = metrics.accuracy_score(y_new, [1 if x > 0.5 else 0 for x in y_predn])
    #t_prc = metrics.average_precision_score(y_new, y_predn)
    precision, recall,_ = metrics.precision_recall_curve(y_new,y_predn)
    t_prc = metrics.auc(recall,precision)
    t_mcc = metrics.matthews_corrcoef(y_new, [1 if x > 0.5 else 0 for x in y_predn])
    t_f1 = metrics.f1_score(y_new, [1 if x > 0.5 else 0 for x in y_predn])
    pre = metrics.precision_score(y_new, [1 if x > 0.5 else 0 for x in y_predn])
    rec = metrics.recall_score(y_new, [1 if x > 0.5 else 0 for x in y_predn])
    tn,fp,fn,tp = metrics.confusion_matrix(y_new, [1 if x > 0.5 else 0 for x in y_predn]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    #print (t_auc, t_prc,t_acc, t_f1, t_mcc,  pre, recall)

    print('save results to local file')
	#save training history data
    myhist = history_callback.history
    all_hist = np.asarray([myhist["loss"], myhist["acc"], myhist["val_loss"], myhist["val_acc"]]).transpose()
    hisname = 'training_history_' + str(index) + '_' + str(cvind) + '.txt'
    if os.path.exists(hisname):
        os.remove(hisname)
    np.savetxt(os.path.join(file_path, hisname), all_hist, delimiter="\t",
               header='loss\tacc\tval_loss\tval_acc')
    return y_predn, t_auc, t_prc, t_acc, t_f1, t_mcc,  pre, rec, sens,spec

def result_process(data):
    result = []
    for l1 in data:
        tmp = []
        for l2 in l1:
            if l2 > 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
        result.append(tmp)
    nr = np.average(data,axis=0)
    #final_re = [1 if x > 0.5 else 0 for x in nr]
    return nr



def model_cross_valid(fpath,optim,batchs,index,cv_fold_num):
    optim = optim
    batchsize = batchs
    file_path = fpath
    data_train, label_train = data(file_path)
    cvind = 0
	#use StratifiedKFold to implement 7fold cross validation
    kfold = StratifiedKFold(n_splits=cv_fold_num,shuffle=True,random_state=seed)
    all_result = [] # save predict data label
    all_metrics = [] # save model metrics
    for train,test in kfold.split(data_train,label_train):
        cv_train = data_train[train]
        cv_train_label = label_train[train]
        cv_test = data_train[test]
        cv_test_label = label_train[test]
        train_1 = []
        train_2 = []
        test_1 = []
        test_2 = []
        for lt in range(len(cv_train)):
            train_1.append(cv_train[lt][:100])
            train_2.append(cv_train[lt][100:])
        train_1 = np.array(train_1)
        train_2 = np.array(train_2)
        for lt in range(len(cv_test)):
            test_1.append(cv_test[lt][:100])
            test_2.append(cv_test[lt][100:])
        test_1 = np.array(test_1)
        test_2 = np.array(test_2)

		model = main_model()
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
        print(model.summary())
        model_path = file_path + '/circrna_best_' + str(index) + '_' + str(cvind) + '.hdf5'
        motif_model = file_path + '/circrna_motif' + str(index) + '_' + str(cvind) + '.h5'


        checkpointer = ModelCheckpoint(filepath=model_path, verbose=1,
                                       save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        print('Training model...')
        history_callback = model.fit([train_1, train_2], y=cv_train_label, nb_epoch=100, batch_size=batchsize,
                                     callbacks=[checkpointer, earlystopper], validation_split=0.1, shuffle=True,
                                     verbose=1)
        print('save motif model')
        if os.path.exists(motif_model):
            os.remove(motif_model)
        model.save(motif_model)
        print('model evaluate')
        tmp = []
        pred_result, auc, prc, acc, f1, mcc, precision, recall,sens,spec = model_eval(file_path,history_callback,model,model_path,test_1,test_2,cv_test_label,batchsize,cvind,index)
        all_result.append(pred_result)
        #'\t'.join([auc, prc, acc, f1, mcc, precision, recall])
        #all_metrics.append('\t'.join([str(auc), str(prc), str(acc), str(f1), str(mcc), str(precision), str(recall), str(sens),str(spec)]))
        all_metrics.append([auc, prc, acc, f1, mcc, precision, recall, sens, spec])
		#save train data,train label, test data,test label for different fold
        if os.path.exists(file_path + '/train_data_' + str(index) + '_' + str(cvind)):
            os.remove(file_path + '/train_data_' + str(index) + '_' + str(cvind))
        if os.path.exists(file_path + '/train_label_' + str(index) + '_' + str(cvind)):
            os.remove(file_path + '/train_label_' + str(index) + '_' + str(cvind))
        if os.path.exists(file_path + '/test_data_' + str(index) + '_' + str(cvind)):
            os.remove(file_path + '/test_data_' + str(index) + '_' + str(cvind))
        if os.path.exists(file_path + '/test_label_' + str(index) + '_' + str(cvind)):
            os.remove(file_path + '/test_label_' + str(index) + '_' + str(cvind))
        if os.path.exists(file_path + '/pred_label_' + str(index) + '_' + str(cvind)):
            os.remove(file_path + '/pred_label_' + str(index) + '_' + str(cvind))
        cur_train_data_out = open(file_path + '/train_data_' + str(index) + '_' + str(cvind),'w')
        cur_train_label_out = open(file_path + '/train_label_' + str(index) + '_' + str(cvind),'w')
        cur_test_data_out = open(file_path + '/test_data_' + str(index) + '_' + str(cvind),'w')
        cur_test_label_out = open(file_path + '/test_label_' + str(index) + '_' + str(cvind),'w')
        #key_seq = {"A":[1.0,0.0,0.0,0.0],"C":[0.0,0.0,0.0,1.0],"G":[0.0,0.0,1.0,0.0],"T":[0.0,1.0,0.0,0.0]}
        for ltrd,ltrl in zip(cv_train,cv_train_label):
            temp = ''
            for kkk in ltrd:
                if kkk[0] == 1.0:
                    temp = temp + 'A'
                elif kkk[1] == 1.0:
                    temp = temp + 'C'
                elif kkk[2] == 1.0:
                    temp = temp + 'G'
                elif kkk[3] == 1.0:
                    temp = temp + 'U'
            print(temp,file=cur_train_data_out)
            print(ltrl,file=cur_train_label_out)
        for lted,ltel in zip(cv_test,cv_test_label):
            temp = ''
            for kkk in lted:
                if kkk[0] == 1.0:
                    temp = temp + 'A'
                elif kkk[1] == 1.0:
                    temp = temp + 'C'
                elif kkk[2] == 1.0:
                    temp = temp + 'G'
                elif kkk[3] == 1.0:
                    temp = temp + 'U'
            print(temp,file=cur_test_data_out)
            print(ltel,file=cur_test_label_out)
        cur_train_data_out.close()
        cur_train_label_out.close()
        cur_test_data_out.close()
        cur_test_label_out.close()
		#save predict results
        if os.path.exists(file_path + '/pred_label_' + str(index) + '_' + str(cvind)):
            os.remove(file_path + '/pred_label_' + str(index) + '_' + str(cvind))
        pred_out = open(file_path + '/pred_label_' + str(index) + '_' + str(cvind),'w')
        for lpred in pred_result:
            print(lpred[0],file=pred_out)
        pred_out.close()
        cvind = cvind + 1

    print('done')
    for ll in all_metrics:
        print(ll)
    return all_metrics

datapath = sys.argv[1] # input data path
resultpath = sys.argv[2] # output path
batch_size = sys.argv[3] # model batch size
cv_fold_num = sys.argv[4] # n-fold cross validation we need
model_run_num = sys.argv[5] # model repeat times

all_out = []
for ind in range(model_run_num):
    print('start iter' + str(ind))
    now_result = model_cross_valid(datapath,'rmsprop',batch_size,ind,cv_fold_num)
    all_out.append('###iter### ' + str(ind))
    for l2 in now_result:
        all_out.append('\t'.join([str(i) for i in l2]))
    all_out.append('###iter### ' + str(ind) + ' ### mean std avg predict result')
    r_average = np.average(now_result,axis=0)
    r_std = np.std(now_result,axis=0)
    all_out.append('\t'.join([str(i) for i in r_average]))
    all_out.append('\t'.join([str(j) for j in r_std]))

if os.path.exists(resultpath):
    os.remove(resultpath)
out1 = open(resultpath,'w')
for con in all_out:
    print(con,file=out1)
out1.close()
