import os
import sys
from keras.models import Sequential,Model, load_model
import keras.backend as K
from seq_logo import *
from seq_motifs import *
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

key_seq = {"A":[1.0,0.0,0.0,0.0],"C":[0.0,1.0,0.0,0.0],"G":[0.0,0.0,1.0,0.0],"U":[0.0,0.0,0.0,1.0]}

def get_motif_fig_new(filter_weights, filter_outs, out_dir, seqs, sample_i,memefile):
    print ('plot motif fig', out_dir)
    if sample_i:
        print ('sampling')
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)
        filter_outs = filter_outs[sample_i]
    num_filters = filter_weights.shape[0]
    filter_size = 10 # filter_weights.shape[2]

    filters_ic = []
    meme_out = meme_intro(out_dir + '/filters_meme.txt',seqs)


    for f in range(num_filters):
        print ('Filter %d' % f)

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (out_dir,f))

        # write possum motif file
        filter_possum(filter_weights[f,:,:], 'filter%d'%f, '%s/filter%d_possum.txt'%(out_dir,f), False)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


    #################################################################
    # annotate filters
    #################################################################
    # run tomtom #-evalue 0.01
    subprocess.call('/home/szhen/meme/bin/tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (out_dir, out_dir, memefile), shell=True)

    #tsv to txt
    tomtom_txt = open(out_dir + '/tomtom/tomtom.txt','w')
    for ltom in open(out_dir + '/tomtom/tomtom.tsv'):
        tmptom = ltom[:-1].split('\t')
        if tmptom[0] != '' and tmptom[0][0] != '#':
            print(' '.join(tmptom),file=tomtom_txt)
    tomtom_txt.close()

    # read in annotations
    filter_names = name_filters(num_filters, '%s/tomtom/tomtom.txt'%out_dir, memefile)


    #################################################################
    # print a table of information
    #################################################################
    table_out = open('%s/table.txt'%out_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print('%3s  %19s  %10s  %5s  %6s  %6s' % header_cols,file=table_out)

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[f,:,:])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean, fstd = plot_score_density(np.ravel(filter_outs[:,:, f]), '%s/filter%d_dens.pdf' % (out_dir,f))

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print('%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols,file=table_out)

    table_out.close()

    if True:
        new_outs = []
        for val in filter_outs:
            new_outs.append(val.T)
        filter_outs = np.array(new_outs)
        print(filter_outs.shape)
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%out_dir)


def encod_data(data):
    data_out = []
    ind1 = []
    ind2 = []
    for line in data:
        temp = []
        temp.extend(np.array(key_seq.get(k)).astype(np.float) for k in line)
        data_out.append(temp)
        ind1.append(line[:100])
        ind2.append(line[100:])
    data_out = np.array(data_out)
    return ind1,ind2,data_out

fpath = sys.argv[1]
dpath = sys.argv[2]
cv_fold_number = sys.argv[3]

specie = ['human','mouse','fly']
dtype = ['pos','neg']
input = ['input1','input2']

index = 0
for mind in range(cv_fold_number):
    for l1 in specie:
        print('Now in:', l1)
        if l1 == 'human':
            test_data_path = fpath + '/' + l1 + '/test_data_2_' + str(mind)
            test_label_path = fpath + '/' + l1 + '/test_label_2_' + str(mind)
            model_path = fpath + '/' + l1 + '/circrna_motif2_' + str(mind) + '.h5'
            memefile = 'Ray2013_rbp_Homo_sapiens.meme'
        elif l1 == 'mouse':
            test_data_path = fpath + '/' + l1 + '/test_data_6_' + str(mind)
            test_label_path = fpath + '/' + l1 + '/test_label_6_' + str(mind)
            model_path = fpath + '/' + l1 + '/circrna_motif6_' + str(mind) + '.h5'
            memefile = 'Ray2013_rbp_Mus_musculus.meme'
        elif l1 == 'fly':
            test_data_path = fpath + '/' + l1 + '/test_data_7_' + str(mind)
            test_label_path = fpath + '/' + l1 + '/test_label_7_' + str(mind)
            model_path = fpath + '/' + l1 + '/circrna_motif7_' + str(mind) + '.h5'
            memefile = 'Ray2013_rbp_Drosophila_melanogaster.meme'
        for l2 in dtype:
            for l3 in input:
                out_dir = dpath + '/' + l1 + '/model' + str(mind) + '/' + l2 + '/' + l3 + '/motif_cnn'
                if os.path.exists(dpath + '/' + l1 + '/model' + str(mind) + '/' + l2 + '/' + l3):
                    shutil.rmtree(dpath + '/' + l1 + '/model' + str(mind) + '/' + l2 + '/' + l3)
                pos_data = []
                neg_data = []
                for l4, l5 in zip(open(test_data_path), open(test_label_path)):
                    if l5[0] == '1':
                        pos_data.append(l4[:-1])
                    elif l5[0] == '0':
                        neg_data.append(l4[:-1])
                print('Encoding data')
                if l2 == 'pos':
                    ind1, ind2, encode_data = encod_data(pos_data)
                elif l2 == 'neg':
                    ind1, ind2, encode_data = encod_data(neg_data)
                test_1 = []
                test_2 = []
                for lt in range(len(encode_data)):
                    test_1.append(encode_data[lt][:100])
                    test_2.append(encode_data[lt][100:])
                test_1 = np.array(test_1)
                test_2 = np.array(test_2)
                print('Loading model')
                print(model_path)
                model = load_model(model_path)
                if l3 == 'input1':
                    input_seq = ind1
                    con_out_model = Model(inputs=model.inputs, outputs=model.get_layer('conv1').output)
                    conv_out = con_out_model.predict([test_1, test_2])
                    conv_weights = con_out_model.get_layer('conv1').get_weights()
                    conv_weights = np.array(conv_weights)
                elif l3 == 'input2':
                    input_seq = ind2
                    con_out_model = Model(inputs=model.inputs, outputs=model.get_layer('conv3').output)
                    conv_out = con_out_model.predict([test_1, test_2])
                    conv_weights = con_out_model.get_layer('conv3').get_weights()
                    conv_weights = np.array(conv_weights)
                # X = np.array(X)
                print(conv_weights[0].shape)
                print(conv_out.shape)
                filter_weights_old = np.transpose(conv_weights[0][:, 0:, :], (2, 1, 0))
                print(filter_weights_old.shape)
                print(conv_out.shape)
                filter_weights = []
                for x in filter_weights_old:
                    x = x - x.mean(axis=0)
                    filter_weights.append(x)
                filter_weights = np.array(filter_weights)
                sample_i = 0
                if os.path.exists(out_dir):
                    os.remove(out_dir)
                os.makedirs(out_dir)
                if index == 0:
                    get_motif_fig_new(filter_weights, conv_out, out_dir, input_seq, sample_i,memefile)
                print('good')

