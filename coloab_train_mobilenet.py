'''
# !curl -O 'http://www.openslr.org/resources/17/musan.tar.gz'
## cp musan.tar.gz /content/drive/My\ Drive/dataset/musan 
# !tar xzf musan.tar.gz'
# !rm musan.tar.gz


!git clone https://github.com/geraldchao/musan_investigation_cnn_rnn
!pip install -r requirements.txt


import sys
sys.path.append('musan_investigation_cnn_rnn')



from google.colab import drive
drive.mount('/content/drive')
drive_path='/content/drive/My\ Drive/dataset'


# process dataset to raw
from musan_investigation_cnn_rnn.create_database_h5py_raw import *
# process raw to derived
from musan_investigation_cnn_rnn.create_database_h5py_derived import *
!cp musan_data_derived.h5 ${drive_path}

# !cp ${drive_path}/musan_data_derived.h5 .

from musan_investigation_cnn_rnn import my_models

root_path = r'%s/musan_data_derived.h5' % '.'

#drive_path='/datasets/audio/musan'
#root_path = r'%s/musan_data_derived.h5' % drive_path

output_path='%s/musan_out' % drive_path

from musan_investigation_cnn_rnn.coloab_train_mobilenet import *

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_training_data( root_path )

# start training
model, fhist = do_train( X_train, Y_train, X_val, Y_val, output_path  )

run_eval(model, fhist, X_val, Y_val, X_test, Y_test, output_path )
'''

import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from pathlib import Path
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
from custom_layers import *
from keras.callbacks import *
from sklearn.utils import shuffle
from multiprocessing.pool import ThreadPool
from math import ceil
import sys, os
import h5py
from h5py import Dataset

from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from plot_cm import plot_cm
from my_models import get_mobile_net as get_model
import inspect

codes=inspect.getsource(inspect.getmodule(inspect.currentframe()))

np.random.seed(7)

dtime = datetime.now().strftime('-%B-%d-%H-%H-%S')
fname = Path(sys.argv[0]).stem

train_split = .65
test_split = .75
modelname = 'MobileNetv1'
num_feat = 64
seg_len = 200
feat = 'mfb'
dtype = np.float16
bsize = 128
num_epochs=10


ecatg = dict((c,i) for (i,c) in enumerate(['noise', 'music', 'speech']))


def load_training_data( root_path ):
    with h5py.File(root_path, mode='r') as db:
        fdict = dict((c, []) for c in ecatg)
        
        for category in db.keys():
            if category not in ecatg: continue
            for item in db[category].keys():
                file = '%s/%s' % ( category, item )
                fdict[category].append(file)

        train, val, test = {}, {}, {}
        for k in fdict:
            np.random.shuffle(fdict[k])
            ut = int(len(fdict[k])*train_split)
            uv = int(len(fdict[k])*test_split)
            train[k], val[k], test[k] = fdict[k][:ut], fdict[k][ut:uv],\
                                    fdict[k][uv:]
        
        def frm_proc(frms):
            #frms = (frms-frms.mean(axis=(1,2), keepdims=True))/(np.amax(np.abs(frms), axis=(1,2), keepdims=True)+1e-2)#/frms.std(axis=(1,2), keepdims=True)
            return frms.astype(dtype)#
        
        def frm_gen(filtered):
            if len(filtered)<seg_len:
                filtered = np.pad(filtered, pad_width=((seg_len-len(filtered), 0), (0,0)), 
                                  mode='wrap')
            seg_points1 = np.arange(seg_len, len(filtered), seg_len)
            seg_points2 = np.arange(seg_len//2, len(filtered), seg_len)

            frms = np.stack(np.split(filtered, seg_points1)[:-1]+
                                    np.split(filtered, seg_points2)[1:-1]+
                                    [filtered[-seg_len:]])
            return frm_proc(frms)
        
        class data_gen:
            def __init__(self, dic):
                self.dic = dic
                self.labels = []
                
            def yield_dat(self):
                for k, files in self.dic.items():
                    ln = len(files)
                    print('Concatenating {} {} files...'.format(ln, k.upper()))
                    for i, fl in enumerate(files):
                        if not i % 50:
                            print('Read', str(i+1), 'out of', str(ln), 'files...')
                            
                        # print('fl: ', fl, type(db[fl]))
                        dat = frm_gen(db[fl][feat][:])
                        lbl = len(dat)*[float(ecatg[k])]
                        self.labels.append(lbl)
                        yield dat

        print('\nConcatenating train data...')
        dg = data_gen(train)
        if 'X_train' in locals():
            del X_train
        X_train = np.vstack(dg.yield_dat())
        Y_train = np.hstack(dg.labels)
        
        print('\nConcatenating validation data...')
        dg = data_gen(val)
        if 'X_val' in locals():
            del X_val
        X_val = np.vstack(dg.yield_dat())
        Y_val = np.hstack(dg.labels)
        
        print('\nConcatenating test data...')    
        dg = data_gen(test)
        if 'X_test' in locals():
            del X_test
        X_test = np.vstack(dg.yield_dat())
        Y_test = np.hstack(dg.labels)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def do_train( X_train, Y_train, X_val, Y_val, output_path ):
    K.clear_session()

    os.makedirs( output_path + '/models', exist_ok=True )
    modelf = output_path + '/models/'+modelname+str(seg_len)+dtime+'.h5'
    modelff = output_path + '/models/'+modelname+str(seg_len)+dtime+'_final.h5'
    

    in_shape = X_train.shape[1:]
    model = get_model(in_shape, name=modelname)
    print(model.summary())

    lr0=5e-4
    opt = Adam(lr=lr0)
    model.compile(opt, 'sparse_categorical_crossentropy', ['acc'])

    lrs = LearningRateScheduler(lambda ep: K.get_value(opt.lr)\
                                if ep < 5 else K.get_value(opt.lr)*.6, 
                                verbose=1)
    mchk = ModelCheckpoint(modelf, save_best_only='True', period=1,
                           verbose=1)

    fhist = model.fit(X_train, Y_train, batch_size=bsize, epochs=num_epochs, 
                      validation_data=[X_val, Y_val],
                      callbacks=[mchk, lrs])

    model.save(modelff)
    return model, fhist

def run_eval(model, fhist, X_val, Y_val, X_test, Y_test, output_path):
    os.makedirs( output_path + '/results', exist_ok=True )
    resultf = output_path + '/results/'+modelname+str(seg_len)+dtime+'.npz'
    
    Yp_val = model.predict(X_val, verbose=1, batch_size=256)
    Yp_test = model.predict(X_test, verbose=1, batch_size=256)
    np.savez(resultf, Y_val=Y_val, Yp_val=Yp_val, 
             Y_test=Y_test, Yp_test=Yp_test, 
             fhist=fhist.history, ecatg=ecatg, codes = codes )
    # train=train, val=val, test=test)

    ll_val = log_loss(Y_val, Yp_val)
    ll_test = log_loss(Y_test, Yp_test)

    acc_val = accuracy_score(Y_val, Yp_val.argmax(-1))
    acc_test = accuracy_score(Y_test, Yp_test.argmax(-1))

    cm_val = confusion_matrix(Y_val, Yp_val.argmax(-1))
    cm_test = confusion_matrix(Y_test, Yp_test.argmax(-1))

    plot_cm(cm_val, list(ecatg.keys()), True, 'Confusion Matrix - Validation')
    plot_cm(cm_test, list(ecatg.keys()), True, 'Confusion Matrix - Test')

    sns = np.vectorize(lambda x: 1 if x==ecatg['speech'] else 0)
    acc_sns_val = accuracy_score(sns(Y_val), sns(Yp_val.argmax(-1)))
    acc_sns_test = accuracy_score(sns(Y_test), sns(Yp_test.argmax(-1)))

    print('Test Accuracy = {:0.4}%'.format(acc_test*100))











