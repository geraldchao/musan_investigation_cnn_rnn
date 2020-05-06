
# This file takes the raw waveform data created by 
# 'create_database_h5py_raw file and produces the final derived
# features that are used by the classifiers

import numpy as np
import python_speech_features as psf
import h5py
from h5py import Dataset
import inspect
from itertools import count
import os
from pathlib import Path

codes=inspect.getsource(inspect.getmodule(inspect.currentframe()))

np.random.seed(7)

fs = 16000          #Sampling Frequency
win_len = .025      #Window length
win_step = .010     #Time stem between consequetive windows

# root_path = r'E:\musan_data_raw.h5'
# target_path = r'e:\musan_data_derived.h5'
# root_path = r'/content/musan/musan_data_raw.h5'
# target_path = r'/content/drive/My\ Drive/dataset/musan_data_derived.h5'
# target_path = r'/content/musan_data_derived.h5'

root_path = r'/datasets/audio/musan/musan_data_raw.h5'
target_path = r'/datasets/audio/musan//musan_data_derived.h5'

if os.path.exists(target_path):
    os.remove(str(target_path))

db = h5py.File(root_path, 'r')
categ = ['noise', 'music', 'speech', 'silence']

# keys = list(db.keys())
# fdict = dict((k, list(kk for kk in keys if kk.split('\\')[0]==k)) for k in categ)


def proc_file(file):
    sig = db[file][:]
    
    frms = psf.sigproc.framesig(sig, .2*fs, .01*fs)
    frms = frms/frms.std(axis=-1, keepdims=True)
    sig = psf.sigproc.deframesig(frms, siglen=len(sig), 
                                  frame_len=.2*fs, frame_step= .01*fs)
    
    mfcc = psf.mfcc(sig, samplerate=fs, numcep=20, nfilt=32, winlen=win_len,
                        winstep=win_step).astype(np.float32)
    mfb = psf.logfbank(sig, samplerate=fs, nfilt=64, winlen= win_len,
                            winstep=win_step).astype(np.float16)

    return (file, mfcc, mfb)


with h5py.File(target_path, mode = 'w-') as fl:
    fl.attrs['codes']=codes
    for key in categ:
        print('\nProcessing', key)

        def visit_file(file_path):
            file = '%s/%s' % (key, file_path)
            print('file: ', file)
            # if not type(db[file]) == Dataset: return
            if db[file].shape[0]>win_len*fs:
                file,mfcc, mfb= proc_file(file)
                grp = fl.create_group(file)
                for nm, vl in zip(('mfcc', 'mfb'), (mfcc, mfb)):
                    grp[nm] = vl

        if key not in db: continue
        group = db[key]
        for item in group.keys():
            visit_file( item )

db.close()
print('\nDone!')
