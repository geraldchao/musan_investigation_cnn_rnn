
# This script takes the musan database and saves it as
# raw waveform in hdf5 file. It also separates the silence
# parts of the files and puts it into a separate group

import numpy as np
from pathlib import Path
import h5py
import librosa as rosa
import inspect
from itertools import count
import os

codes=inspect.getsource(inspect.getmodule(inspect.currentframe()))

np.random.seed(7)

fs = 16000          #Sampling Frequency
win_len = .025      #Window length
win_step = .010     #Time stem between consequetive windows

# root_path = Path(r'E:\musan')                #path to the musan database
# target_path = Path(r'e:\musan_data_raw.h5')  #path to the output file
# root_path = Path(r'/content/drive/My\ Drive/dataset/musan')
# target_path = Path(r'/content/drive/My\ Drive/dataset/musan_data_raw.h5')
root_path = Path(r'/datasets/audio/musan')
target_path = Path(r'/datasets/audio/musan/musan_data_raw.h5')
if target_path.exists():
    os.remove(str(target_path))

catg = ['noise', 'music', 'speech']

roots = {}
for c in catg:
    roots[c]=root_path/c

fdict = dict((k, list(roots[k].glob('**/*.wav'))) for k in roots)

# silence = {}
    

def proc_file(file):
    sig, _ = rosa.core.load(str(file), sr=fs)
    sp = rosa.effects.split(sig, top_db=40, frame_length=int(win_len*fs), 
                            hop_length=int(win_step*fs))
    isig = np.concatenate(list(np.arange(*v) for v in sp), 0)
    isil = np.setdiff1d(np.arange(len(sig)), isig)
    sigp = sig[isig]
    sigsilence = sig[isil]
    # silence[file]=sig[isil]
    
    return (file,sigp, sigsilence)

with h5py.File(target_path, mode = 'w') as fl:
    fl.attrs['codes']=codes
    for key in fdict:
        print('\nProcessing', key, 'files: total =', len(fdict[key]))
        for i, processed in zip(count(),map(proc_file, iter(fdict[key]))):                                  
            file_path, sig, sigsilence = processed
            file_path = str(file_path.relative_to(root_path))
            file_key = ':'.join( file_path.split('/')[1:]) # strip out class
            file = '%s/%s' % ( key, file_key ) # to avoid nested files
            fl[file] = sig
            if not (i+1) % 5:
                print(key.upper(), 'File', i+1, 'of', str(len(fdict[key])) + ' ' + file)
            k = 'silence/'+file
            fl[k] = sigsilence
                
    '''
    print('\nProcessing silence... Len=', len(silence))
    for i, file in enumerate(silence):
        k = 'silence/'+str(file.relative_to(root_path))
        fl[k] = silence[file]
        if not (i+1) % 5:
            print(key.upper(), 'File', i+1, 'of', str(len(fdict[key])) + ' ' + file)
    ''' 
        
print('\nDone!')

