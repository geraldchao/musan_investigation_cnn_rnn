import librosa as rosa
import numpy as np
import python_speech_features as psf
from my_models import get_mobile_net

'''
from demo import *

fn_weights = 'models/MobileNetv1200-May-07-00-00-52_final.h5'
model=load_musan_mobilenet_model( fn_weights, model_name='MobileNetv1' )


fn_wav = '/Users/gerald/tmp/test-video-content/sample_audio.wav'
signal, sound_regions = file__sound_regions( fn_wav, sample_rate, win_len, win_step )

'''


sample_rate = 16000          #Sampling Frequency
win_len = .025      #Window length
win_step = .010     #Time stem between consequetive windows

num_feat = 64
seg_len = 200


categ = ['noise', 'music', 'speech', 'silence']


def file__sound_regions( file_name, sample_rate, win_len, win_step ):
    signal, _ = rosa.core.load( file_name, sr=sample_rate)
    sound_regions = rosa.effects.split(signal, top_db=40, frame_length=int(win_len*sample_rate), 
                            hop_length=int(win_step*sample_rate))
    '''
    idx_signal = np.concatenate(list(np.arange(*v) for v in sp), 0)
    idx_silence = np.setdiff1d(np.arange(len(signal)), idx_signal)
    signal_with_sound = signal[idx_signal]
    '''
    return signal, sound_regions

def signal__mel_features( signal, sample_rate, win_len, win_step ):

    fs = sample_rate
    
    frms = psf.sigproc.framesig(signal, .2*fs, .01*fs)
    frms = frms/frms.std(axis=-1, keepdims=True)
    sig = psf.sigproc.deframesig(frms, siglen=len(signal), 
                                  frame_len=.2*fs, frame_step= .01*fs)
    
    # mfcc = psf.mfcc(sig, samplerate=fs, numcep=20, nfilt=32, winlen=win_len,
    # winstep=win_step).astype(np.float32)
    mfb = psf.logfbank(sig, samplerate=fs, nfilt=64, winlen= win_len,
                            winstep=win_step).astype(np.float16)

    # return (mfcc, mfb)
    return mfb


def mfb_feature__model_inputs( mfb_feature, seg_len = 200 ):
    
    filtered = mfb_feature
    if len(filtered)<seg_len:
        filtered = np.pad(filtered, pad_width=((seg_len-len(filtered), 0), (0,0)), 
                          mode='wrap')
    seg_points1 = np.arange(seg_len, len(filtered), seg_len)
    seg_points2 = np.arange(seg_len//2, len(filtered), seg_len)

    frames = np.stack(np.split(filtered, seg_points1)[:-1]+
                      np.split(filtered, seg_points2)[1:-1]+
                      [filtered[-seg_len:]])
    return frames.astype( np.float16 )

    
    
def load_musan_mobilenet_model( fn_weights, model_name='MobileNetv1' ):
    in_shape = (seg_len, num_feat)
    model = get_mobile_net(in_shape, name=model_name, load_weights=False)
    model.load_weights( fn_weights )
    return model



def wav_file__predictions( model, fn_wav ):
    signal, sound_regions = file__sound_regions( fn_wav, sample_rate, win_len, win_step )

    model_inputs = []
    input_idx__region_idx = []
    for ridx, region in enumerate( sound_regions ):
        signal_region = signal[ region[0]:region[1] ]
        mfb_feature = signal__mel_features( signal_region, sample_rate, win_len, win_step ) # ex: (2240,) -> (13, 64)
        model_input = mfb_feature__model_inputs( mfb_feature, seg_len ) # ex: (1, 200, 64)
        input_idx__region_idx.extend( [ridx] * model_input.shape[0] ) # can be multiple batches
        model_inputs.append( model_input )

    model_inputs = np.vstack(model_inputs) # ex: (988, 200, 64)
    predictions = model.predict(model_inputs, batch_size=256) # ex: (988, 3)

    for region_idx, class_probs in zip( input_idx__region_idx, predictions ):
        region = sound_regions[ region_idx ]
        region__class_prediction( region, class_probs )
    
