'''
Created on 29/08/2018

@author: wblgers
'''
from __future__ import print_function
import warnings
import os
'''from scikits.talkbox.features import mfcc
from scipy.io import wavfile'''
from hmmlearn import hmm
import numpy as np
import librosa
import math

warnings.filterwarnings('ignore')
'''def extract_mfcc(full_audio_path):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave, nwin=int(sample_rate * 0.03), fs=sample_rate, nceps=12)[0]
    return mfcc_features'''

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1))
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix

'''def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, fs, hop_length=128, n_fft=1024)
    return mfcc'''

def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[1]
        '''feature = extract_mfcc(dir+fileName)'''
        feature = get_mfcc(dir + fileName)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    '''startprobPrior = np.array([0.7, 0.2, 0.1, 0.0, 0.0, 0.0]),
    transmatPrior = np.array([
        [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, ],
        [0.1, 0.1, 0.5, 0.1, 0.1, 0.1, ],
        [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, ],
        [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, ],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, ],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, ],
    ]),'''

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models

def main():
    trainDir = './train_audio/'
    trainDataSet = buildDataSet(trainDir)
    print("Finish prepare the training data")

    hmmModels = train_GMMHMM(trainDataSet)
    print("Finish training of the GMM_HMM models for digits 0-9")

    testDir = './test_audio/'
    testDataSet = buildDataSet(testDir)

    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")


if __name__ == '__main__':
    main()