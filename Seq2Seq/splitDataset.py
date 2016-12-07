import codecs
import string
import argparse
import numpy as np
import random
import h5py
import json
import pdb

def createVocab(langData, params):
    
    vocab = {}
    for i in range(len(langData)):
        for w in langData[i]:
            if w not in vocab:
                vocab[w] = 0
            vocab[w] += 1
    # Sort vocabulary and print top 20 words
    sortedVocab = sorted([(count, w) for w, count in vocab.iteritems()], reverse=True)
    print('--------------------')
    print('--- Top 20 words ---')
    print('--------------------')
    
    for i in range(20):
        print('%.12s : %.5d'%(sortedVocab[i][1], sortedVocab[i][0]))
    
    totalCount = 0
    finalVocab = []

    for i in range(len(sortedVocab)):
        if sortedVocab[i][1] >= params['word_count_thresh'] and totalCount <= params['max_vocab_size']:
            totalCount += 1
            finalVocab.append(sortedVocab[i][1])
    
    print('\nTotal Vocab size: %d'%(len(finalVocab)))
    return finalVocab

def insertUNK(langData, vocab):

    vocabSet = set(vocab)
    for i in range(len(langData)):
        langData[i] = [w if w in vocabSet else 'UNK' for w in langData[i]]

# Encodes the sentence as a sequence of indices from vocab inverted index
def encodeDataset(langData, langInvIdx, maxLength):
    
    encodedData = np.zeros((len(langData), maxLength))
    
    for i in range(len(langData)):
        encodedData[i][0:len(langData[i])] = np.array([langInvIdx[w] for w in langData[i]])

    return encodedData

def main(params):

    # Loading preprocessed data
    # Data format: <language 1 sentence>::<language 2 sentence>\n
    f = open(params['prepro_data'])

    lang1data = []
    lang2data = []
    for line in f:
        buf = line.replace('/n', '').split('::')
        lang1sent = buf[0].split()
        lang2sent = buf[1].split()
        if len(lang1sent) <= params['max_length_filter'] and len(lang2sent) <= params['max_length_filter']:
            lang1data.append(lang1sent)
            lang2data.append(lang2sent)
    print('Loaded data!')

    # Shuffle the data once
    langJoined = list(zip(lang1data, lang2data))
    random.shuffle(langJoined)
    lang1data, lang2data = zip(*langJoined)
   
    maxLengthLang1 = 0
    maxLengthLang2 = 0

    # Find the maximum length of sentences in the data
    for i in range(len(lang1data)):
        maxLengthLang1 = max(maxLengthLang1, len(lang1data[i]))
        maxLengthLang2 = max(maxLengthLang2, len(lang2data[i]))

    # Create data splits (train, val and test)
    lang1Splits = {"train":[], "val":[], "test":[]}
    lang2Splits = {"train":[], "val":[], "test":[]}
    
    for i in range(len(lang1data)):
        if i < params['num_test']:
            split = 'test'
        elif i < params['num_test']+params['num_val']:
            split = 'val'
        else:
            split = 'train'
        lang1Splits[split].append(lang1data[i])
        lang2Splits[split].append(lang2data[i])
  
    # Sort the data based on length of sentences first in each split
    # This is done in order to minimze the effect of NULL tokens in the end
    for split in ['train', 'val', 'test']:
        lang1Lengths = []
        for i in range(len(lang1Splits[split])):
            lang1Lengths.append(len(lang1Splits[split][i]))
        combined = zip(lang1Lengths, lang1Splits[split], lang2Splits[split])
        combined_sorted = sorted(combined, reverse=True)
        lang1SplitTemp = []
        lang2SplitTemp = []
        for i in range(len(lang1Splits[split])):
            lang1SplitTemp.append(combined_sorted[i][1])
            lang2SplitTemp.append(combined_sorted[i][2])
        lang1Splits[split] = lang1SplitTemp
        lang2Splits[split] = lang2SplitTemp
    
    for split in ['train', 'val', 'test']:
        with open('/home/santhosh/Projects/MachineTranslationHarvard/seq2seq-attn/data/src-%s.txt'%(split), 'w') as fopen:
            for s in lang1Splits[split]:
                fopen.write(' '.join(s)+'\n')
        with open('/home/santhosh/Projects/MachineTranslationHarvard/seq2seq-attn/data/targ-%s.txt'%(split), 'w') as fopen:
            for s in lang2Splits[split]:
                fopen.write(' '.join(s)+'\n')
    
if __name__ == "__main__":
    
    random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepro_data', required=True, help='input preprocessed dataset')
    parser.add_argument('--max_length_filter', default=30, type=int, help='filter out all data longer than this threshold (to reduce computational load)')
    parser.add_argument('--max_vocab_size', default=30000, type=int, help='limit the vocabulary size')
    parser.add_argument('--word_count_thresh', default=5, type=int, help='remove infrequent words from the vocabulary and set them as UNK')
    parser.add_argument('--num_test', default=40000, type=int, help='number of test samples')
    parser.add_argument('--num_val', default=21965, type=int, help='number of validation samples')
    parser.add_argument('--save_h5', required=True, help='Save path for encoded data in .h5 format')
    parser.add_argument('--save_json', required=True, help='Save path for auxilliary data in .json format')

    args = parser.parse_args()
    params = vars(args)
    main(params)

