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

    pdb.set_trace()
    print('Created data splits!')
    # Create lang1 and lang2 vocabs from train+val
    lang1vocab = createVocab(lang1Splits['train']+lang1Splits['val'], params)
    lang2vocab = createVocab(lang2Splits['train']+lang2Splits['val'], params)
    
    # UNK the data
    for split in ['train', 'val', 'test']:
        insertUNK(lang1Splits[split], lang1vocab)
        insertUNK(lang2Splits[split], lang2vocab)
    
    lang1vocab.append('UNK')
    lang2vocab.append('UNK')
    print('Created vocabularies and UNKed data!') 
    # Create index and inverted index
    lang1invidx = {}
    lang2invidx = {}
    lang1idx = {}
    lang2idx = {}

    for i in range(len(lang1vocab)):
        lang1invidx[lang1vocab[i]] = i+1 #+1 ensures 1-indexing needed for lua
        lang1idx[i+1] = lang1vocab[i]
    for i in range(len(lang2vocab)):
        lang2invidx[lang2vocab[i]] = i+1 #+1 ensures 1-indexing needed for lua
        lang2idx[i+1] = lang2vocab[i]

    encodedDataLang1 = {}
    encodedDataLang2 = {}
    
    for split in ['train', 'val', 'test']:
        encodedDataLang1[split] = encodeDataset(lang1Splits[split], lang1invidx, maxLengthLang1)
        encodedDataLang2[split] = encodeDataset(lang2Splits[split], lang2invidx, maxLengthLang2)
        
    with h5py.File(params['save_h5'], 'w') as hf:
        for split in ['train', 'val', 'test']:
            hf.create_dataset('lang1/%s'%(split), data=encodedDataLang1[split])
            hf.create_dataset('lang2/%s'%(split), data=encodedDataLang2[split])
    
    print('Encoded data and saved h5 file to %s'%(params['save_h5']))

    jsonData = {}
    jsonData["ix_to_word_lang1"] = lang1idx
    jsonData["ix_to_word_lang2"] = lang2idx
    jsonData["word_to_ix_lang1"] = lang1invidx
    jsonData["word_to_ix_lang2"] = lang2invidx
    jsonData["num_train"] = len(encodedDataLang1['train'])
    jsonData["num_val"] = len(encodedDataLang1["val"])
    jsonData["num_test"] = len(encodedDataLang1["test"])
   
    json.dump(jsonData, open(params['save_json'], 'w'))
    print('Saved auxilliary data to %s'%(params['save_json']))

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

