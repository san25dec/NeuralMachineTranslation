import codecs
import string
import pdb

# Function to preprocess a sentence and output words in a list
def prepro_sentence_english(sent):
    sent_prepro = sent.encode('utf-8').lower().translate(None, string.punctuation).strip()
    return sent_prepro

def prepro_sentence_hindi(sent):
    sent_prepro = sent.encode('utf-8').lower().translate(None, string.punctuation).strip()
    return sent_prepro

datasetPath = 'TIDESdataset/hindencorp05.plaintext'
preprocessedSavePath = 'TIDESdataset/preprocessed-data.txt'

with codecs.open(datasetPath, encoding='utf-8') as f:
    inputData = f.read().split('\n')

dataset = {}
dataset['english'] = []
dataset['hindi'] = []

''' 
    Dataset format:
    
    <source identifier>  <alignment type>  <alignment quality>  <english segments>  <hindi segments>

'''

# Ignoring data that does not have 1-1 alignment and preprocessing
for i in range(len(inputData)):
    data = inputData[i].split('\t') # 5 tab separated columns in the data
    
    if len(data) == 5:

        alignment = data[1]
        englishSegment = data[3]
        hindiSegment = data[4]
        
        # preprocess and save the data
        if alignment == '1-1':
            englishSegmentPrepro = prepro_sentence_english(englishSegment)
            hindiSegmentPrepro = prepro_sentence_hindi(hindiSegment)
            if len(englishSegmentPrepro.split()) <= 30 and len(hindiSegmentPrepro.split()) <= 30 and len(englishSegmentPrepro.split()) > 5 and len(hindiSegmentPrepro.split()) > 5:
                dataset['english'].append(prepro_sentence_english(englishSegment))
                dataset['hindi'].append(prepro_sentence_hindi(hindiSegment))

print('Filtered data, using only sentences with 1-1 alignment. Also removed sentences longer than 30 in length: %d/%d sentences used'%(len(dataset['english']), len(inputData)))

# formatted saving of data in text format

f = open(preprocessedSavePath, 'w')
for i in range(len(dataset['english'])):
    f.write('%s::%s\n'%(dataset['hindi'][i], dataset['english'][i]))

f.close()







