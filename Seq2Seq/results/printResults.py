import json
import sys
import string

filePath = sys.argv[1]
savePath = sys.argv[2]
with open(filePath) as inFile:
    translations = json.load(inFile)

def prepro_sentence_hindi(sent):
    sent_prepro = sent.encode('utf-8').lower().translate(None, string.punctuation).strip()
    return sent_prepro

with open(savePath, 'w') as outFile:
    for i in range(13000, 13200):
        outFile.write(''.join(['-' for j in range(40)])+'\n')
        outFile.write('Hindi: %s\n'%(prepro_sentence_hindi(translations[i]['input'])))
        outFile.write('Translated: %s\n'%(''.join([w if ord(w) < 128 else '' for w in translations[i]['prediction']])))
        outFile.write('Actual: %s\n'%(''.join([w if ord(w) < 128 else '' for w in translations[i]['actual']])))

