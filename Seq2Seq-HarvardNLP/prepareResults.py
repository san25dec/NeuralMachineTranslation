import json
import sys

if len(sys.argv) != 5:
    print('Input format: python prepareResults.py <input target file> <input source file> <input result file> <output file>')
    sys.exit()

input_target = sys.argv[1]
input_source = sys.argv[2]
input_result = sys.argv[3]
output_path = sys.argv[4]

targetEntries = []
sourceEntries = []
resultEntries = []

with open(input_target) as infile:
    for line in infile:
        targetEntries.append(line.replace('/n', ''))

with open(input_source) as infile:
    for line in infile:
        sourceEntries.append(line.replace('/n', ''))

with open(input_result) as infile:
    for line in infile:
        resultEntries.append(line.replace('/n', ''))

outputEntry = []
print(len(targetEntries))
print(len(sourceEntries))
print(len(resultEntries))
for i in range(len(targetEntries)):
    entry = {"seqNo": i+1, "prediction": resultEntries[i], "actual": targetEntries[i], "input": sourceEntries[i]}
    outputEntry.append(entry)

json.dump(outputEntry, open(output_path, 'w'))
