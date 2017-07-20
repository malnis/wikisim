from __future__ import division
import os
import nltk
import json

"""
Gets stats on the POS tag data of mentions and non-mentions.
"""

pathStrt = '/users/cs/amaral/wsd-datasets'
dsPath = os.path.join(pathStrt,'wiki-mentions.30000.json')

with open(dsPath, 'r') as dataFile:
    dataLines = []
    skip = 0
    amount = 30000 # do 30000 for full
    i = 0
    for line in dataFile:
        if i >= skip:
            dataLines.append(json.loads(line.decode('utf-8').strip()))
        i += 1
        if i >= skip + amount:
            break
            
mentionB = {}
mentionC = {}
mentionA = {}

nonmentionB = {}
nonmentionC = {}
nonmentionA = {}

mentions = 0
nonmentions = 0

lnum = 0
for line in dataLines:
    lnum += 1
    print 'Line: ' + str(lnum)
    
    pos = nltk.pos_tag(line['text'])
    for i in range(len(line['text'])):
        # before
        if i == 0:
            keyB = 'NONE'
        else:
            keyB = pos[i-1][1]
            
        # current
        keyC = pos[i][1]
        
        # after
        if i == len(line['text']) - 1:
            keyA = 'NONE'
        else:
            keyA = pos[i+1][1]
        
        if i in [mnt[0] for mnt in line['mentions']]: # is mention
            mentions += 1
            # before
            try:
                mentionB[keyB][0] += 1
            except:
                mentionB[keyB] = [1]
            # current
            try:
                mentionC[keyC][0] += 1
            except:
                mentionC[keyC] = [1]
            # after
            try:
                mentionA[keyA][0] += 1
            except:
                mentionA[keyA] = [1]
        else: # is nonmention
            nonmentions += 1
            # before
            try:
                nonmentionB[keyB][0] += 1
            except:
                nonmentionB[keyB] = [1]
            # current
            try:
                nonmentionC[keyC][0] += 1
            except:
                nonmentionC[keyC] = [1]
            # after
            try:
                nonmentionA[keyA][0] += 1
            except:
                nonmentionA[keyA] = [1]
                
# apply portion to each pos tag (mentions)
for key in mentionB.keys():
    mentionB[key].append(mentionB[key][0]/mentions)
for key in mentionC.keys():
    mentionC[key].append(mentionC[key][0]/mentions)
for key in mentionA.keys():
    mentionA[key].append(mentionA[key][0]/mentions)
# apply portion to each pos tag (nonmentions)
for key in nonmentionB.keys():
    nonmentionB[key].append(nonmentionB[key][0]/nonmentions)
for key in nonmentionC.keys():
    nonmentionC[key].append(nonmentionC[key][0]/nonmentions)
for key in nonmentionA.keys():
    nonmentionA[key].append(nonmentionA[key][0]/nonmentions)
            
print mentionB
print
print mentionC
print
print mentionA
print
print nonmentionB
print
print nonmentionC
print
print nonmentionA
print

with open('/users/cs/amaral/wikisim/wikification/pos-data/pos-mention-bef.tsv', 'w') as f:
    for key in mentionB.keys():
        f.write(key + '\t' + str(mentionB[key][0]) + '\t' + str(mentionB[key][1]) + '\n')
        
with open('/users/cs/amaral/wikisim/wikification/pos-data/pos-mention-cur.tsv', 'w') as f:
    for key in mentionC.keys():
        f.write(key + '\t' + str(mentionC[key][0]) + '\t' + str(mentionC[key][1]) + '\n')
        
with open('/users/cs/amaral/wikisim/wikification/pos-data/pos-mention-aft.tsv', 'w') as f:
    for key in mentionA.keys():
        f.write(key + '\t' + str(mentionA[key][0]) + '\t' + str(mentionA[key][1]) + '\n')
        
with open('/users/cs/amaral/wikisim/wikification/pos-data/pos-nonmention-bef.tsv', 'w') as f:
    for key in nonmentionB.keys():
        f.write(key + '\t' + str(nonmentionB[key][0]) + '\t' + str(nonmentionB[key][1]) + '\n')
        
with open('/users/cs/amaral/wikisim/wikification/pos-data/pos-nonmention-cur.tsv', 'w') as f:
    for key in nonmentionC.keys():
        f.write(key + '\t' + str(nonmentionC[key][0]) + '\t' + str(nonmentionC[key][1]) + '\n')
        
with open('/users/cs/amaral/wikisim/wikification/pos-data/pos-nonmention-aft.tsv', 'w') as f:
    for key in nonmentionA.keys():
        f.write(key + '\t' + str(nonmentionA[key][0]) + '\t' + str(nonmentionA[key][1]) + '\n')