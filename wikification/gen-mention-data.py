from __future__ import division
"""
Generate data to be used for entity recognition.
"""

"""
Mention Data of form [pos before, pos on, pos after, mentions prob].
[0:3] are to be One Hot Encoded.
"""


import requests
import json
import os
from wikification import *
from wikipedia import title2id
import copy
import sys
import copy
import nltk
from pycorenlp import StanfordCoreNLP
scnlp = StanfordCoreNLP('http://localhost:9000')

# need this for novelty detection
#from sklearn.svm import OneClassSVM

# for pos on pos parts
#ohe = OneHotEncoder(n_values=[46,46,46])

# convert pos values to numbers
posDict = {
    "$":0,
    "''":1,
    "(":2,
    ")":3,
    ",":4,
    "--":5,
    ".":6,
    ":":7,
    "CC":8,
    "CD":9,
    "DT":10,
    "EX":11,
    "FW":12,
    "IN":13,
    "JJ":14,
    "JJR":15,
    "JJS":16,
    "LS":17,
    "MD":18,
    "NN":19,
    "NNP":20,
    "NNPS":21,
    "NNS":22,
    "PDT":23,
    "POS":24,
    "PRP":25,
    "PRP$":26,
    "RBR":27,
    "RBS":28,
    "RP":29,
    "SYM":30,
    "TO":31,
    "UH":32,
    "VB":33,
    "VBD":34,
    "VBG":35,
    "VBN":36,
    "VBP":37,
    "VBZ":38,
    "WDT":39,
    "WP":40,
    "WP$":41,
    "WRB":42,
    "``":43,
    "None":44,
    "NONE":45,
    "RB":46
}

posBefDict = {
    'IN':0,
    'DT':1,
    'NNP':2,
    'JJ':3,
    ',':4,
    'CC':5,
    'NN':6,
    'VBD':7,
    'CD':8,
    '(':9,
    'TO':10,
    'FAIL':11
}

posCurDict = {
    'NNP':0,
    'NN':1,
    'JJ':2,
    'NNS':3,
    'CD':4,
    'NNPS':5,
    'FAIL':6
}

posAftDict = {
    ',':0,
    '.':1,
    'IN':2,
    'NNP':3,
    'CC':4,
    'NN':5,
    'VBD':6,
    ':':7,
    'VBZ':8,
    'POS':9,
    'NNS':10,
    'TO':11,
    'FAIL':12
}

def normalize(nums):
    """Normalizes a list of nums to its sum + 1"""
    
    numSum = sum(nums) + 1 # get max
    
    # fill with normalized
    normNums = []
    for num in nums:
        normNums.append(num/numSum)
        
    return normNums

pathStrt = '/users/cs/amaral/wsd-datasets'
dsPath = os.path.join(pathStrt,'wiki-mentions.30000.json')

newData = []

# exclude non-mentions to treat as novelty detection
# include non-mentions to treat as classification
nonMentions = True

with open(dsPath, 'r') as dataFile:
    dataLines = []
    skip = 0
    amount = 30000
    i = 0
    for line in dataFile:
        if i >= skip:
            dataLines.append(json.loads(line.decode('utf-8').strip()))
        i += 1
        if i >= skip + amount:
            break
            
errors = 0
        
lnum = 0
for line in dataLines:
    
    oMentions = copy.deepcopy(line['mentions']) # mentions in original form
    oText = " ".join(copy.deepcopy(line['text']))
    #uni = unicode(oText, 'utf-8')
    #print uni
    line['mentions'] = mentionStartsAndEnds(line, True)

    #Get POS tags of all text
    postrs = nltk.pos_tag(copy.deepcopy(line['text']))

    # get stanford core mentions
    try:
        stnfrdMentions0 = scnlp.annotate(oText.encode('utf-8'), properties={
                'annotators': 'entitymentions',
                'outputFormat': 'json'})
    except:
        errors += 1
        print 'Error #' + str(errors) + ' on line #' + str(lnum)
        lnum += 1
        continue
    stnfrdMentions = []
    for sentence in stnfrdMentions0['sentences']:
        for mention in sentence['entitymentions']:
            stnfrdMentions.append(mention['text'])

    for i in range(len(line['text'])):
        
        if nonMentions == False and i not in [item[0] for item in oMentions]:
            continue
        
        newData.append([]) # add new row to mention data at mIdx
             
        """ 
        Append POS tags of before, on, and after mention.
        """
        if i == 0:
            bef = 'NONE'
        else:
            bef = postrs[i-1][1] # pos tag of before
        if bef in posBefDict:
            bef = posBefDict[bef]
        else:
            bef = posBefDict['FAIL']
            
        on = postrs[i][1] # pos tag of mention
        if on in posCurDict:
            on = posCurDict[on]
        else:
            on = posCurDict['FAIL']
        
        if i == len(line['text']) - 1:
            aft = 'NONE'
        else:
            aft = postrs[i+1][1] # pos tag of after
        if aft in posAftDict:
            aft = posAftDict[aft]
        else:
            aft = posAftDict['FAIL']
        
        newData[-1].extend([bef, on, aft])
        
        """
        Append mention probability.
        """
        newData[-1].append(mentionProb(line['text'][i]))
        
        """
        Find whether Stanford NER decides the word to be mention.
        """
        if line['text'][i] in stnfrdMentions:
            stnfrdMentions.remove(line['text'][i])
            newData[-1].append(1)
        else:
            newData[-1].append(0)
            
        """
        Whether starts with capital.
        """
        if line['text'][i][0].isupper():
            newData[-1].append(1)
        else:
            newData[-1].append(0)
            
        """
        Whether there is an exact match in Wikipedia.
        """
        if title2id(line['text'][i]) is not None:
            newData[-1].append(1)
        else:
            newData[-1].append(0)
            
        """
        Whether word contains a space.
        """
        if ' ' in line['text'][i]:
            newData[-1].append(1)
        else:
            newData[-1].append(0)
            
        """
        Whether the word contains only ascii characters.
        """
        try:
            line['text'][i].decode('ascii')
            newData[-1].append(1)
        except:
            newData[-1].append(0)
        
        # put in whether is mention or not only if including nonMentions
        if nonMentions == True:
            if i in [item[0] for item in oMentions]:
                newData[-1].append(1)
            else:
                newData[-1].append(0)
    
        #print newData[-1]
        
    lnum += 1
    print 'Line: ' + str(lnum)
    
# nov for novelty, cls for classification
#with open('/users/cs/amaral/wikisim/wikification/learning-data/er-10000-nov.txt', 'w') as f:
with open('/users/cs/amaral/wikisim/wikification/learning-data/er-30000-cls-v2.txt', 'w') as f:
    for data in newData:
        f.write(str(data)[1:-1] + '\n')