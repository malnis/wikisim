
"""
This file/cell is to generate training data for entity linking for a supervised model.
Each row has form: (id, isTrueEntity, popularity, context1, context2, word2vec, coherence, mentionId)
"""

from __future__ import division
import requests
import json
import os
from wikification import *
import copy
import sys
 
def normalize(nums):
    """Normalizes a list of nums to its sum + 1"""
    
    numSum = sum(nums) + 1 # get max
    
    # fill with normalized
    normNums = []
    for num in nums:
        normNums.append(num/numSum)
        
    return normNums

pathStrt = '/users/cs/amaral/wsd-datasets'
dsPath = os.path.join(pathStrt,'wiki-mentions.5000.json')

with open(dsPath, 'r') as dataFile:
    dataLines = []
    i = 0
    for line in dataFile:
        dataLines.append(json.loads(line.decode('utf-8').strip()))
        i += 1
        if i > 5000:
            break
        
cPerM = 20 # candidates per mention

allCands = []

# word2vec loading
try:
    word2vec
except:
    print 'loading word2vec'
    word2vec = gensim_loadmodel('/users/cs/amaral/cgmdir/WikipediaClean5Negative300Skip10.Ehsan/WikipediaClean5Negative300Skip10')

print 'word2vec loaded'
    
f = 0

mNum = 0
# see each line
for line in dataLines:
    
    oMentions = copy.deepcopy(line['mentions']) # mentions in original form
    oText = " ".join(copy.deepcopy(line['text']))
    
    line['mentions'] = mentionStartsAndEnds(line)
    # get what should be all candidates
    candidates = generateCandidates(line, 999, True)
    
    i = 0
    for i in range(0, len(candidates)):
        entId = title2id(oMentions[i][1])
        j = 0
        candsRepl = []
        for cand in candidates[i]:
            if j >= cPerM:
                break
            
            if cand[0] == entId:
                candsRepl.append([entId, 1, cand[1]]) # put in correct cand id and popularity
                j += 1
            elif j < cPerM:
                candsRepl.append([cand[0], 0, cand[1]]) # put false cand in
                j += 1
        candidates[i] = candsRepl
    
    i = 0 # index of mention
    
    hasCoherence = False # whether coherence scores for this line were obtained
    
    # see each mention
    for mention in oMentions:
    
        entId = title2id(mention[1]) # id of the true entity
                
        candList = candidates[i]
        
        # normalize popularity scores
        cScrs = []
        for cand in candList:
            cScrs.append(cand[2])
        cScrs = normalize(cScrs)
        j = 0
        for cand in candList:
            cand[2] = cScrs[j]
            j += 1
          
        # get score from context1 method
        context = getMentionsInSentence(line, line['mentions'][i]) # get context for some w methods
        cScrs = getContext1Scores(line['text'][mention[0]], context, candList)
        cScrs = normalize(cScrs)
        # apply score to candList
        for j in range(0, len(candList)):
            candList[j].append(cScrs[j])
            
        # get score from context2 method
        context = getMentionsInSentence(line, line['mentions'][i]) # get context for some w methods
        cScrs = getContext2Scores(line['text'][mention[0]], context, candList)
        cScrs = normalize(cScrs)
        # apply score to candList
        for j in range(0, len(candList)):
            candList[j].append(cScrs[j])
        
        # get score form word2vec
        context = getMentionSentence(oText, line['mentions'][i], asList = True)
        cScrs = getWord2VecScores(context, candList)
        #cScrs = normalize(cScrs)
        # apply score to candList
        for j in range(0, len(candList)):
            candList[j].append(cScrs[j])

        # get score from coherence
        if hasCoherence == False:
            cohScores = coherence_scores_driver(candidates, 5, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb")
            hasCoherence = True
        for j in range(0, len(candList)):
            candList[j].append(cohScores[i][j])
            
        # put the mention id
        for j in range(len(candList)):
            candList[j].append(mNum)
            
        allCands.append(candList)
        
        mNum += 1
        
        i += 1
    f += 1
    print 'Line: ' + str(f)
        

with open('/users/cs/amaral/wikisim/wikification/learning-data/el-5000-hybridgen.txt', 'w') as f:
    for thing in allCands:
        for thingy in thing:
            f.write(str(thingy)[1:-1] + '\n')