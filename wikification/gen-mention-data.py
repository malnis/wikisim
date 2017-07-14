
"""
Generate data to be used for entity recognition.
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
dsPath = os.path.join(pathStrt,'wiki-mentions.30000.json')

with open(dsPath, 'r') as dataFile:
    dataLines = []
    i = 0
    for line in dataFile:
        dataLines.append(json.loads(line.decode('utf-8').strip()))
        i += 1
        if i >= 1:
            break
        
lnum = 0
for line in dataLines:
    
    oMentions = copy.deepcopy(line['mentions']) # mentions in original form
    oText = " ".join(copy.deepcopy(line['text']))
    
    line['mentions'] = mentionStartsAndEnds(line)
    
    # see each mention
    for mention in oMentions:
        print mention
        
        
    lnum += 1
    print 'Line: ' + str(lnum)
        

with open('/users/cs/amaral/wikisim/wikification/learning-data/er-10000.txt', 'w') as f:
    