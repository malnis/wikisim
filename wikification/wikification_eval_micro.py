
"""
This is for testing performance of different wikification methods (Micro).
"""

from wikification import *
from IPython.display import clear_output
import copy
from datetime import datetime
import tagme
import os
import json

tagme.GCUBE_TOKEN = "f6c2ba6c-751b-4977-a94c-c140c30e9b92-843339462"
    

pathStrt = '/users/cs/amaral/wsd-datasets'
#pathStrt = 'C:\\Temp\\wsd-datasets'

# the data sets for performing on
datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')},
            {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')},
            {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},
            {'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]

# many different option for combonations of datasets for smaller tests
#datasets = [{'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}]
#datasets = [{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]
datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')}]
#datasets = [{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')}]

# 'popular', 'context1', 'context2', 'word2vec', 'coherence', 'tagme'
methods = ['popular', 'context2', 'coherence']

if 'word2vec' in methods:
    try:
        word2vec
    except:
        word2vec = gensim_loadmodel('/users/cs/amaral/cgmdir/WikipediaClean5Negative300Skip10.Ehsan/WikipediaClean5Negative300Skip10')

doSplit = True
doManual = False

verbose = True

maxCands = 20

performances = {}

# for each dataset, run all methods
for dataset in datasets:
    performances[dataset['name']] = {}
    # get the data from dataset
    dataFile = open(dataset['path'], 'r')
    dataLines = []
    
    # put in all lines that contain proper ascii
    for line in dataFile:
        dataLines.append(json.loads(line.decode('utf-8').strip()))
        
    print dataset['name'] + '\n'
    
    # run each method on the data set
    for mthd in methods:
        print mthd
        print str(datetime.now()) + '\n'
        
        # reset counters
        totalMentions = 0
        totalRightS = 0
        totalRightM = 0
        totalLines = 0
        
        # each method tests all lines
        for line in dataLines:
            if verbose:
                print str(totalLines + 1)
            
            # get absolute text indexes and entity id of each given mention
            trueEntities = mentionStartsAndEnds(copy.deepcopy(line), forTruth = True) # the ground truth
            
            oData = copy.deepcopy(line)
            
            totalMentions += len(trueEntities)
            
            # get results for pre split string
            if doSplit and mthd <> 'tagme': # presplit no work on tagme
                # original split string with mentions given
                resultS = wikifyEval(copy.deepcopy(line), True, maxC = maxCands, method = mthd)
                totalRightS += precision(trueEntities, resultS) * len(trueEntities)
                
                if verbose:
                    print 'Split: ' + str(totalMentions) + ', ' + str(totalRightS)
                
            # get results for manually split string
            if doManual:
                # tagme has separate way to do things
                if mthd == 'tagme':
                    antns = tagme.annotate(" ".join(line['text']))
                    resultM = []
                    for an in antns.get_annotations(0.005):
                        resultM.append([an.begin,an.end,title2id(an.entity_title)])
                else:
                    # unsplit string to be manually split and mentions found
                    resultM = wikifyEval(" ".join(line['text']), False, maxC = maxCands, method = mthd)
                
                totalRightM += precision(trueEntities, resultM) * len(trueEntities)
                
                if verbose:
                    print 'Manual: ' + str(totalMentions) + ', ' + str(totalRightM)
                
            totalLines += 1
        
        # record results for this method on this dataset
        # [avg precision split, avg precision manual, avg recall split, avg recall manual]
        performances[dataset['name']][mthd] = {'S F1':totalRightS/totalMentions,
                                               'M F1':totalRightM/totalMentions
                                              }

with open('/users/cs/amaral/wikisim/wikification/wikification_results.txt', 'a') as resultFile:
    resultFile.write('\nmaxC: ' + str(maxCands) + '\n' + str(datetime.now()) + '\n\n')
    for dataset in datasets:
        resultFile.write(dataset['name'] + ':\n')
        for mthd in methods:
            if doSplit and doManual:
                resultFile.write(mthd + ':'
                       + '\n    S Micro F1 :' + str(performances[dataset['name']][mthd]['S F1'])
                       + '\n    M Micro F1 :' + str(performances[dataset['name']][mthd]['M F1']) + '\n')
            elif doSplit:
                resultFile.write(mthd + ':'
                       + '\n    S Micro F1 :' + str(performances[dataset['name']][mthd]['S F1']) + '\n')
            elif doManual:
                resultFile.write(mthd + ':'
                       + '\n    M Micro F1 :' + str(performances[dataset['name']][mthd]['M F1']) + '\n')
                
    resultFile.write('\n' + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' + '\n')
