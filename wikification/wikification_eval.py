
"""
This is for testing performance of different wikification methods.
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
datasets = [{'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]
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
methods = ['context2']

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
        totalPrecS = 0
        totalPrecM = 0
        totalRecS = 0
        totalRecM = 0
        totalF1S = 0
        totalF1M = 0
        totalLines = 0
        
        # each method tests all lines
        for line in dataLines:
            if verbose:
                print str(totalLines + 1)
            
            # get absolute text indexes and entity id of each given mention
            trueEntities = mentionStartsAndEnds(copy.deepcopy(line), forTruth = True) # the ground truth
            
            oData = copy.deepcopy(line)
            
            # get results for pre split string
            if doSplit and mthd <> 'tagme': # presplit no work on tagme
                # original split string with mentions given
                resultS = wikifyEval(copy.deepcopy(line), True, maxC = maxCands, method = mthd)
                precS = precision(trueEntities, resultS) # precision of pre-split
                recS = recall(trueEntities, resultS) # recall of pre-split
                try:
                    f1S = (2*precS*recS)/(precS+recS)
                except:
                    f1S = 0
                
                if verbose:
                    print 'Split: ' + str(precS) + ', ' + str(recS) + ', ' + str(f1S)
                
                # track results
                totalPrecS += precS
                totalRecS += recS
                totalF1S += f1S
                
                j = 0
                for mention in oData['mentions']:
                    try:
                        print oData['text'][mention[0]].encode('utf-8') + ':  ' + mention[1] + ' --> ' + id2title(resultS[j][2])
                    except:
                        pass
                    j += 1
                
            else:
                totalPrecS = 0
                totalRecS = 0
                totalF1S = 0
                
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
                
                precM = precision(trueEntities, resultM) # precision of manual split
                recM = recall(trueEntities, resultM) # recall of manual split
                try:
                    f1M = (2*precM*recM)/(precM+recM)
                except:
                    f1M = 0
                
                if verbose:
                    print 'Manual: ' + str(precM) + ', ' + str(recM) + ', ' + str(f1M)
                    
                # track results
                totalPrecM += precM
                totalRecM += recM
                totalF1M += f1M
            else:
                totalPrecM = 0
                totalRecM = 0
                totalF1M = 0
                
            totalLines += 1
        
        # record results for this method on this dataset
        # [avg precision split, avg precision manual, avg recall split, avg recall manual]
        performances[dataset['name']][mthd] = {'S Prec':totalPrecS/totalLines, 
                                               'M Prec':totalPrecM/totalLines,
                                              'S Rec':totalRecS/totalLines, 
                                               'M Rec':totalRecM/totalLines,
                                               'S F1':totalF1S/totalLines,
                                               'M F1':totalF1M/totalLines
                                              }

with open('/users/cs/amaral/wikisim/wikification/wikification_results.txt', 'a') as resultFile:
    resultFile.write('\nmaxC: ' + str(maxCands) + '\n\n')
    for dataset in datasets:
        resultFile.write(dataset['name'] + ':\n')
        for mthd in methods:
            if doSplit and doManual:
                resultFile.write(mthd + ':'
                       + '\n    S Prec :' + str(performances[dataset['name']][mthd]['S Prec'])
                       + '\n    S Rec :' + str(performances[dataset['name']][mthd]['S Rec'])
                       + '\n    S F1 :' + str(performances[dataset['name']][mthd]['S F1'])
                       + '\n    M Prec :' + str(performances[dataset['name']][mthd]['M Prec'])
                       + '\n    M Rec :' + str(performances[dataset['name']][mthd]['M Rec'])
                       + '\n    M F1 :' + str(performances[dataset['name']][mthd]['M F1']) + '\n')
            elif doSplit:
                resultFile.write(mthd + ':'
                       + '\n    S Prec :' + str(performances[dataset['name']][mthd]['S Prec'])
                       + '\n    S Rec :' + str(performances[dataset['name']][mthd]['S Rec']) 
                       + '\n    S F1 :' + str(performances[dataset['name']][mthd]['S F1']) + '\n')
            elif doManual:
                resultFile.write(mthd + ':'
                       + '\n    M Prec :' + str(performances[dataset['name']][mthd]['M Prec'])
                       + '\n    M Rec :' + str(performances[dataset['name']][mthd]['M Rec'])
                       + '\n    M F1 :' + str(performances[dataset['name']][mthd]['M F1']) + '\n')
                
    resultFile.write('\n' + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' + '\n')