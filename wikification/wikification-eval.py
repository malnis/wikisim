"""
This is for testing performance of wikification.
Micro and Macro of BOT and normal.
"""

from __future__ import division
from wikification import *
from IPython.display import clear_output
import copy
from datetime import datetime
import tagme
import os
import json
from sets import Set

tagme.GCUBE_TOKEN = "f6c2ba6c-751b-4977-a94c-c140c30e9b92-843339462"
pathStrt = '/users/cs/amaral/wsd-datasets'

# the data sets for performing on
datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')},
            {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')},
            {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},
            {'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]

# many different option for combinations of datasets for smaller tests
#datasets = [{'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}]
#datasets = [{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')}]
#datasets = [{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')}]
datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]

# 'popular', 'context1', 'context2', 'word2vec', 'coherence', 'tagme', 'multi'
methods = ['multi']
# 'lmart', 'gbr', 'etr', 'rfr'
mlModel = 'lmart' # to be used with method multi
erMethod = 'cls1' # method for entity recognition / mention extraction

if 'word2vec' in methods:
    try:
        word2vec
    except:
        word2vec = gensim_loadmodel('/users/cs/amaral/cgmdir/WikipediaClean5Negative300Skip10.Ehsan/WikipediaClean5Negative300Skip10')
        
# can do both, none would be pointless
doSplit = True # mentions are given
doManual = False # mentions not given

verbose = True # decides how much stuff to ouput

maxCands = 20 # amount of candidates for entity candidate generation (20 prefered)
doHybrid = False # whether to do hybrid candidate generation (False prefered)


performances = {} # record data here

# for each dataset, run all methods
for dataset in datasets:
    performances[dataset['name']] = {}
    # get the data from dataset
    dataFile = open(dataset['path'], 'r')
    dataLines = []
    
    # get all lines
    for line in dataFile:
        dataLines.append(json.loads(line.decode('utf-8').strip()))
        
    print '\n' + dataset['name'] + '\n'
    
    # run each method on the data set
    for mthd in methods:
        print mthd
        print str(datetime.now()) + '\n'
        
        ## reset counters
        # micro scores
        totalMicroPrecS = 0
        totalMicroPrecM = 0
        totalMicroRecS = 0
        totalMicroRecM = 0
        # macro scores
        totalMacroPrecS = 0
        totalMacroPrecM = 0
        totalMacroRecS = 0
        totalMacroRecM = 0
        # BOT micro scores
        totalBotMicroPrecS = 0
        totalBotMicroPrecM = 0
        totalBotMicroRecS = 0
        totalBotMicroRecM = 0
        # BOT macro scores
        totalBotMacroPrecS = 0
        totalBotMacroPrecM = 0
        totalBotMacroRecS = 0
        totalBotMacroRecM = 0
        # amount of lines done in dataset
        totalLines = 0
        
        # each method tests all lines
        for line in dataLines:
            if verbose:
                print str(totalLines + 1)
            
            # get absolute text indexes and entity id of each given mention
            trueEntities = mentionStartsAndEnds(copy.deepcopy(line), forTruth = True) # the ground truth
            
            oData = copy.deepcopy(line) # copy of the line data
            
            # get results for pre split string
            if doSplit and mthd <> 'tagme': # presplit no work on tagme
                # original split string with mentions given
                resultS = wikifyEval(copy.deepcopy(line), True, hybridC = doHybrid, maxC = maxCands, 
                                     method = mthd, model = mlModel, erMethod = erMethod)
                precS = precision(trueEntities, resultS) # precision of pre-split
                recS = recall(trueEntities, resultS) # recall of pre-split
                
                # micro scores
                totalMicroPrecS += len(trueEntities) * precS
                totalMicroRecS += len(trueEntities) * recS
                # macro scores
                totalMacroPrecS += precS
                totalMacroRecS += recS
                
                # get bot scores
                trueSet = Set()
                for truEnt in trueEntities:
                    trueSet.add(truEnt[2])
                mySet = Set()
                for res in resultS:
                    mySet.add(res[2])
                precS = len(trueSet & mySet)/len(mySet)
                recS = len(trueSet & mySet)/len(trueSet)
                
                # BOT micro scores
                totalBotMicroPrecS += len(trueSet) * precS
                totalBotMicroRecS += len(trueSet) * recS
                # BOT macro scores
                totalBotMacroPrecS += precS
                totalBotMacroRecS += recS
                
                if verbose:
                    print 'Split: ' + str(precS) + ', ' + str(recS) + ', ' + str(f1S)
                
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
                    resultM = wikifyEval(" ".join(line['text']), True, hybridC = doHybrid, 
                                         maxC = maxCands, method = mthd, model = mlModel, erMethod = erMethod)
                
                precM = precision(trueEntities, resultM) # precision of manual split
                recM = recall(trueEntities, resultM) # recall of manual split
                
                """
                I think the math for micro scores are wrong in manual
                """
                
                # micro scores
                totalMicroPrecM += len(trueEntities) * precM
                totalMicroRecM += len(trueEntities) * recM
                # macro scores
                totalMacroPrecM += precM
                totalMacroRecM += recM
                
                # get bot scores
                trueSet = Set()
                for truEnt in trueEntities:
                    trueSet.add(truEnt[2])
                mySet = Set()
                for res in resultM:
                    mySet.add(res[2])
                precM = len(trueSet & mySet)/len(mySet)
                recM = len(trueSet & mySet)/len(trueSet)
                
                # BOT micro scores
                totalBotMicroPrecM += len(trueSet) * precM
                totalBotMicroRecM += len(trueSet) * recM
                # BOT macro scores
                totalBotMacroPrecM += precM
                totalBotMacroRecM += recM
                
            totalLines += 1
        
        # record results for this method on this dataset
        performances[dataset['name']][mthd] = 
                  {'S Micro Prec':1/1, 
                   'M Micro Prec':1/1,
                   'S Micro Rec':1/1, 
                   'M Micro Rec':1/1,
                   'S Micro F1':1/1,
                   'M Micro F1':1/1,

                   'S Macro Prec':totalMacroPrecS/totalLines,
                   'M Macro Prec':totalMacroPrecM/totalLines,
                   'S Macro Rec':totalMacroRecS/totalLines, 
                   'M Macro Rec':totalMacroRecM/totalLines,
                   'S Macro F1':(2*totalMacroPrecS*totalMacroRecS)/(totalMacroPrecS+totalMacroRecS),
                   'M Macro F1':(2*totalMacroPrecM*totalMacroRecM)/(totalMacroPrecM+totalMacroRecM),

                   'S BOT Micro Prec':1/1, 
                   'M BOT Micro Prec':1/1,
                   'S BOT Micro Rec':1/1, 
                   'M BOT Micro Rec':1/1,
                   'S BOT Micro F1':1/1,
                   'M BOT Micro F1':1/1,

                   'S BOT Macro Prec':totalBotMacroPrecS/totalLines,
                   'M BOT Macro Prec':totalBotMacroPrecM/totalLines,
                   'S BOT Macro Rec':totalBotMacroRecS/totalLines, 
                   'M BOT Macro Rec':totalBotMacroRecM/totalLines,
                   'S BOT Macro F1':(2*totalBotMacroPrecS*totalBotMacroRecS)/(totalBotMacroPrecS+totalBotMacroRecS),
                   'M BOT Macro F1':(2*totalBotMacroPrecM*totalBotMacroRecM)/(totalBotMacroPrecM+totalBotMacroRecM)
                   }

with open('/users/cs/amaral/wikisim/wikification/wikification_results.txt', 'a') as resultFile:
    resultFile.write('\n' + str(datetime.now()) + '\n' 
                     + 'maxCands: ' + str(maxCands) + '\n'
                     + 'mlModel: ' + mlModel + '\n'
                     + 'erMethod: ' + erMethod + '\n'
                     + 'doHybrid: ' + str(doHybrid) + '\n'
                     + str(datetime.now()) + '\n\n')
    comment = ''
    resultFile.write('Comment: ' + comment + '\n\n')
    for dataset in datasets:
        resultFile.write(dataset['name'] + ':\n')
        for mthd in methods:
            if doSplit and doManual:
                resultFile.write(mthd + ':'
                       + '\n    S Micro Prec :' + str(performances[dataset['name']][mthd]['S Micro Prec'])
                       + '\n    S Micro Rec :' + str(performances[dataset['name']][mthd]['S Micro Rec']) 
                       + '\n    S Micro F1 :' + str(performances[dataset['name']][mthd]['S Micro F1'])
                       + '\n    S Macro Prec :' + str(performances[dataset['name']][mthd]['S Macro Prec'])
                       + '\n    S Macro Rec :' + str(performances[dataset['name']][mthd]['S Macro Rec']) 
                       + '\n    S Macro F1 :' + str(performances[dataset['name']][mthd]['S Macro F1'])
                       + '\n    S BOT Micro Prec :' + str(performances[dataset['name']][mthd]['S BOT Micro Prec'])
                       + '\n    S BOT Micro Rec :' + str(performances[dataset['name']][mthd]['S BOT Micro Rec']) 
                       + '\n    S BOT Micro F1 :' + str(performances[dataset['name']][mthd]['S BOT Micro F1'])
                       + '\n    S BOT Macro Prec :' + str(performances[dataset['name']][mthd]['S BOT Macro Prec'])
                       + '\n    S BOT Macro Rec :' + str(performances[dataset['name']][mthd]['S BOT Macro Rec']) 
                       + '\n    S BOT Macro F1 :' + str(performances[dataset['name']][mthd]['S BOT Macro F1']) + '\n'
                       + '\n    M Micro Prec :' + str(performances[dataset['name']][mthd]['M Micro Prec'])
                       + '\n    M Micro Rec :' + str(performances[dataset['name']][mthd]['M Micro Rec']) 
                       + '\n    M Micro F1 :' + str(performances[dataset['name']][mthd]['M Micro F1'])
                       + '\n    M Macro Prec :' + str(performances[dataset['name']][mthd]['M Macro Prec'])
                       + '\n    M Macro Rec :' + str(performances[dataset['name']][mthd]['M Macro Rec']) 
                       + '\n    M Macro F1 :' + str(performances[dataset['name']][mthd]['M Macro F1'])
                       + '\n    M BOT Micro Prec :' + str(performances[dataset['name']][mthd]['M BOT Micro Prec'])
                       + '\n    M BOT Micro Rec :' + str(performances[dataset['name']][mthd]['M BOT Micro Rec']) 
                       + '\n    M BOT Micro F1 :' + str(performances[dataset['name']][mthd]['M BOT Micro F1'])
                       + '\n    M BOT Macro Prec :' + str(performances[dataset['name']][mthd]['M BOT Macro Prec'])
                       + '\n    M BOT Macro Rec :' + str(performances[dataset['name']][mthd]['M BOT Macro Rec']) 
                       + '\n    M BOT Macro F1 :' + str(performances[dataset['name']][mthd]['M BOT Macro F1']) + '\n')
            elif doSplit:
                resultFile.write(mthd + ':'
                       + '\n    S Micro Prec :' + str(performances[dataset['name']][mthd]['S Micro Prec'])
                       + '\n    S Micro Rec :' + str(performances[dataset['name']][mthd]['S Micro Rec']) 
                       + '\n    S Micro F1 :' + str(performances[dataset['name']][mthd]['S Micro F1'])
                       + '\n    S Macro Prec :' + str(performances[dataset['name']][mthd]['S Macro Prec'])
                       + '\n    S Macro Rec :' + str(performances[dataset['name']][mthd]['S Macro Rec']) 
                       + '\n    S Macro F1 :' + str(performances[dataset['name']][mthd]['S Macro F1'])
                       + '\n    S BOT Micro Prec :' + str(performances[dataset['name']][mthd]['S BOT Micro Prec'])
                       + '\n    S BOT Micro Rec :' + str(performances[dataset['name']][mthd]['S BOT Micro Rec']) 
                       + '\n    S BOT Micro F1 :' + str(performances[dataset['name']][mthd]['S BOT Micro F1'])
                       + '\n    S BOT Macro Prec :' + str(performances[dataset['name']][mthd]['S BOT Macro Prec'])
                       + '\n    S BOT Macro Rec :' + str(performances[dataset['name']][mthd]['S BOT Macro Rec']) 
                       + '\n    S BOT Macro F1 :' + str(performances[dataset['name']][mthd]['S BOT Macro F1']) + '\n')
            elif doManual:
                resultFile.write(mthd + ':'
                       + '\n    M Micro Prec :' + str(performances[dataset['name']][mthd]['M Micro Prec'])
                       + '\n    M Micro Rec :' + str(performances[dataset['name']][mthd]['M Micro Rec']) 
                       + '\n    M Micro F1 :' + str(performances[dataset['name']][mthd]['M Micro F1'])
                       + '\n    M Macro Prec :' + str(performances[dataset['name']][mthd]['M Macro Prec'])
                       + '\n    M Macro Rec :' + str(performances[dataset['name']][mthd]['M Macro Rec']) 
                       + '\n    M Macro F1 :' + str(performances[dataset['name']][mthd]['M Macro F1'])
                       + '\n    M BOT Micro Prec :' + str(performances[dataset['name']][mthd]['M BOT Micro Prec'])
                       + '\n    M BOT Micro Rec :' + str(performances[dataset['name']][mthd]['M BOT Micro Rec']) 
                       + '\n    M BOT Micro F1 :' + str(performances[dataset['name']][mthd]['M BOT Micro F1'])
                       + '\n    M BOT Macro Prec :' + str(performances[dataset['name']][mthd]['M BOT Macro Prec'])
                       + '\n    M BOT Macro Rec :' + str(performances[dataset['name']][mthd]['M BOT Macro Rec']) 
                       + '\n    M BOT Macro F1 :' + str(performances[dataset['name']][mthd]['M BOT Macro F1']) + '\n')
                
    resultFile.write('\n' + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' + '\n')