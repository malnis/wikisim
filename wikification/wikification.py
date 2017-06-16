
from __future__ import division
import sys
sys.path.append('../wikisim/')
from wikipedia import *
from operator import itemgetter
import requests
import json
import nltk
import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg
from calcsim import *
sys.path.append('../')
from wsd.wsd import *

MIN_MENTION_LENGTH = 3 # mentions must be at least this long
MIN_FREQUENCY = 20 # anchor with frequency below is ignored

def get_solr_count(s):
    """ Gets the number of documents the string occurs 
        NOTE: Multi words should be quoted
    Arg:
        s: the string (can contain AND, OR, ..)
    Returns:
        The number of documents
    """

    q='+text:(\"%s\")'%(s,)
    qstr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'indent':'on', 'wt':'json', 'q':q, 'rows':0}
    r = requests.get(qstr, params=params)
    try:
        if 'response' not in r.json():
            return 0
        else:
            return r.json()['response']['numFound']
    except:
        return 0

def get_mention_count(s):
    """
    Description:
        Returns the amount of times that the given string appears as a mention in wikipedia.
    Args:
        s: the string (can contain AND, OR, ..)
    Return:
        The amount of times the given string appears as a mention in wikipedia
    """
    
    result = anchor2concept(s)
    rSum = 0
    for item in result:
        rSum += item[1]
        
    return rSum

def mentionProb(text):
    """
    Description:
        Returns the probability that the text is a mention in Wikipedia.
    Args:
        text: 
    Return:
        The probability that the text is a mention in Wikipedia.
    """
    
    totalMentions = get_mention_count(text)
    totalAppearances = get_solr_count(text.replace(".", ""))
    
    if totalAppearances == 0:
        return 0 # a mention never used probably is not a good link
    else:
        return totalMentions/totalAppearances

def destroyExclusiveOverlaps(textData):
    """
    Description:
        Removes all overlaps that start at same letter from text data, so that only the best mention in an
        overlap set is left.
    Args:
        textData: [[start, end, text, anchProb],...]
    Return:
        textData minus the unesescary elements that overlap.
    """
    
    newTextData = [] # textData minus the unesescary parts of the overlapping
    overlappingSets = [] # stores arrays of the indexes of overlapping items from textData
    
    # creates the overlappingSets array
    i = 0
    while i < len(textData)-1:
        # even single elements considered overlapping set
        # this is root of overlapping set
        overlappingSets.append([i])
        overlapIndex = len(overlappingSets) - 1
        theBegin = textData[i][0]
        
        # look at next words until not overlap
        for j in range(i+1, len(textData)):
            # if next word starts before endiest one ends
            if textData[j][0] == theBegin:
                overlappingSets[overlapIndex].append(j)
                i = j # make sure not to repeat overlap set
            else:
                # add final word
                if j == len(textData) - 1:
                    overlappingSets.append([j])
                break
        i += 1
                    
    # get only the best overlapping element of each set
    for oSet in overlappingSets:
        bestIndex = 0
        bestScore = -1
        for i in oSet:
            score = mentionProb(textData[i][2])
            if score > bestScore:
                bestScore = score
                bestIndex = i
        
        # put right item in new textData
        newTextData.append(textData[bestIndex])
        
    return newTextData

def destroyResidualOverlaps(textData):
    """
    Description:
        Removes all overlaps from text data, so that only the best mention in an
        overlap set is left.
    Args:
        textData: [[start, end, text, anchProb],...]
    Return:
        textData minus the unesescary elements that overlap.
    """
    
    newTextData = [] # to be returned
    oSet = [] # the set of current overlaps
    rootWIndex = 0 # the word to start looking from for finding root word
    rEnd = 0 # the end index of the root word
    
    # keep looping as long as overlaps
    while True:
        oSet = []
        oSet.append(textData[rootWIndex])
        for i in range(rootWIndex + 1, len(textData)):
            # if cur start before root end
            if textData[i][0] < textData[rootWIndex][1]:
                oSet.append(textData[i])
            else:
                break # have all overlap words

        
        bestIndex = 0
        # deal with the overlaps
        if len(oSet) > 1:
            bestProb = 0
            
            # choose the most probable
            i = 0
            for mention in oSet:
                prob = mentionProb(mention[2])
                if prob > bestProb:
                    bestProb = prob
                    bestIndex = i
                i += 1
        else:
            rootWIndex += 1 # move up one if no overlaps
                
        # remove from old text data all that is not best
        for i in range(0, len(oSet)):
            if i <> bestIndex:
                textData.remove(oSet[i])
                
        # add the best to new
        if not (oSet[bestIndex] in newTextData):
            newTextData.append(oSet[bestIndex])
            
        if rootWIndex >= len(textData):
            break
    
    return newTextData
    
def mentionStartsAndEnds(textData, forTruth = False):
    """
    Description:
        Takes in a list of mentions and turns each of its mentions into the form: [wIndex, start, end]. 
        Or if forTruth is true: [[start,end,entityId]]
    Args:
        textData: {'text': [w1,w2,w3,...] , 'mentions': [[wordIndex,entityTitle],...]}, to be transformed 
            as described above.
        forTruth: Changes form to use.
    Return:
        The mentions in the form [[wIndex, start, end],...]]. Or if forTruth is true: [[start,end,entityId]]
    """
    
    curWord = 0 
    curStart = 0
    for mention in textData['mentions']:
        while curWord < mention[0]:
            curStart += len(textData['text'][curWord]) + 1
            curWord += 1
            
        ent = mention[1] # store entity title in case of forTruth
        mention.pop() # get rid of entity text
        
        if forTruth:
            mention.pop() # get rid of wIndex too
            
        mention.append(curStart) # start of the mention
        mention.append(curStart + len(textData['text'][curWord])) # end of the mention
        
        if forTruth:
            mention.append(title2id(ent)) # put on entityId
    
    return textData['mentions']
     
def mentionExtract(text):
    """
    Description:
        Takes in a text and splits it into the different words/mentions.
    Args:
        phrase: The text to be split.
    Return:
        The text split it into the different words / mentions: 
        {'text':[w1,w2,...], 'mentions': [[wIndex,begin,end],...]}
    """
    
    addr = 'http://localhost:8983/solr/enwikianchors20160305/tag'
    params={'overlaps':'ALL', 'tagsLimit':'5000', 'fl':'id','wt':'json','indent':'on'}
    r = requests.post(addr, params=params, data=text.encode('utf-8'))
    textData0 = r.json()['tags']
    
    splitText = [] # the text now in split form
    mentions = [] # mentions before remove inadequate ones
    
    textData = [] # [[begin,end,word,anchorProb],...]
    
    i = 0 # for wordIndex
    # get rid of extra un-needed Solr data, and add in anchor probability
    for item in textData0:
        totalMentions = get_mention_count(text[item[1]:item[3]])
        totalAppearances = get_solr_count(text[item[1]:item[3]].replace(".", ""))
        if totalAppearances == 0:
            anchorProb = 0
        else:
            anchorProb = totalMentions/totalAppearances
        # put in the new clean textData
        textData.append([item[1], item[3], text[item[1]:item[3]], anchorProb, i])
        i += 1
        
        # also fill split text
        splitText.append(text[item[1]:item[3]])
    
    # get rid of overlaps
    textData = destroyExclusiveOverlaps(textData)
    textData = destroyResidualOverlaps(textData)
        
    # gets the POS labels for the words
    postrs = []
    for item in textData:
        postrs.append(item[2])
    postrs = nltk.pos_tag(postrs)
    for i in range(0,len(textData)):
        textData[i].append(postrs[i]) # [5][1] is index of type of word
    
    mentionPThrsh = 0.001 # for getting rid of unlikelies
    
    # put in only good mentions
    for item in textData:
        if (item[3] >= mentionPThrsh # if popular enough, and either some type of noun or JJ or CD
                and (item[5][1][0:2] == 'NN' or item[5][1] == 'JJ' or item[5][1] == 'CD')):
            mentions.append([item[4], item[0], item[1]]) # wIndex, start, end
    
    # get in same format as dataset provided data
    newTextData = {'text':splitText, 'mentions':mentions}
    
    return newTextData

def generateCandidates(textData, maxC):
    """
    Description:
        Generates up to maxC candidates for each possible mention word in phrase.
    Args:
        textData: A text in split form along with its suspected mentions.
        maxC: The max amount of candidates to accept.
    Return:
        The top maxC candidates for each possible mention word in textData.
    """
    
    candidates = []
    
    for mention in textData['mentions']:
        results = sorted(anchor2concept(textData['text'][mention[0]]), key = itemgetter(1), 
                          reverse = True)
        candidates.append(results[:maxC]) # take up to maxC of the results
    
    return candidates

def precision(truthSet, mySet):
    """
    Description:
        Calculates the precision of mySet against the truthSet.
    Args:
        truthSet: The 'right' answers for what the entities are. [[start,end,id],...]
        mySet: My code's output for what it thinks the right entities are. [[start,end,id],...]
    Return:
        The precision: (# of correct entities)/(# of found entities)
    """
    
    numFound = len(mySet)
    numCorrect = 0 # incremented in for loop
    
    truthIndex = 0
    myIndex = 0
    
    while truthIndex < len(truthSet) and myIndex < len(mySet):
        if mySet[myIndex][0] < truthSet[truthIndex][0]:
            if mySet[myIndex][1] > truthSet[truthIndex][0]:
                # overlap with mine behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine not even reach truth
                myIndex += 1
                
        elif mySet[myIndex][0] == truthSet[truthIndex][0]:
            # same mention (same start atleast)
            if truthSet[truthIndex][2] == mySet[myIndex][2]:
                numCorrect += 1
                truthIndex += 1
                myIndex += 1
            elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                # truth ends first
                truthIndex += 1
            else:
                # mine ends first
                myIndex += 1
                  
        elif mySet[myIndex][0] > truthSet[truthIndex][0]:
            if mySet[myIndex][0] < truthSet[truthIndex][1]:
                # overlap with truth behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine beyond mention, increment truth
                truthIndex += 1

    #print 'correct: ' + str(numCorrect) + '\nfound: ' + str(numFound)
    if numFound == 0:
        return 0
    else:
        return (numCorrect/numFound)

def recall(truthSet, mySet):
    """
    Description:
        Calculates the recall of mySet against the truthSet.
    Args:
        truthSet: The 'right' answers for what the entities are. [[start,end,id],...]
        mySet: My code's output for what it thinks the right entities are. [[start,end,id],...]
    Return:
        The recall: (# of correct entities)/(# of actual entities)
    """
    
    numActual = len(truthSet)
    numCorrect = 0 # incremented in for loop)
    
    truthIndex = 0
    myIndex = 0
    
    while truthIndex < len(truthSet) and myIndex < len(mySet):
        if mySet[myIndex][0] < truthSet[truthIndex][0]:
            if mySet[myIndex][1] > truthSet[truthIndex][0]:
                # overlap with mine behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine not even reach truth
                myIndex += 1
                
        elif mySet[myIndex][0] == truthSet[truthIndex][0]:
            # same mention (same start atleast)
            if truthSet[truthIndex][2] == mySet[myIndex][2]:
                numCorrect += 1
                truthIndex += 1
                myIndex += 1
            elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                # truth ends first
                truthIndex += 1
            else:
                # mine ends first
                myIndex += 1
                  
        elif mySet[myIndex][0] > truthSet[truthIndex][0]:
            if mySet[myIndex][0] < truthSet[truthIndex][1]:
                # overlap with truth behind
                if truthSet[truthIndex][2] == mySet[myIndex][2]:
                    numCorrect += 1
                    truthIndex += 1
                    myIndex += 1
                elif truthSet[truthIndex][1] < mySet[myIndex][1]:
                    # truth ends first
                    truthIndex += 1
                else:
                    # mine ends first
                    myIndex += 1
            else:
                # mine beyond mention, increment truth
                truthIndex += 1
                
    if numActual == 0:
        return 0
    else:
        return (numCorrect/numActual)
    
def mentionPrecision(trueMentions, otherMentions):
    """
    Description:
        Calculates the precision of otherMentions against the trueMentions.
    Args:
        trueMentions: The 'right' answers for what the mentions are.
        otherMentions: Our mentions obtained through some means.
    Return:
        The precision: (# of correct mentions)/(# of found mentions)
    """
    
    numFound = len(otherMentions)
    numCorrect = 0 # incremented in for loop
    
    trueIndex = 0
    otherIndex = 0
    
    while trueIndex < len(trueMentions) and otherIndex < len(otherMentions):
        # if mentions start and end on the same
        if (trueMentions[trueIndex][0] == otherMentions[otherIndex][0]
               and trueMentions[trueIndex][1] == otherMentions[otherIndex][1]):
            #print ('MATCH: [' + str(trueMentions[trueIndex][0]) + ',' + str(trueMentions[trueIndex][1]) + ']' + trueMentions[trueIndex][2] 
            #       + ' <===> [' + str(otherMentions[otherIndex][0]) + ',' + str(otherMentions[otherIndex][1]) + ']' + otherMentions[otherIndex][2])
            numCorrect += 1
            trueIndex += 1
            otherIndex += 1
        # if true mention starts before the other starts
        elif trueMentions[trueIndex][0] < otherMentions[otherIndex][0]:
            #print ('FAIL: [' + str(trueMentions[trueIndex][0]) + ',' + str(trueMentions[trueIndex][1]) + ']' + trueMentions[trueIndex][2] 
            #       + ' <XXX> [' + str(otherMentions[otherIndex][0]) + ',' + str(otherMentions[otherIndex][1]) + ']' + otherMentions[otherIndex][2])
            trueIndex += 1
        # if other mention starts before the true starts (same doesnt matter)
        elif trueMentions[trueIndex][0] >= otherMentions[otherIndex][0]:
            #print ('FAIL: [' + str(trueMentions[trueIndex][0]) + ',' + str(trueMentions[trueIndex][1]) + ']' + trueMentions[trueIndex][2] 
            #       + ' <XXX> [' + str(otherMentions[otherIndex][0]) + ',' + str(otherMentions[otherIndex][1]) + ']' + otherMentions[otherIndex][2])
            otherIndex += 1
        else:
            print 'AAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHH!!!!!!!!!!!!!!!!!!!'

    #print 'correct: ' + str(numCorrect) + '\nfound: ' + str(numFound)
    if numFound == 0:
        return 0
    else:
        return (numCorrect/numFound)

def mentionRecall(trueMentions, otherMentions):
    """
    Description:
        Calculates the recall of otherMentions against the trueMentions.
    Args:
        trueMentions: The 'right' answers for what the mentions are.
        otherMentions: Our mentions obtained through some means.
    Return:
        The recall: (# of correct entities)/(# of actual entities)
    """
    
    numActual = len(trueMentions)
    numCorrect = 0 # incremented in for loop)
    
    trueIndex = 0
    otherIndex = 0
    
    while trueIndex < len(trueMentions) and otherIndex < len(otherMentions):
        # if mentions start and end on the same
        if (trueMentions[trueIndex][0] == otherMentions[otherIndex][0]
               and trueMentions[trueIndex][1] == otherMentions[otherIndex][1]):
            numCorrect += 1
            trueIndex += 1
            otherIndex += 1
        # if true mention starts before the other starts
        elif trueMentions[trueIndex][0] < otherMentions[otherIndex][0]:
            trueIndex += 1
        # if other mention starts before the true starts (same doesnt matter)
        elif trueMentions[trueIndex][0] >= otherMentions[otherIndex][0]:
            otherIndex += 1
        
    print 'correct: ' + str(numCorrect) + '\nactual: ' + str(numActual)
    if numActual == 0:
        return 0
    else:
        return (numCorrect/numActual)
    
def getSurroundingWords(text, mIndex, window, asList = False):
    """
    Description:
        Returns the words surround the given mention. Expanding out window elements
        on both sides.
    Args:
        text: A list of words.
        mIndex: The index of the word that is the center of where to get surrounding words.
        window: The amount of words to the left and right to get.
        asList: Whether to return the words as a list, otherwise just a string.
    Return:
        The words that surround the given mention. Expanding out window elements
        on both sides.
    """
    
    imin = mIndex - window
    imax = mIndex + window + 1
    
    # fix extreme bounds
    if imin < 0:
        imin = 0
    if imax > len(text):
        imax = len(text)
        
    if asList == True:
        words = (text[imin:mIndex] + text[mIndex+1:imax])
    else:
        words = " ".join(text[imin:mIndex] + text[mIndex+1:imax])
    
    # return surrounding part of word minus the mIndex word
    return words

def getMentionSentence(text, mention, asList = False):
    """
    Description:
        Returns the sentence of the mention, minus the mention.
    Args:
        text: The text to get the sentence from.
        index: The mention.
        asList: Whether to return the words as a list, otherwise just a string.
    Return:
        The sentence of the mention, minus the mention.
    """
    
    # the start and end indexes of the sentence
    sStart = 0
    sEnd = 0
    
    # get sentences using nltk
    sents = nltk.sent_tokenize(text)
    
    # find sentence that mention is in
    curLen = 0
    for s in sents:
        curLen += len(s)
        # if greater than begin of mention
        if curLen > mention[1]:
            # remove mention from string to not get bias from self referencing article
            if asList == True:
                sentence = (s.replace(text[mention[1]:mention[2]],"")).split(" ")
            else:
                sentence = s.replace(text[mention[1]:mention[2]],"")
            
            return sentence
        
    # in case it missed
    if asList == True:
        return []
    else:
        return ""

def escapeStringSolr(text):
    """
    Description:
        Escapes a given string for use in Solr.
    Args:
        text: The string to escape.
    Return:
        The escaped text.
    """
    
    text = text.replace("\\", "\\\\\\")
    text = text.replace('+', r'\+')
    text = text.replace("-", "\-")
    text = text.replace("&&", "\&&")
    text = text.replace("||", "\||")
    text = text.replace("!", "\!")
    text = text.replace("(", "\(")
    text = text.replace(")", "\)")
    text = text.replace("{", "\{")
    text = text.replace("}", "\}")
    text = text.replace("[", "\[")
    text = text.replace("]", "\]")
    text = text.replace("^", "\^")
    text = text.replace("\"", "\\\"")
    text = text.replace("~", "\~")
    text = text.replace("*", "\*")
    text = text.replace("?", "\?")
    text = text.replace(":", "\:")
    
    return text

def bestContext1Match(mentionStr, context, candidates):
    """
    Description:
        Uses Solr to find the candidate that gives the highest relevance when given the context.
    Args:
        mentionStr: The mention as it appears in the text
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The index of the candidate with the best relevance score from the context.
    """
    
    # put text in right format
    context = escapeStringSolr(context)
    mentionStr = escapeStringSolr(mentionStr)
    
    strIds = ['id:' +  str(strId[0]) for strId in candidates]
    
    # select all the docs from Solr with the best scores, highest first.
    addr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'fl':'id score', 'fq':" ".join(strIds), 'indent':'on',
            'q':'text:('+context.encode('utf-8')+')^1 title:(' + mentionStr.encode('utf-8')+')^1.35',
            'wt':'json'}
    r = requests.get(addr, params = params)
    
    if 'response' not in r.json():
        return 0 # default to most popular
    
    if 'docs' not in r.json()['response']:
        return 0
    
    results = r.json()['response']['docs']
    if len(results) == 0:
        return 0 # default to most popular
    
    bestId = long(r.json()['response']['docs'][0]['id'])
    
    #for doc in r.json()['response']['docs']:
        #print '[' + id2title(doc['id']) + '] -> ' + str(doc['score'])
    
    # find which index has bestId
    bestIndex = 0
    for cand in candidates:
        if cand[0] == bestId:
            return bestIndex
        else:
            bestIndex += 1
            
    return bestIndex # in case it was missed

def bestContext2Match(context, candidates):
    """
    Description:
        Uses Solr to find the candidate that gives the highest relevance when given the context.
    Args:
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The index of the candidate with the best relevance score from the context.
    """
    
    # put text in right format
    context = escapeStringSolr(context)
    strIds = ['entityid:' +  str(strId[0]) for strId in candidates]
    
    # dictionary to hold scores for each id
    scoreDict = {}
    for cand in candidates:
        scoreDict[str(cand[0])] = 0
    
    # select all the docs from Solr with the best scores, highest first.
    addr = 'http://localhost:8983/solr/enwiki20160305_context/select'
    params={'fl':'entityid', 'fq':" ".join(strIds), 'indent':'on',
            'q':'_context_:('+context.encode('utf-8')+')',
            'wt':'json'}
    r = requests.get(addr, params = params)
    
    if 'response' not in r.json():
        return 0 # default to most popular
    
    if 'docs' not in r.json()['response']:
        return 0
    
    results = r.json()['response']['docs']
    if len(results) == 0:
        return 0 # default to most popular
    
    for doc in r.json()['response']['docs']:
        scoreDict[str(doc['entityid'])] += 1
    
    # get the index that has the best score
    bestScore = 0
    bestIndex = 0
    curIndex = 0
    for cand in candidates:
        if scoreDict[str(cand[0])] > bestScore:
            bestScore = scoreDict[str(cand[0])]
            bestIndex = curIndex
        curIndex += 1
            
    return bestIndex

def bestWord2VecMatch(context, candidates):
    """
    Description:
        Uses word2vec to find the candidate with the best similarity to the context.
    Args:
        context: The words that surround the target word as a list.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The index of the candidate with the best similarity score with the context.
    """
    
    ctxVec = pd.Series(sp.zeros(300)) # default zero vector
    # add all context words together
    for word in context:
        ctxVec += getword2vector(word)
        
    # compare context vector to each of the candidates
    bestIndex = 0
    bestScore = 0
    i = 0
    for cand in candidates:
        eVec = getentity2vector(str(cand[0]))
        score = 1-sp.spatial.distance.cosine(ctxVec, eVec)
        #print '[' + id2title(cand[0]) + ']' + ' -> ' + str(score)
        # update score and index
        if score > bestScore: 
            bestIndex = i
            bestScore = score
            
        i += 1 # next index
            
    return bestIndex
    
def wikifyPopular(textData, candidates):
    """
    Description:
        Chooses the most popular candidate for each mention.
    Args:
        textData: A text in split form along with its suspected mentions.
        candidates: A list of list of candidates that each have the entity id and its frequency/popularity.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCandidates = []
    i = 0 # track which mention's candidates we are looking at
    # for each mention choose the top candidate
    for mention in textData['mentions']:
        if len(candidates[i]) > 0:
            topCandidates.append([mention[1], mention[2], candidates[i][0][0]])
        i += 1 # move to list of candidates for next mention
            
    return topCandidates

def wikifyContext(textData, candidates, oText, useSentence = False, window = 7, method2 = False):
    """
    Description:
        Chooses the candidate that has the highest relevance with the surrounding window words.
    Args:
        textData: A textData in split form along with its suspected mentions.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
        oText: The original text to be used for getting sentence.
        useSentence: Whether to set use whole sentence as context, or just windowsize.
        window: How many words on both sides of a mention to search for context.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCandidates = []
    i = 0 # track which mention's candidates we are looking at
    # for each mention choose the top candidate
    for mention in textData['mentions']:
        if len(candidates[i]) > 0:
            if not useSentence:
                context = getSurroundingWords(textData['text'], mention[0], window)
            else:
                context = getMentionSentence(oText, mention)
            #print '\nMention: ' + textData['text'][mention[0]]
            #print 'Context: ' + context
            if method2 == False:
                bestIndex = bestContext1Match(textData['text'][mention[0]], context, candidates[i])
            else:
                bestIndex = bestContext2Match(context, candidates[i])
            topCandidates.append([mention[1], mention[2], candidates[i][bestIndex][0]])
        i += 1 # move to list of candidates for next mention
        
    return topCandidates

def wikifyWord2Vec(textData, candidates, oText, useSentence = False, window = 5):
    """
    Description:
        Chooses the candidates that have the highest similarity to the context.
    Args:
        textData: A textData in split form along with its suspected mentions.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
        oText: The original text to be used for getting sentence.
        useSentence: Whether to set use whole sentence as context, or just windowsize.
        window: How many words on both sides of a mention to search for context.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCandidates = []
    i = 0 # track which mention's candidates we are looking at
    # for each mention choose the top candidate
    for mention in textData['mentions']:
        if len(candidates[i]) > 0:
            if not useSentence:
                context = getSurroundingWords(textData['text'], mention[0], window, asList = True)
            else:
                context = getMentionSentence(oText, mention, asList = True)
            #print '\nMention: ' + textData['text'][mention[0]]
            #print 'Context: ' + " ".join(context)
            bestIndex = bestWord2VecMatch(context, candidates[i])
            topCandidates.append([mention[1], mention[2], candidates[i][bestIndex][0]])
        i += 1 # move to list of candidates for next mention
        
    return topCandidates

def wikifyCoherence(textData, candidates, ws = 5):
    """
    Description:
        Chooses the candidates that have the highest coherence according to rvs pagerank method.
    Args:
        textData: A textData in split form along with its suspected mentions.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
        ws: How many words on both sides of a mention to search for context.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    topCands = [] # the top candidate from each candidate list
    candsScores = coherence_scores_driver(candidates, ws, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb")
    i = -1 # track what mention we are on
    for cScores in candsScores:
        i += 1
        
        if len(cScores) == 0:
            continue # nothing to do with this one
            
        bestScore = sorted(cScores, reverse = True)[0]
        curIndex = 0
        for score in cScores:
            if score == bestScore:
                topCands.append([textData['mentions'][i][1], textData['mentions'][i][2], candidates[i][curIndex][0]])
                break
            curIndex += 1
            
    return topCands

def wikifyEval(text, mentionsGiven, maxC = 20, method='popular', strict = False):
    """
    Description:
        Takes the text (maybe text data), and wikifies it for evaluation purposes using the desired method.
    Args:
        text: The string to wikify. Either as just the original string to be modified, or in the 
            form of: [[w1,w2,...], [[wid,entityId],...] if the mentions are given.
        mentionsGiven: Whether the mentions are given to us and the text is already split.
        maxC: The max amount of candidates to extract.
        method: The method used to wikify.
        strict: Whether to use such rules as minimum metion length, or minimum frequency of concept.
    Return:
        All of the proposed entities for the mentions, of the form: [[start,end,entityId],...].
    """
    
    if not(mentionsGiven): # if words are not in pre-split form
        textData = mentionExtract(text) # extract mentions from text
        oText = text # the original text
    else: # if they are
        textData = text
        textData['mentions'] = mentionStartsAndEnds(textData) # put mentions in right form
        oText = " ".join(text['text'])
    
    # get rid of small mentions
    if strict:
        textData['mentions'] = [item for item in textData['mentions']
                    if  len(textData['text'][item[0]]) >= MIN_MENTION_LENGTH]
    
    candidates = generateCandidates(textData, maxC)
    
    if method == 'popular':
        wikified = wikifyPopular(textData, candidates)
    elif method == 'context1':
        wikified = wikifyContext(textData, candidates, oText, useSentence = True, window = 7)
    elif method == 'context2':
        wikified = wikifyContext(textData, candidates, oText, useSentence = True, window = 7, method2 = True)
    elif method == 'word2vec':
        wikified = wikifyWord2Vec(textData, candidates, oText, useSentence = False, window = 5)
    elif method == 'coherence':
        wikified = wikifyCoherence(textData, candidates, ws = 5)
    
    # get rid of very unpopular mentions
    if strict:
        wikified = [item for item in wikified
                    if item[3] >= MIN_FREQUENCY]
        
    return wikified