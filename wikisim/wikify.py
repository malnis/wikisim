from wikipedia import *
from operator import itemgetter
import requests

MIN_MENTION_LENGTH = 3 # mentions must be at least this long
MIN_FREQUENCY = 20 # anchor with frequency below is ignored

def stripSmallMentions(potAnchors):
    """
    Description:
        Removes potential anchors with mentions that are too small from the list.
    Args:
        potAnchors: The list of potential anchors, along with some additional information.
            [{'start', 'end', 'mention', 'mention variations'},...]
    Return:
        A new list of potential anchors.
    """
    newPotAnchors = [] # the new list
    for potAnchor in potAnchors:
        if potAnchor['end'] - potAnchor['start'] >= MIN_MENTION_LENGTH:
            newPotAnchors.append(potAnchor)
    
    return newPotAnchors

def getMostFrequentConcept(mentions):
    """
    Description:
        Finds the mention with the candidate concept with the most frequency.
    Args:
        mentions: A list of mentions to look for the most popular in.
    Return:
        A dictionary of the form {'mention', 'conceptId', 'freq'}.
    """
    
    # The inputted mentions along with the frequency of thier most popular concept
    mentionConceptFreqs = []
    for mention in mentions:
        # gets the most frequent concept from the current asr
        mostFrequent = sorted(anchor2concept(mention), key = itemgetter(1), reverse = True)[0]
        # mostFrequent[0] is conceptId, mostFrequent[1] is frequency of that concept
        mentionConceptFreqs.append((mention, mostFrequent[0], mostFrequent[1]))
        
    # get the mention with the highest freqency
    bestMention = sorted(mentionConceptFreqs, key = itemgetter(2), reverse = True)[0]
    
    bestMentionDict = {}
    bestMentionDict['mention'] = bestMention[0]
    bestMentionDict['conceptId'] = bestMention[1]
    bestMentionDict['freq'] = bestMention[2]
    
    return bestMentionDict

def wikifyPopular(potAnchors, useOriginalMention):
    """
    Description:
        Takes in a list potential anchors and returns the resulting anchors.
    Args:
        potAnchors: The list of potential anchors, along with some additional information.
            [{'start', 'end', 'mention', 'mentionVars'},...]
        useOriginalMention: Whether to use the mention from the original or one of the word forms.
    Return:
        The potential anchors along with the corresponding concept and frequency of that concept.
    """
    
    # if not using original mention, use the mention variation with the most frequent results
    # either way adds 'conceptId', and 'freq'
    if useOriginalMention == False:
        for potAnchor in potAnchors:
            bestMention = getMostFrequentConcept(potAnchor['mentionVars'])
            potAnchor['conceptId'] = bestMention['conceptId']
            potAnchor['freq'] = bestMention['freq']
    else:
        for potAnchor in potAnchors:
            mentionData = sorted(anchor2concept(potAnchor['mention']), 
                                 key = itemgetter(1), reverse = True)[0]
            potAnchor['conceptId'] = mentionData[0]
            potAnchor['freq'] = mentionData[1]
            
    return potAnchors

def wikify(query, useOriginalMention, method='popular'):
    """
    Description:
        Takes the query string, and wikifies it using the desired method.
    Args:
        query: The string to wikify.
        mehtod: The method used to wikify.
        useOriginalMention: Whether to use mention from the original or a potential variation.
    Return:
        The anchors along with their best matched concept from wikipedia.
    """
    
    # first get the potential anchors from solr
    addr = 'http://localhost:8983/solr/enwikianchors20160305/tag'
    params={'overlaps':'ALL', 'tagsLimit':'5000', 'fl':'id','wt':'json','indent':'on'}
    r = requests.post(addr, params=params, data=query)
    queryResult = r.json()['tags']
    
    # an array of dictionaries to hold the data of each potential anchor
    potAnchors = [] 
    # convert queryResult to potAnchors (much cleaner)
    for record in queryResult:
        potAnchors.append({'start':record[1], 'end':record[3], 
                            'mention':query[record[1]:record[3]],
                          'mentionVars':record[5]})
        
    # don't use any potential anchors below size threshold
    potAnchors = stripSmallMentions(potAnchors)
    
    if method == 'popular':
        potAnchors = wikifyPopular(potAnchors, useOriginalMention)
    
    # deal with overlap on anchors here
    
    # return anchors in finalized form
    anchors = []
    for potAnchor in potAnchors:
        if potAnchor['freq'] >= MIN_FREQUENCY: 
            anchors.append({'start':potAnchor['start'], 'end':potAnchor['end'], 
                            'mention':potAnchor['mention'],
                            'wikiTitle':id2title(potAnchor['conceptId'])})
    
    return anchors