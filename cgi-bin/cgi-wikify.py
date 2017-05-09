#!/users/grad/sajadi/backup/anaconda2/bin/python
#/home/sajadi/anaconda2/bin/python


import json
import cgi, cgitb 

import sys

sys.path.insert(0,'..')
from wikisim.wikify import *

log('cgi-wikify started');

# Import modules for CGI handling 

# Create instance of FieldStorage 
form = cgi.FieldStorage() 

# Get data from fields (fill in with option later)
inputText = form.getvalue('inputtext')

log('param %s', inputText);

print "Content-type:application/json\r\n\r\n"

if inputText == "":
	print json.dumps({"err": "Text cannot be empty."})
	exit()

# get the wikification results
anchors = wikify(inputText, True, method='popular')
newText = ""
anchors = sorted(anchors, key=itemgetter('start')) # make sure anchors are sorted
anchorIndex = 0 # keep track of current anchor added
i = 0 
while i < len(inputText):
    if anchorIndex < len(anchors) and i == anchors[anchorIndex]['start']:
        anchor = anchors[anchorIndex]
        newText += ("<a href=\"https://en.wikipedia.org/wiki/" + anchor['wikiTitle']
                   + "\" target=\"_blank\">" + anchor['mention'] + "</a>")
        i = anchors[anchorIndex]['end']
        anchorIndex += 1
    else:
        newText += inputText[i]
        i += 1

print json.dumps({"text":newText})
		
close()

log('finished')
