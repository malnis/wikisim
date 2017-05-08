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

newText = inputText # text to be updated with anchor tags

# highest ending index first so can replace in string without worry
# though this is a problem untill overlaps are dealt with
anchors = sorted(anchors, key=itemgetter('end'), reverse=True)
for anchor in anchors:
	newText = (newText[0:anchor['start']-1] + 
		"<a href=\"\\https://en.wikipedia.org/wiki/" + anchor['wikiTitle'] 
		+ "\">" + anchor['mention'] + "</a>" + newText[anchor['end']:len(inputText)])

print json.dumps({"text":newText})
		
close()

log('finished')
