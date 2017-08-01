#!/users/cs/amaral/anaconda2/bin/python
#/users/grad/sajadi/backup/anaconda2/bin/python
#/home/sajadi/anaconda2/bin/python


import json
import cgi, cgitb 

import sys

sys.path.insert(0,'..')
from wikification.wikification import annotateText

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
result = annotateText(inputText)
print json.dumps({"text":result})
		
close()

log('finished')
