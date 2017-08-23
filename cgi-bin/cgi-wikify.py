#!/users/cs/amaral/anaconda2/bin/python

import json
import cgi, cgitb 

import sys

sys.path.insert(0,'../wikification')
from wikification import annotateText

# Import modules for CGI handling 

# Create instance of FieldStorage 
form = cgi.FieldStorage() 

# Get data from fields (fill in with option later)
inputText = form.getvalue('inputtext')
sys.stderr.write('Input Text is: ' + inputText + ' ')
print "Content-type:application/json\r\n\r\n"

if inputText == "":
    print json.dumps({"err": "Text cannot be empty."})
    exit()

# get the wikification results
result = annotateText(inputText)
sys.stderr.write('Result is: ' + inputText + ' ')
print json.dumps({"text":result})
