
"""
Train model and everything here in a script because ssh and jupyter are failing me.
"""

allX = []
allY = []
allMId = []

trainX = []
trainY = []
trainMId = []
valiX = []
valiY = []
valiMId = []
testX = []
testY = []
testMId = []

linesToUse = 1000000 # limit amount of total data
totalLines = 0
# first try with just getting all data
with open('/users/cs/amaral/wikisim/wikification/learning-data/el-5000-hybridgen.txt', 'r') as f:
    for line in f:
        totalLines += 1
        if totalLines > linesToUse:
            break
        data = line.split(',')
        allX.append([float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6])])
        allY.append(int(data[1]))
        allMId.append(long(data[7]))
        
# split 75, 25
trainLines = int(totalLines * 0.75)
valiLines = int(totalLines * 0.0)
testLines = int(totalLines * 0.25)

for i in range(0, trainLines):
    trainX.append(allX[i])
    trainY.append(allY[i])
    trainMId.append(allMId[i])

for i in range(trainLines, trainLines + valiLines):
    valiX.append(allX[i])
    valiY.append(allY[i])
    valiMId.append(allMId[i])
    
for i in range(trainLines + valiLines, trainLines + valiLines + testLines):
    testX.append(allX[i])
    testY.append(allY[i])
    testMId.append(allMId[i])
    
print len(trainX)
print len(testX)

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sys
sys.path.append('./pyltr/')
import pyltr

etr = ExtraTreesRegressor(n_estimators=200, min_samples_split=5, random_state=1, n_jobs=-1)
etr.fit(trainX, trainY)

rfr = RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=1, n_jobs=-1)
rfr.fit(trainX, trainY)

gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, loss='ls', random_state=1)
gbr.fit(trainX, trainY)

lmart = pyltr.models.LambdaMART(n_estimators=300, learning_rate=0.1, verbose = 1)
lmart.fit(trainX, trainY, trainMId)

"""
Save the model.
"""

import pickle

pickle.dump(etr, open('/users/cs/amaral/wikisim/wikification/ml-models/model-etr-2.pkl', 'wb'))
pickle.dump(rfr, open('/users/cs/amaral/wikisim/wikification/ml-models/model-rfr-2.pkl', 'wb'))
pickle.dump(gbr, open('/users/cs/amaral/wikisim/wikification/ml-models/model-gbr-2.pkl', 'wb'))
pickle.dump(lmart, open('/users/cs/amaral/wikisim/wikification/ml-models/model-lmart-2.pkl', 'wb'))