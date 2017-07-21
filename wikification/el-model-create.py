# originally model-create.py
"""
Train model and everything here in a script because ssh and jupyter are failing me.
"""

allX = []
allY = []
allMId = []


trainX = []
trainY = []
trainMId = []

bigTrainX = []
bigTrainY = []
bigTrainMId = []

valiX = []
valiY = []
valiMId = []

testX = []
testY = []
testMId = []


linesToUse = 10000000 # limit amount of total data
totalLines = 0
# first try with just getting all data
with open('/users/cs/amaral/wikisim/wikification/learning-data/el-10000-hybridgen.txt', 'r') as f:
    for line in f:
        totalLines += 1
        if totalLines > linesToUse:
            break
        data = line.split(',')
        allX.append([float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6])])
        allY.append(int(data[1]))
        allMId.append(long(data[7]))
        
# split 60,20,20 or 80,20 with bigTrain
trainLines = int(totalLines * 0.6)
valiLines = int(totalLines * 0.2)
testLines = int(totalLines * 0.2)

for i in range(0, trainLines):
    trainX.append(allX[i])
    trainY.append(allY[i])
    trainMId.append(allMId[i])
    
for i in range(0, trainLines + valiLines):
    bigTrainX.append(allX[i])
    bigTrainY.append(allY[i])
    bigTrainMId.append(allMId[i])

for i in range(trainLines, trainLines + valiLines):
    valiX.append(allX[i])
    valiY.append(allY[i])
    valiMId.append(allMId[i])
    
for i in range(trainLines + valiLines, trainLines + valiLines + testLines):
    testX.append(allX[i])
    testY.append(allY[i])
    testMId.append(allMId[i])
    
print 'about to start training'

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
#from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sys
sys.path.append('./pyltr/')
import pyltr

abc = AdaBoostClassifier(n_estimators=300)
abc.fit(bigTrainX, bigTrainY)

print 'adaboost done'

bgc = BaggingClassifier(verbose=1, n_estimators=300)
bgc.fit(bigTrainX, bigTrainY)

print 'bagging done'

etc = ExtraTreesClassifier(verbose=1, n_estimators=300, min_samples_split=5)
etc.fit(bigTrainX, bigTrainY)

print 'extra trees done'

gbc = GradientBoostingClassifier(verbose=1, n_estimators=300, min_samples_split=5)
gbc.fit(bigTrainX, bigTrainY)

print 'gradient boosting done'

rfc = RandomForestClassifier(verbose=1, n_estimators=300, min_samples_split=5)
rfc.fit(bigTrainX, bigTrainY)

print 'random forest done'

"""lsvc = LinearSVC(verbose=1)
lsvc.fit(bigTrainX, bigTrainY)

print 'linear svc done'

#nsvc = NuSVC(verbose=True)
#nsvc.fit(bigTrainX, bigTrainY)

#print 'nusvc done'

svc = SVC(verbose=True)
svc.fit(bigTrainX, bigTrainY)

print 'svc done'

monitor = pyltr.models.monitors.ValidationMonitor(
    valiX, valiY, valiMId, metric=pyltr.metrics.NDCG(k=10), stop_after=250)
lmart = pyltr.models.LambdaMART(n_estimators=300, learning_rate=0.1, verbose = 1)
lmart.fit(trainX, trainY, trainMId, monitor=monitor)

print 'lmart done'"""

"""
Save the model.
"""

import pickle

pickle.dump(abc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-abc-10000-hyb.pkl', 'wb'))
pickle.dump(bgc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-bgc-10000-hyb.pkl', 'wb'))
pickle.dump(etc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-etc-10000-hyb.pkl', 'wb'))
pickle.dump(gbc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-gbc-10000-hyb.pkl', 'wb'))
pickle.dump(rfc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-rfc-10000-hyb.pkl', 'wb'))
#pickle.dump(lsvc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-lsvc-10000-pop.pkl', 'wb'))
#pickle.dump(nsvc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-nsvc-10000-hyb.pkl', 'wb'))
#pickle.dump(svc, open('/users/cs/amaral/wikisim/wikification/ml-models/model-svc-10000-pop.pkl', 'wb'))
#pickle.dump(lmart, open('/users/cs/amaral/wikisim/wikification/ml-models/model-lmart-10000-pop.pkl', 'wb'))

print 'models saved'