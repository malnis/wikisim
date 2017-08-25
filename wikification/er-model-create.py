from __future__ import division

"""
Create the machine learned models for detection of mentions
"""

import os
from sklearn.utils import shuffle

def createModels(isNovel, posNegRatio = 1, posToUse = 48428, scores = None):
    """
    Description:
        Tests different models on er.
    Args:
        isNovel: Whether to use novelty detection, or classification algorithms.
        posNegRatio: How many negative examples to have per positive example.
        posToUse: Amount of positive examples to use (48428 from 10000 dataset).
    Return:
        Nothing, but creates the models and saves the best.
    """

    poss = [] # positive instances
    negs = [] # negative instances

    # with 10000 file
        # poss: 48428
        # negs: 454139
        # about 1:10 pos to neg
    
    negToUse = posToUse * posNegRatio

    if isNovel == True:
        
        # train it on 80% of positive then test on 20% of positive with the 
        # same amount of negatives
        
        # get the nov datasets
        # positive examples
        with open('/users/cs/amaral/wikisim/wikification/learning-data/er-10000-nov.txt', 'r') as f:
            for line in f:
                data = line.split(',')
                for i in range(len(data)):
                    if i == 3: # float data
                        data[i] = float(data[i])
                    else: # int data
                        data[i] = int(data[i])
                poss.append(data)
        
        # some negatives
        with open('/users/cs/amaral/wikisim/wikification/learning-data/er-10000-cls-neg.txt', 'r') as f:
            for line in f:
                data = line.split(',')
                for i in range(len(data)):
                    if i == 3: # float data
                        data[i] = float(data[i])
                    else: # int data
                        data[i] = int(data[i])
                negs.append(data)
                
        poss = shuffle(poss)
        
        trainAmount = long(0.8 * len(poss))
        testAmount = long(0.2 * len(poss))

        XTrain = poss[0:trainAmount]
        
        XTestPos = poss[trainAmount:trainAmount + testAmount]
        XTestNeg = [data[:-1] for data in shuffle(negs)[:testAmount]]
        
        from sklearn.svm import OneClassSVM
        
        svm = OneClassSVM()
        svm.fit(XTrain)
        
        posPred = svm.predict(XTestPos)
        negPred = svm.predict(XTestNeg)
        
        wrong = 0
        for pred in posPred:
            if pred == -1:
                wrong += 1
        print 'Positive Accuracy:', 1 - wrong/testAmount
        
        wrong = 0
        for pred in negPred:
            if pred == 1:
                wrong += 1
        print 'Negative Accuracy:', 1 - wrong/testAmount
        
    else:
        # get the cls datasets
        # positive examples
        with open('/users/cs/amaral/wikisim/wikification/learning-data/er-10000-cls-pos.txt', 'r') as f:
            for line in f:
                data = line.split(',')
                for i in range(len(data)):
                    if i == 3: # float data
                        data[i] = float(data[i])
                    else: # int data
                        data[i] = int(data[i])
                poss.append(data)

        with open('/users/cs/amaral/wikisim/wikification/learning-data/er-10000-cls-neg.txt', 'r') as f:
            for line in f:
                data = line.split(',')
                for i in range(len(data)):
                    if i == 3: # float data
                        data[i] = float(data[i])
                    else: # int data
                        data[i] = int(data[i])
                negs.append(data)

        poss = shuffle(poss)[:posToUse]
        negs = shuffle(negs)[:negToUse]
        poss.extend(negs)
        allData = shuffle(poss)

        trainAmount = long(0.6 * len(allData))
        valiAmount = long(0.2 * len(allData))
        testAmount = long(0.2 * len(allData))

        XTrain = [data[:-1] for data in allData[0:trainAmount]]
        yTrain = [data[-1] for data in allData[0:trainAmount]]

        XVali = [data[:-1] for data in allData[trainAmount:trainAmount + valiAmount]]
        yVali = [data[-1] for data in allData[trainAmount:trainAmount + valiAmount]]

        XTest = [data[:-1] for data in allData[trainAmount + valiAmount:trainAmount + valiAmount + testAmount]]
        yTest= [data[-1] for data in allData[trainAmount + valiAmount:trainAmount + valiAmount + testAmount]]

        # try these classifiers
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import BaggingClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from sklearn.svm import SVC
        
        from sklearn.metrics import classification_report
        
        
        print 'adaboost:'
        abc = AdaBoostClassifier(n_estimators=300)
        abc.fit(XTrain, yTrain)
        print classification_report(yVali, abc.predict(XVali))
        #scores['abc'] = abc.score(XVali, yVali)
        #print 'adaboost:', scores['abc']

        print 'bagging:'
        bgc = BaggingClassifier(verbose=0, n_estimators=300)
        bgc.fit(XTrain, yTrain)
        print classification_report(yVali, bgc.predict(XVali))
        #scores['bgc'] = bgc.score(XVali, yVali)
        #print 'bagging:', scores['bgc']

        print 'extra trees:'
        etc = ExtraTreesClassifier(verbose=0, n_estimators=300, min_samples_split=5)
        etc.fit(XTrain, yTrain)
        print classification_report(yVali, etc.predict(XVali))
        #scores['etc'] = etc.score(XVali, yVali)
        #print 'extra trees:', scores['etc']

        print 'gradient boosting:'
        gbc = GradientBoostingClassifier(verbose=0, n_estimators=300, min_samples_split=5)
        gbc.fit(XTrain, yTrain)
        print classification_report(yVali, gbc.predict(XVali))
        #scores['gbc'] = gbc.score(XVali, yVali)
        #print 'gradient boosting:', scores['gbc']

        print 'random forest:'
        rfc = RandomForestClassifier(verbose=0, n_estimators=300, min_samples_split=5)
        rfc.fit(XTrain, yTrain)
        print classification_report(yVali, rfc.predict(XVali))
        #scores['rfc'] = rfc.score(XVali, yVali)
        #print 'random forest:', scores['rfc']

        print 'linear svc:'
        lsvc = LinearSVC(verbose=0)
        lsvc.fit(XTrain, yTrain)
        print classification_report(yVali, lsvc.predict(XVali))
        #scores['lsvc'] = lsvc.score(XVali, yVali)
        #print 'linear svc:', scores['lsvc']

        print 'svc:'
        svc = SVC(verbose=False)
        svc.fit(XTrain, yTrain)
        print classification_report(yVali, svc.predict(XVali))
        #scores['svc'] = svc.score(XVali, yVali)
        #print 'svc:', scores['svc']
        

createModels(True, posToUse = 48428)
