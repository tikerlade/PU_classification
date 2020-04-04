import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class PU_classifier:
    
    def __init__(self):
        self.c = 0 # p(s = 1| y = 1)
        self.nontraditional_clf = LogisticRegression() # Classifier label/unlabel
    
    def fit(self, labeled, unlabeled):
        labeled_train, labeled_test = train_test_split(labeled)
        
        X_train = np.concatenate([labeled_train, unlabeled])
        y_train = np.concatenate([np.ones(len(labeled_train)), np.zeros(len(unlabeled))])
        
        X_test = labeled_test
        y_test = np.ones(len(X_test))
        
        self.nontraditional_clf.fit(X_train, y_train)

        predictions = self.nontraditional_clf.predict_proba(X_test)[:, 1]
        self.c = sum(predictions)/len(predictions)
        
    def predict_proba(self, X):
        nontraditional_predict = self.nontraditional_clf.predict_proba(X)[:, 1]
        pos_predict = nontraditional_predict / self.c
        neg_predict = 1 - pos_predict
        
        return np.array(list(zip(neg_predict, pos_predict)))
    
    def predict(self, X, treshold=0.5):
        proba_predict = self.predict_proba(X)
        class_predict = list(map(int, proba_predict[:, 1] > treshold))
        
        return np.array(class_predict)
    
    def get_weight(self, x):
        prediction = self.nontraditional_clf.predict_proba(x)[0][1]
        
        comp1 = (1 - self.c)/self.c
        comp2 = prediction / (1 - prediction)
        
        return comp1*comp2
        
    def weight_fit(self, labeled, unlabeled):
        self.fit(labeled, unlabeled)
        
        X_train = labeled
        y_train = np.ones(len(labeled))
        weights = np.ones(len(labeled))

        for idx, x in enumerate(unlabeled):
            weight = self.get_weight([x])
            
            X_train = np.append(X_train, [x, x])
            y_train = np.append(y_train, [1, 0])
            weights = np.append(weights, [weight, 1-weight])

        X_train = X_train.reshape((-1, labeled.shape[1]))
        self.nontraditional_clf.fit(X_train, y_train, sample_weight=weights)

        predictions = self.nontraditional_clf.predict_proba(labeled)[:, 1]
        self.c = sum(predictions)/len(predictions)

    
    def coefficients(self):
        return self.clf.coef_