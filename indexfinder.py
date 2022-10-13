#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from itertools import *
#

class IndexFinder:
    def __init__(self, X, y, seed):
        #
        self.data = X
        self.target = y
        #
        self.boolen = X.corr()[y] > 0 # Calculate correlations with target feachure
        self.positive = self.boolen[self.boolen == True].index.values[1:] # determine positive
        self.negative = self.boolen[self.boolen != True].index.values # determine negative
        #
        self.num_of_positive = len(self.positive)
        self.num_of_negative = len(self.negative)
        self.num_of_combs = (2**self.num_of_positive)*(2**self.num_of_negative)
        #
        self.super_list = None
        self.top = None
        self.combinations = None
        #
        self.best_index = None
        self.seed = seed
        #
        self.best_accuracy = None
        self.best_f1 = None
        #
    def get_explor(self):
        # TO INVESTIGATE DATA BEFORE TRANING
        
        print("Positive feach:", self.num_of_positive)
        print("Negative feach:", self.num_of_negative)
        print("All feach:", self.num_of_combs)
        #
    def comb_generation(self, n_members):
        #
        posbox = [] # positive combinations
        negbox = [] # negative combinations
        count = 1
        for _ in range(n_members):
            posbox += list(combinations(self.positive, count))
            negbox += list(combinations(self.negative, count))
            count+=1
        #
        print('Positive combinations: ', len(posbox)) 
        print('Negative combinations: ', len(negbox))
        #
        self.combinations = list(product(posbox, negbox)) # All possible combinations between positive and negative values
        print('All combinations done. There are', len(self.combinations), ' combinations')
        
    def fit(self, n_models:int):
        #
        index_name = "+".join(self.combinations[0][0])+'/'+"+".join(self.combinations[0][1]) # make random index
        X = self.data.copy()
        X[index_name] = X[list(self.combinations[0][0])].sum(axis=1)/X[list(self.combinations[0][1])].sum(axis=1)
        #
        best_model = LogisticRegression(random_state=self.seed)
        #
        X_train, X_test, y_train, y_test = train_test_split(X[[index_name]], X[self.target], 
                                                            random_state=self.seed, 
                                                            stratify=X[self.target], train_size=0.75)
        # Fit model with random index
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        #
        self.best_index = index_name # Get some index - this is a primary best index
        #
        self.best_accuracy = accuracy_score(y_test, y_pred)
        self.best_f1 = f1_score(y_test, y_pred)
        #
        print("Primary index:", self.best_index)
        print("Primary accuracy:", self.best_accuracy)
        print("Primary f1:", self.best_f1)
        #
        print()
        #
        finder_dect = {self.best_index:[self.best_accuracy, self.best_f1]}
        #
        for comb in self.combinations[:n_models]:
            index_name = "+".join(comb[0])+'/'+"+".join(comb[1])
            X = self.data.copy()
            X[index_name] = X[list(comb[0])].sum(axis=1)/X[list(comb[1])].sum(axis=1)
            #
            X_train, X_test, y_train, y_test = train_test_split(X[[index_name]], X[self.target], 
                                                                random_state=self.seed, stratify=X[self.target], 
                                                                train_size=0.75)
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            #
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            # print(index_name, accuracy, f1)
            finder_dect[index_name] = [accuracy, f1]
            #
            if accuracy > self.best_accuracy and f1 > self.best_f1:
                #best_model = model
                self.best_index = index_name
                self.best_accuracy = accuracy
                self.best_f1 = f1

            else:
                continue
        #
        print("Best index:", self.best_index)
        print("Best accuracy:", self.best_accuracy)
        print("Best f1:", self.best_f1)
        
        #
        self.super_list = pd.DataFrame(finder_dect, )
        self.super_list = self.super_list.T
        self.super_list.columns = ["Accuracy", "F1"]
#         self.top_100 = self.super_list.sort_values(['Accuracy','F1'], ascending=[False, False]).head(100)
        #
    def get_top(self, n=100):
        self.top = self.super_list.sort_values(['Accuracy','F1'], ascending=[False, False]).head(n)
        print(self.top)

