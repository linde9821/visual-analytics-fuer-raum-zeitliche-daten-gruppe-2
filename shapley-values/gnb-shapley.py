#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import shap


# In[3]:


tat = pd.read_csv("../data/lucas_organic_carbon_training_and_test_data_NEW.csv")
targets = pd.read_csv("../data/lucas_organic_carbon_target.csv")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(tat)
scaled_df = pd.DataFrame(scaled_data, columns=tat.columns)

tat_train, tat_test, targets_train, targets_test = train_test_split(scaled_data, targets, test_size=0.2, random_state=42)


# In[4]:


gnb = joblib.load('../models/gnb.pkl')
type(gnb)


# In[5]:


explainer = shap.explainers.Permutation(gnb.predict, tat_train, max_evals = 8001)


# In[6]:


shap_values = explainer(tat_train)


# In[ ]:


joblib.dump(shap_values, './gnb-shapley_values')
print("Shapley values saved")


# In[ ]:


#shap.plots.beeswarm(shap_values)


# In[ ]:


#explainer = shap.TreeExplainer(gnb)
#shap_values = explainer(tat_train)

#   InvalidModelError: Model type not yet supported by TreeExplainer: <class 'sklearn.naive_bayes.GaussianNB'>

