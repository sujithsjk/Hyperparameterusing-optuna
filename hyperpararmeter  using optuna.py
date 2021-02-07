#!/usr/bin/env python
# coding: utf-8

# In[3]:


import optuna
import sklearn
from sklearn import datasets
def objective(trial):
      iris = sklearn.datasets.load_iris()
      n_estimators = trial.suggest_int('n_estimators', 2, 20)
      max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
      clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
      return sklearn.model_selection.cross_val_score(clf, iris.data, iris.target, 
           n_jobs=-1, cv=3).mean()


# In[4]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


# In[5]:


trial = study.best_trial
print('Accuracy: {}'.format(trial.value))


# In[6]:


print("Best hyperparameters: {}".format(trial.params))


# In[ ]:


optuna.visualization.plot_optimization_history(study)


# In[ ]:


optuna.visualization.plot_slice(study)


# In[ ]:




