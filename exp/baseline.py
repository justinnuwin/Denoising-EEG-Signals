#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import mne

from repo import SubjectData


# os.environ["TASK"] = 'baseline'
# os.environ["MAT"] = 's03'

# In[3]:


import logging
logging.basicConfig(filename=f'logs/{os.environ["TASK"]}.log', level=logging.DEBUG)


# In[23]:


sd = SubjectData(f'data/{os.environ["MAT"]}')
logging.info(f'File: {os.environ["MAT"]}')


# In[24]:


tmin = -1
tmax = 4
reject_criteria = {'eeg': 350e-6}       # 150 µV The default from the overview tutorial
filter_freqs = (7, 30)
filter_props = dict(picks=['eeg'], fir_design='firwin', skip_by_annotation='edge')


# In[25]:


imagery_events = mne.find_events(sd.raw_imagery_left, stim_channel=sd.stim_channel)   # Same stim for imagery l/r
im_left_epochs = mne.Epochs(sd.raw_imagery_left, imagery_events, tmin=tmin, tmax=tmax,
                            preload=True, reject=reject_criteria).filter(*filter_freqs, **filter_props)
im_right_epochs = mne.Epochs(sd.raw_imagery_right, imagery_events, tmin=tmin, tmax=tmax,
                             preload=True, reject=reject_criteria).filter(*filter_freqs, **filter_props)


# In[26]:


from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne.decoding import CSP


# In[27]:


#im_left_epochs = im_left_epochs#.copy().crop(tmin=1., tmax=2.)   # Not sure why the guy cropped this
im_left_data = im_left_epochs.copy().pick_types(eeg=True)
im_right_data = im_right_epochs.copy().pick_types(eeg=True)
im_left_labels = im_left_epochs.events[:, -1] - 1   # Label: 0
im_right_labels = im_right_epochs.events[:, -1]     # Label: 1

im_data = np.vstack((im_left_data.get_data(), im_right_data.get_data()))
im_labels = np.hstack((im_left_labels, im_right_labels))

print('Left Imagery Data Shape:', im_left_data.get_data().shape)
print('Right Imagery Data Shape:', im_right_data.get_data().shape)
logging.info(f'Left Imagery Data Shape: {im_left_data.get_data().shape}')
logging.info(f'Right Imagery Data Shape: {im_right_data.get_data().shape}')


# In[28]:


cv = ShuffleSplit(10, test_size=0.2, random_state=42)
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, im_data, im_labels, cv=cv, n_jobs=8)

class_balance = np.mean(im_labels == im_labels[0])
class_balance = max(class_balance, 1. - class_balance)


# In[29]:


print("Classification accuracy:", np.mean(scores))
print("Class balance:", class_balance)
logging.info(f"Classification scores: {list(scores)}")
logging.info(f"Classification accuracy: {np.mean(scores)}")
logging.info(f"Class balance: {class_balance}")

