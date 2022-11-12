import streamlit as st  
import numpy as np 

import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split  

from sklearn.decomposition import PCA 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier  

from sklearn.metrics import accuracy_score

st.title('Cancer Detection')
st.subheader('Cancer Detection Machine Learning Model')

data = datasets.load_breast_cancer()
X = data.data 
y = data.target

st.write('X shakli:', X.shape)
st.write('Maqsadli raqam:', len(np.unique(y)))