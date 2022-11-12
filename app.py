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

import pandas as pd 
df = pd.DataFrame(X, columns=data.feature_names) 
df[:5]

clf_name = st.sidebar.selectbox(
        'Machine Learning algoritmni tanlang',
        ('KNN', 'SVM', 'Random Forest')
)

st.write(f"""
    Siz **{clf_name}** klassifikatoridan foydalyapsiz
""")

if clf_name == 'SVM':
    clf = SVC(C=C)
elif clf_name == 'KNN';
    clf =KNeighborsClassifier(n_neighbors=K)
else:
    clf = RandomForestClassifier(n_estimators=max_depth,
        max_depth=n_estimators, random_state=42)