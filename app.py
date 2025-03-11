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

#app ki heading
st.write("""
# Explore different ML models and datasets         
""")
# data set ka name ak ak box may daal ka sidebar pay laga do.
dataset_name=st.sidebar.selectbox(
    "Select Dataset",
    ('Iris','Breast_Cancer','Wine')
)
# issi ka nichay sidebar main classifier name
classifier_name=st.sidebar.selectbox("Select Classifer",('SVM','KNN','Random Forest'))
# ab hum funcation define karain gay dataset ko lopad karnay kay liye
def load_dataset(dataset_name):
    data=None
    if dataset_name=='iris':
        data=datasets.load_iris()
    elif dataset_name=='Breast_Cancer':
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    x=data.data
    y=data.target
    return x,y
# ab hum iss funcation ko call karain gay
x,y=load_dataset(dataset_name)

# ab hum dataset ki shape aur classes target main check karain ga aur streamlit main print karain ga
st.write(f"shape of dataset: {x.shape}")
st.write(f"number of classes in target variable: {len(np.unique(y))}")

#AB HUM CLASSIFIERS KA PARAMETERS KO USER INPUT MAY ADD KARAYN GY
def add_parameter_ui(classifier_name):
    params=dict() # create empty parameters dictionary
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    elif classifier_name=='KNN':
        k=st.sidebar.slider('k',1,15)
        params['k']=k
    else:
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators
    return params
# ab iss funtion ko call karain ga
params=add_parameter_ui(classifier_name)

# ab hum classiier banayen gay base on classifier name and parameters

def get_classifier(classifier_name,params):
    clf=None
    if classifier_name=='SVM':
        clf=SVC(C=params['C'])
    elif classifier_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['k'])
        
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'])
    return clf
# ab hum iss function ko call karain ga
clf=get_classifier(classifier_name,params)

# ab dataset ko test train may split krty hain 80 20
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
#ab hum iss data ko train karty hain
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

#model ki accuracy check karain
acc=accuracy_score(y_test,y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc:.2f}")

### PLOT DATASET ###
# ab hum dataset ko plot karty hain 2 dimensions pe using pca
pca=PCA(2)
X_projected=pca.fit_transform(x)
# ab apna tranform data ko 0 aur 1 dimension may slice kr dain gy
X1=X_projected[:,0]
X2=X_projected[:,1]
#st.write(X2)
fig=plt.figure()
plt.scatter(X1,X2,c=y,cmap='viridis',alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
st.pyplot(fig)


