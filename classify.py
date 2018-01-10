#can we predict if an NBA player will be above or below average
#in terms of (win shares per 48 minutes?)
#1/8/17

#1. Keras k-fold cross validation model 
#import libraries 
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd 
#random seed for reproduction
seed=32244
numpy.random.seed(seed)
#import dataset
nba_ws=pd.read_csv("NBA draft class.csv")
nba_ws.info()
#select variables in the model
nba_ws_sub=nba_ws[["Pk","MP","FG%","3P%","FT%","TRB.1","AST.1","WS/48>.061"]]
filtered_nba_ws_sub=nba_ws_sub[pandas.notnull(nba_ws_sub[["Pk","MP","FG%","3P%","FT%","TRB.1","AST.1","WS/48>.061"]])]
filtered_nba_ws_sub1=filtered_nba_ws_sub.dropna(how="any")
filtered_nba_ws_sub1 #removed nan values 
len(filtered_nba_ws_sub1.columns)
nba_df=filtered_nba_ws_sub1.values #type numpy.ndarray 
#split data into X (predictors) and y (response)
X=nba_df[:,0:6].astype(float)
y=nba_df[:,7]

##2. random forest classification (u of oxford)
#decision trees high variance/creates different sampoling of trees (allows for cross validation)
#gini index calculates the purity of the data created by the split point (0 is a perfect split)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(filtered_nba_ws_sub1)
plt.show()

#split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=14141)

#build the random forest model 
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=95,oob_score=True,random_state=14141)
rf_model.fit(X_train,y_train)   
#evaluate the random forest model
from sklearn.metrics import accuracy_score 
predict=rf_model.predict(X_test)
accuracy_score=accuracy_score(y_test,predict)
accuracy_score #85.7% accuracy 

#confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix=pd.DataFrame(confusion_matrix(y_test,predict))
conf_matrix 
sns.heatmap(conf_matrix)
plt.show()

###3. rf regression (predict WS/season)
from sklearn.preprocessing import StandardScaler
#explore data 
nba_ws.info()
nba_ws['WS/yr']=nba_ws['WS']/nba_ws['Yrs']
nba_predict=nba_ws[['MP','PTS','FG%','3P%','FT%','TRB.1','AST.1','WS/yr']]
nba_predict_filter=nba_predict.dropna(how="any")
len(nba_predict_filter.columns)
nba_predict_np=nba_predict_filter.values #type numpy.ndarray 
#features and target
X=nba_predict_np[:,0:6]
y=nba_predict_np[:,7]
#split the dataset 
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=834)
#standardize the dataset 
scaler=StandardScaler().fit(X_train)
X_train_scaling=pd.DataFrame(scaler.transform(X_train))
X_test_scaling=pd.DataFrame(scaler.transform(X_test))
#PCA(reduce the dimensionality of the dataset)
from sklearn.decomposition import PCA

pca=PCA()
pca.fit(X_train)
cpts=pd.DataFrame(pca.transform(X_train))
x_axis=np.arange(1,pca.n_components_+1)
pca_scaling=PCA()
pca_scaling.fit(X_train_scaling)
cpts_scaling=pd.DataFrame(pca.transform(X_train_scaling))

#visualizations

#random forest model
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=500,oob_score=True,random_state=0)
rf.fit(X_train,y_train)

from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
predict_train=rf.predict(X_train)
predict_test=rf.predict(X_test)
#model evaluations
test_score=r2_score(y_test,predict_test)
spearman=spearmanr(y_test,predict_test)

###4. keras classification model 2
nba_ws_sub=nba_ws[["Pk","MP","FG%","3P%","FT%","TRB.1","AST.1","WS/48>.061"]]
filtered_nba_ws_sub=nba_ws_sub[pandas.notnull(nba_ws_sub[["Pk","MP","FG%","3P%","FT%","TRB.1","AST.1","WS/48>.061"]])]
filtered_nba_ws_sub1=filtered_nba_ws_sub.dropna(how="any")
filtered_nba_ws_sub1 #removed nan values 
len(filtered_nba_ws_sub1.columns)
nba_df=filtered_nba_ws_sub1.values #type numpy.ndarray 
#split data into X (predictors) and y (response)
X=nba_df[:,0:6].astype(float)
y=nba_df[:,7]
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=834)
#reshape and normalize inputs
#####aside:
from keras.datasets import mnist 
# the data, shuffled and split between train and test sets 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
############
input_dim=461
X_train=X_train.reshape(461,6) #rows,columns 
X_test=X_test.reshape(154,6)
X_train=X_test.astype('float32')
X_test=X_test.astype('float32')
#convert class vectors to binary class matrices 
from keras.utils import np_utils 
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
#model building the pyramids
from keras.models import Sequential 
from keras.layers import Dense, Activation 

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
#evaluate the model
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=50, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold) #X,y works 
results.mean()*100 
results.std()*100 
#source: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

##rerun the model with data preparation
estimators=[] #empty list
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=create_baseline,epochs=50,batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True)
results=cross_val_score(pipeline,X,y,cv=kfold) #84.42% accuracy 

##tuning layers and number of neurons in the model

#a. evaluate a smaller network
#take baseline model with 6 neurons and reduce by half (forces a choice)


# smaller model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# smaller model
def small_model():
	# create model
	model = Sequential()
	model.add(Dense(3, input_dim=6, kernel_initializer='normal', activation='relu')) #input layers 
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid')) #output layers 
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=small_model,epochs=2,batch_size=4,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=7,shuffle=True)
results=cross_val_score(pipeline,X,y,cv=kfold) 

##larger network 
def model_grande():
    #create the model
    model=Sequential()
    model.add(Dense(6,input_dim=6,kernel_initializer='normal',activation='relu')) #all inputs layer  
    model.add(Dense(3,kernel_initializer='normal',activation='relu')) #subset inputs layer 
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid')) #output layer 
    #model compile
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model 

estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=model_grande,epochs=15,batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=5,shuffle=True)
results_x=cross_val_score(pipeline,X,y,cv=kfold)
results_x.mean()*100 #85.04%
results_x.std()*100 #3.48 