import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression, SGDRegressor , LassoCV ,Ridge, Lasso
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
pd.options.display.mpl_style = 'default'
#read the dataset file
wine= pd.read_csv('/Users/shwetaanchan/Desktop/WineQuality.csv',sep=‘,’)
df = wine


#boxplot
df[df.columns[:11]].plot(kind='box') #requires scaling of the dataframe
scaler = MinMaxScaler()
df[df.columns[:11]] = df[df.columns[:11]].apply(lambda x: scaler.fit_transform(x)) #scaled
df[df.columns[:11]].plot(kind='box') 

#hist
df.hist()

## add a binary variable for classification
df["quality_binary"] = df["quality"].apply(lambda x: 1 if x >=6 else 0)

## create the train and test dataset
# create random list of indices

N = len(df)
l = range(N)
shuffle(l)

# get splitting indicies
trainLen = int(N*0.6)
testLen  = int(N*0.4)

# get training,  and test sets
training = df.ix[l[:trainLen]]
test     = df.ix[l[trainLen:]]

##### Modelling Part ######




    
####################            CLASSIFICATION      ##################
#feature variables
xtrain = training[training.columns[:11]]
# output labels
ytrain = training["quality_binary"]

xtest = test[test.columns[:11]]
ytest =  test["quality_binary"]

def get_score(xtrain,ytrain,xtest,ytest,classifier):
    classifier.fit(xtrain,ytrain)
    ypred = classifier.predict(xtest)
    cm = confusion_matrix(ytest, ypred)
    score = classifier.score(xtest,ytest)
    print (score)
    print (cm)
    return(score,cm)
    

## ##############   Apply Hinge Loss (SVM) 
model_hinge_l2,cm_hinge_l2 = get_score(xtrain,ytrain,xtest,ytest,SGDClassifier(loss="hinge",penalty="l2"))
## score  0.76326530612244903

## Apply Hinge Loss (SVM) with penalty l1
model_hinge_l1,cm_hinge_l1 = get_score(xtrain,ytrain,xtest,ytest,SGDClassifier(loss="hinge",penalty="l1"))
## score 0.75714285714285712
## Apply Hinge Loss (SVM) with penalty l0
model_hinge_l0,cm_hinge_l0 = get_score(xtrain,ytrain,xtest,ytest,SGDClassifier(loss="hinge",penalty=None))
# 0.77346938775510199


##############    Apply Log Loss  
model_log_l2,cm_log_l2 = get_score(xtrain,ytrain,xtest,ytest,SGDClassifier(loss="log",penalty="l2"))
## score  0.765306122449
## Apply log Loss with penalty l1
model_log_l1,cm_log_l1 = get_score(xtrain,ytrain,xtest,ytest,SGDClassifier(loss="log",penalty="l1"))
## score 0.758163265306
## Apply log Loss  with penalty l0
model_log_l0,cm_log_l0 = get_score(xtrain,ytrain,xtest,ytest,SGDClassifier(loss="log",penalty=None))
# 0.760204081633

############## Apply logistic regression
model_logit_l2, cm_logit_l2 = get_score(xtrain,ytrain,xtest,ytest,LogisticRegression(penalty="l2"))
#0.752040816327
model_logit_l1, cm_logit_l1 = get_score(xtrain,ytrain,xtest,ytest,LogisticRegression(penalty="l1"))
#0.765306122449
model_logit_l0, cm_logit_l0 = get_score(xtrain,ytrain,xtest,ytest,LogisticRegression())
#0.752040816327


####################            REGRESSION       ##################
#feature variables
xtrain = training[training.columns[:11]]
# output labels
ytrain = training["quality"]

xtest = test[test.columns[:11]]
ytest =  test["quality"]

##function to fit the model,compute r-square and return the score for the test data
def get_score_regression(xtrain,ytrain,xtest,ytest,regr):
    print(regr.fit(xtrain,ytrain))
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(xtest) - ytest) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(xtest, ytest))
    score = regr.score(xtest, ytest)     
    return (score)

### linear regression model
model_reg_l2 = get_score_regression(xtrain,ytrain,xtest,ytest,LinearRegression())
##r-sq 0.28

### SGD Regressor
model_sgdreg_l2 = get_score_regression(xtrain,ytrain,xtest,ytest,SGDRegressor(penalty="l2"))
##r-sq 0.17
model_sgdreg_l1 = get_score_regression(xtrain,ytrain,xtest,ytest,SGDRegressor(penalty="l1"))
##r-sq 0.17
model_sgdreg_l0 = get_score_regression(xtrain,ytrain,xtest,ytest,SGDRegressor())
##r-sq 0.17

###
model_ridge = get_score_regression(xtrain,ytrain,xtest,ytest,Ridge(alpha=0.5))
##r-sq 0.28
##
model_lasso = get_score_regression(xtrain,ytrain,xtest,ytest,Lasso(alpha=0.00001))
##r-sq 0.28
