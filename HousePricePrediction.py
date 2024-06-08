#Loading packages

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#Function definitions

#Defining Root Mean Square Logarithmic Error as Kaggle uses that as the score

def rmsle(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Lengths do not match!"
    return np.sqrt( np.mean( np.power( np.log1p(y_pred)-np.log1p(y_true), 2 ) ) )

#Function to bin levels in a categorical feature below a threshold percentage together into one level called "others"

def cut_levels(feature, threshold, new_level):
    value_counts = round(feature.value_counts()/sum(feature.value_counts())*100, 2)
    labels = value_counts.index[value_counts < threshold]
    feature[np.in1d(feature, labels)] = new_level



#Loading data

df_Original = pd.read_csv('train.csv')

#Saving and dropping the dependant variable
#Y_SalePrice = df_Original.SalePrice
#df_Original = df_Original.drop(["SalePrice"], axis = 1)

#Displaying first 5 rows
df_Original.head(5)

#Removing Id column

df_PreProcessed = df_Original.drop(["Id"], axis = 1)

df_PreProcessed.head(5)

#Selecting categorical variable

FreqCountDict = {}

for column in df_PreProcessed:
    FreqCount = df_PreProcessed[column].value_counts()
    if len(FreqCount) < 50:
        FreqCountDict[column] = FreqCount

#Removing two continous features classified as categorical
FeaturesToRemove = ['YrSold', 'PoolArea', 'MiscVal']

for key in FeaturesToRemove:
    if key in FreqCountDict:
        del FreqCountDict[key]
        
#All the categorical columns

print(FreqCountDict.keys(), "\n\n Number of categorical variables = ", len(FreqCountDict))


#Looking at categorical features with low distinguishing capability

lowDistFeatures = []

for column in FreqCountDict.keys():
    percentSeries = round(FreqCountDict[column]/sum(FreqCountDict[column])*100, 2)
    if (percentSeries > 97).any():
        lowDistFeatures.append(column)
        
print("Features with very little distinguishing capability : ", lowDistFeatures)

for key in lowDistFeatures:
    del FreqCountDict[key]

df_PreProcessed = df_PreProcessed.drop(lowDistFeatures, axis = 1)


#Looking at categorical variables which have the same level for 90% of the data

catFeatures = []

for column in FreqCountDict.keys():
    percentSeries = round(FreqCountDict[column]/sum(FreqCountDict[column])*100, 2)
    if (percentSeries > 80).any():
        catFeatures.append(column)
        cut_levels(df_PreProcessed[column], 5, 'Others')
        
        
print("Features which have the same level for 90% of the data : \n\n", catFeatures)

#for column in catFeatures:
    #print("\n\nColumn : ", column, "\n\n", round(df_PreProcessed[column].value_counts()/sum(df_PreProcessed[column].value_counts())*100, 2))


#Treating NAs in categorical variables before OneHotEncoding
#We first get the list of all categorical features with NAs as the data description has NA as a value in
#most categorical features

CategoricalFeatures_WithNAs = []

for cat_feature in list(FreqCountDict.keys()):
    if df_PreProcessed[cat_feature].isnull().any():
        CategoricalFeatures_WithNAs.append(cat_feature)

CategoricalFeatures_WithNAs

#Looking at the "data_description.txt" file, only MasVnrType & Electrical features are truly NAs. NAs in the rest
#of the features are explained in data description file

print("Number of NAs in MasVnrType : ", df_PreProcessed['MasVnrType'].isnull().sum())
print("\nNumber of NAs in Electrical : ", df_PreProcessed['Electrical'].isnull().sum())

#Replacing NAs with relevant category

df_PreProcessed['MasVnrType'] = df_PreProcessed['MasVnrType'].fillna("None")
df_PreProcessed['Electrical'] = df_PreProcessed['Electrical'].fillna("SBrkr")

#Setting all other categorical feature NAs to "Absent"
df_PreProcessed[['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual',
                 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']] = df_PreProcessed[['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual',
                 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']].fillna("Absent")



#Treating NAs of numerical variables

#Identifying the numerical columns with NAs
print(df_PreProcessed.dtypes[df_PreProcessed.isnull().any().values])

print("\nNumber of NAs in LotFrontage : ", df_PreProcessed['LotFrontage'].isnull().sum())
print("\nNumber of NAs in MasVnrArea : ", df_PreProcessed['MasVnrArea'].isnull().sum())
print("\nNumber of NAs in GarageYrBlt : ", df_PreProcessed['GarageYrBlt'].isnull().sum())

#Replacing LotFrontage with mean value
df_PreProcessed['LotFrontage'] = df_PreProcessed['LotFrontage'].fillna(np.nanmean(df_PreProcessed.LotFrontage.values))

#Replacing LotFrontage with mean value
df_PreProcessed['MasVnrArea'] = df_PreProcessed['MasVnrArea'].fillna(np.nanmean(df_PreProcessed['MasVnrArea']))

#Replacing NAs in GarageYrBlt to 0
df_PreProcessed['GarageYrBlt'] = df_PreProcessed['GarageYrBlt'].fillna(0)


#Calculating House & Garage Age

df_PreProcessed["GarageAge"] = df_PreProcessed.YrSold - df_PreProcessed.GarageYrBlt

#Treating all NA cases in GarageYrBlt
df_PreProcessed.loc[df_PreProcessed.GarageAge > 500, "GarageAge"] = 0

df_PreProcessed["HouseAge"] = df_PreProcessed.YrSold - df_PreProcessed.YearBuilt

#Creating a categorical feature that indicates whether a house was sold brand new
#df_PreProcessed["BrandNewHouse"] = 0
#df_PreProcessed.loc[df_PreProcessed.HouseAge == 0, "BrandNewHouse"] = 1

df_PreProcessed["RemodellingAge"] = df_PreProcessed.YrSold - df_PreProcessed.YearRemodAdd

#Removing YearBuilt, YrSold, GarageYrBlt

df_PreProcessed = df_PreProcessed.drop(["GarageYrBlt", "YearBuilt", "YrSold", "YearRemodAdd"], axis = 1)


# In[170]:


#Scaling continous features
#Saving continous features

#continousFeatures = [i for i in df_PreProcessed.columns if i not in FreqCountDict.keys()]

#sc = StandardScaler()  
#df_PreProcessed.loc[:,continousFeatures] = sc.fit_transform(df_PreProcessed.loc[:,continousFeatures])


# In[171]:


#OneHotEncoding categorical features

for cat_feature in list(FreqCountDict.keys()):
    df_PreProcessed = df_PreProcessed.join(pd.get_dummies(df_Original[cat_feature], prefix = cat_feature, prefix_sep= "_"))
    df_PreProcessed = df_PreProcessed.drop([cat_feature], axis = 1)

print("Length of original DataFrame : ", len(df_Original.columns),
      "\n\nLength of DataFrame after OneHotEncoding : ", len(df_PreProcessed.columns))


# In[172]:


#Running linear regression to set baseline

#Splitting into test and train datasets on a 70/30 split
train = df_PreProcessed[:round(0.7*len(df_PreProcessed))]
test = df_PreProcessed[round(0.7*len(df_PreProcessed)) + 1:len(df_PreProcessed)]

#Removing dependent variable from test and train set
Y = train.SalePrice
X = train.drop(["SalePrice"], axis = 1)

Y_Truth = test.SalePrice
test = test.drop(["SalePrice"], axis = 1)

#Training the regression model
LinReg = LinearRegression()
LinReg.fit(X=X, y=Y)

predY = LinReg.predict(X=test)

print("\n R^2 = ", r2_score(y_pred=predY, y_true= Y_Truth))

print("\n RMSLE = ", rmsle(y_pred=predY, y_true= Y_Truth))


# In[173]:


#Running RandomForest Classifier

#Splitting into test and train datasets on a 70/30 split
train = df_PreProcessed[:round(0.7*len(df_PreProcessed))]
test = df_PreProcessed[round(0.7*len(df_PreProcessed)) + 1:len(df_PreProcessed)]

#Removing dependent variable from test and train set
Y = train.SalePrice
X = train.drop(["SalePrice"], axis = 1)

Y_Truth = test.SalePrice
test = test.drop(["SalePrice"], axis = 1)

regressor = RandomForestRegressor(n_estimators = 30, random_state = 0)  
regressor.fit(X, Y)  
predY = regressor.predict(test)

print("\n R^2 = ", r2_score(y_pred = predY, y_true = Y_Truth))

print("\n RMSLE = ", rmsle(y_pred = predY, y_true = Y_Truth))


