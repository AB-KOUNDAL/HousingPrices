
"""
Note:
1)	Download the “PEP1.csv” using the link given in the Feature Engineering project problem statement
2)	For a detailed description of the dataset, you can download and refer to data_description.txt using the link given in the Feature Engineering project problem statement




Perform the following steps:
1.	Understand the dataset:
a.	Identify the shape of the dataset
b.	Identify variables with null values
c.	Identify variables with unique values
2.	Generate a separate dataset for numerical and categorical variables
3.	EDA of numerical variables:
a.	Missing value treatment
b.	Identify the skewness and distribution
c.	Identify significant variables using a correlation matrix
d.	Pair plot for distribution and density
4.	EDA of categorical variables
a.	Missing value treatment
b.	Count plot for bivariate analysis
c.	Identify significant variables using p-values and Chi-Square values
5.	Combine all the significant categorical and numerical variables
6.	Plot box plot for the new dataset to find the variables with outliers


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, probplot

#a.	Identify the shape of the dataset
housingDfOG = pd.read_csv("HousingData.csv",index_col=0)
#print(housingDfOG.head())
print("shape of the data set is  "+ str(housingDfOG.shape))


#b.	Identify variables with null values
housingDf= housingDfOG
housingDfNull = housingDfOG.isnull().any()
#nullData= housingDfNull.loc[:, columns_with_null]
#print(nullData)


#c.	Identify variables with unique values
uniqueDf=housingDfOG.nunique()
print(uniqueDf)


#2.	Generate a separate dataset for numerical and categorical variables
numerical_df = housingDfOG.select_dtypes(include=['int', 'float'])
categorical_df = housingDfOG.select_dtypes(include=['object'])

#Missing value treatment
housingNM= numerical_df.copy()
housingNM.isnull().any()
print("Housing Null metrics --- > ")
print(housingNM.isnull().any())
housingNM.fillna(value=0)

#Identify the skewness and distribution