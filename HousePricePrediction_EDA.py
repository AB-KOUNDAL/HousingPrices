"""Project Details - Project Statement:
While searching for the dream house, the buyer looks at various factors, not just at the height of the basement ceiling or the proximity to an east-west railroad.
Using the dataset, find the factors that influence price negotiations while buying a house.
There are 79 explanatory variables describing every aspect of residential homes in Ames, Iowa.
"""



""" Task 1:Understand the dataset:
            a.	Identify the shape of the dataset
            b.	Identify variables with null values
            c.	Identify variables with unique values
"""

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, stats

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#import CSV
housing_df = pd.read_csv("HousingData.csv",index_col=0)

# 	Identify the shape of the dataset
print("housing_df number of columns -- ")
print( housing_df.shape)

# Identify variables with null values
print("housing_df number of null columns total -- ")
print( housing_df.isnull().sum())
print( housing_df.isnull().sum().sum())

print("housing_df number of null unique total -- ")
print( housing_df.nunique().sum())
print( housing_df.nunique())


#Task 2.	Generate a separate dataset for numerical and categorical variables

housing_num_df = housing_df.select_dtypes(include=[float,int])
print("housing_df number of numerical columns -- ")
print(housing_num_df.shape)
print("housing_df number of numerical null columns -- ")
print( housing_num_df.isnull().sum().sum())
print("housing_df number of numerical unique columns -- ")
print( housing_num_df.nunique().sum())



housing_cat_df = housing_df.select_dtypes(exclude=[float,int])
print("housing_df number of Categorical columns -- ")
print(housing_cat_df.shape)
print("housing_df number of Categorical null columns -- ")
print( housing_cat_df.isnull().sum().sum())
print("housing_df number of Categorical unique columns -- ")
print( housing_cat_df.nunique().sum())
#print( housing_cat_df.nunique())


# Identify Null values
print("housing_df number of Numerical columns Null Values-- ")
print(housing_num_df.isnull().any())

print("housing_df number of Categorical columns Null Values-- ")
print(housing_cat_df.isnull().any())


"""Task 3.	EDA of numerical variables:
            a.	Missing value treatment
            b.	Identify the skewness and distribution
            c.	Identify significant variables using a correlation matrix 
            d.	Pair plot for distribution and density
        
"""

# Identify the skewness and distribution
skewness = housing_num_df.skew()
print("skewness")
print(skewness)
housing_numC_df= housing_num_df.apply(lambda col: col.fillna(col.median()))

#	Identify significant variables using a correlation matrix
correlation_matrix= housing_num_df.corr()
print("Correlation")
print(correlation_matrix["SalePrice"].sort_values(ascending=False))

significant_correlations = correlation_matrix[(correlation_matrix > 0.5) | (correlation_matrix < -0.5)]
print("\nSignificant Correlations (|r| > 0.5):\n", significant_correlations)


Significant_matrix_col= housing_num_df[["SalePrice","OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF"]]
print("\nSignificant Correlations (|r| > 0.5):\n", significant_correlations)
print(Significant_matrix_col)

pair_plot= sns.pairplot(Significant_matrix_col, y_vars=["SalePrice"])
plt.show()


"""4.	EDA of categorical variables
            a.	Missing value treatment
            b.	Count plot for bivariate analysis
            c.	Identify significant variables using p-values and Chi-Square values
"""

print(housing_cat_df.head())
sns.countplot( x="SaleCondition",hue= "MSZoning", data= housing_cat_df)
plt.show()


print(housing_cat_df.head())
print("housing_df number of Categorical columns -- ")
print(housing_cat_df.shape)



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



housing_catC_df   =  housing_cat_df.apply(lambda col: col.fillna(col.mode()[0]))
print("categorical_vars_filled")

housing_catC_df['SalePrice'] = pd.qcut(housing_numC_df['SalePrice'], q=4, labels=False)
print(housing_catC_df.head())


chi_square_results = {}
for col in housing_catC_df:
    if col != 'SalePrice':
        contingency_table = pd.crosstab(housing_catC_df[col],housing_catC_df['SalePrice'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi_square_results[col] = {'chi2': chi2, 'p-value': p}

        # Convert the results to a DataFrame for better visualization
chi_square_results_df = pd.DataFrame(chi_square_results).T
chi_square_results_df.sort_values(by='p-value', inplace=True)
#housing_Cat_Sig_df= chi_square_results_df[chi_square_results_df['p-value'] >0.05 | chi_square_results_df['p-value'] <-0.05]
# Filter significant variables based on the threshold
# Sorting the DataFrame
#chi_square_results_df.sort_values(by='P-Value', inplace=True)

# Filtering Significant Variables
housing_Cat_Sig_df =  chi_square_results_df[chi_square_results_df['p-value'] < 0.05].index.tolist()
print(housing_Cat_Sig_df)

significant_numerical_vars = ["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF"]

housing_significant_vars = housing_numC_df[significant_numerical_vars].copy()
housing_significant_vars = housing_significant_vars.join(housing_catC_df[housing_Cat_Sig_df])


housing_significant_vars.to_csv('Housing_significant_vars.csv', index=False)
print(housing_significant_vars.head())



file_path = 'Housing_significant_vars.csv'  # Update with the actual path to your CSV file
combined_significant_vars = pd.read_csv(file_path)

# Plot box plots for numerical variables to identify outliers

numerical_columns = combined_significant_vars.select_dtypes(include=['int64', 'float64']).columns

# Close any existing plots
def plot_box_plots(df, columns, plots_per_fig=6):
    """
    Plots box plots for the specified columns in the DataFrame.

    :param df: DataFrame containing the data
    :param columns: List of column names to plot
    :param plots_per_fig: Number of plots per figure
    """
    # Close any existing plots
    plt.close('all')

    # Calculate the number of figures needed
    num_figures = (len(columns) + plots_per_fig - 1) // plots_per_fig

    for fig_num in range(num_figures):
        plt.figure(figsize=(20, 15))
        for i in range(plots_per_fig):
            col_idx = fig_num * plots_per_fig + i
            if col_idx < len(columns):
                col = columns[col_idx]
                plt.subplot(plots_per_fig // 2, 2, i + 1)
                sns.boxplot(data=df, y=col)
                plt.title(f'Box Plot of {col}')
        plt.tight_layout()
        plt.pause(3)



# Load the combined significant variables dataset
file_path = 'Housing_significant_vars.csv'  # Update with the actual path to your CSV file
combined_significant_vars = pd.read_csv(file_path)

# Get the list of numerical columns
numerical_columns = combined_significant_vars.select_dtypes(include=['int64', 'float64']).columns

# Plot box plots for numerical variables to identify outliers
plot_box_plots(combined_significant_vars, numerical_columns, plots_per_fig=6)