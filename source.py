#!/usr/bin/env python
# coding: utf-8

# # An Analysis of the Cybercrime landscape in an AI World📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# ### How is AI Reshaping CyberCrime?

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# ### Since AI has become more readily available for the masses, has there been a distinct rise in cybercrime? Whether frequency or sophistication?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# ### There has been a distinct rise in cybercrime sophistication. But it is a two sided coin, AI is used by threatactors but it is also being used by cybersecurity software and agents.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# ### Global Cybersecurity Threats (2015-2024) https://www.kaggle.com/datasets/atharvasoundankar/global-cybersecurity-threats-2015-2024
# ### Known Exploited Vulneratiblities Catalog https://www.cisa.gov/known-exploited-vulnerabilities-catalog
# ### Global Dataset of Cyber Incidents https://zenodo.org/records/14965395
# ### NIST National Vulnerabiltiy Database https://nvd.nist.gov/vuln/data-feeds 
# 
# ### I'm going to use the datasets to see what trends or interesting statistics can be gleamed to prove or disprove my theory that AI has contributed greatly to the increase in cybercrime.
# 
# ## Todo
# ### Need to find an ai adoption dataset, so i can compare the usage of ai to the statistics of the cybercrimes

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# ### This analysis will employ a methodology to examine the relationship between AI availability and the increase in cybercrime. By leveraging multiple datasets spanning the period from 2017-2025, I will conduct trend analysis, correlation studies, and statistical evaluation to test the hypothesis that AI technologies have significantly contributed to the increase in cybercrime sophistication and incidents.

# # Explorartory Data Analysis
# 
# ## Visualizations
# 
# ##### Cyber Attacks by Year (2015-2024)
# * Shows cyber attacks by year. Shows if there has been an increase of decrease year over year
# ##### Global Threats - Attacks by Year and Industry (Stacked Bar):
# * Shows attack distribution across industries from 2015-2024. Reveals consistent attack volumes with Healthcare and IT as frequent targets
# ##### Distribution of Attack Types (2015-2024) - Pie Chart:
# * Even distribution across attack types (DDoS, Malware, Phishing, etc.) shows organizations must defend against diverse threats rather than focusing on single vectors
# ##### Cyber Incident Attributions by Year - Line Chart:
# * Shows yearly cyber incident attributions. Peaks may correspond to major geopolitical events or increased threat intelligence sharing
# ##### Top 10 Initiator Countries (Attackers) - Horizontal Bar Chart:
# * Identifies countries most frequently accused of attacks, revealing the geopolitical landscape of cyber conflict and offensive capabilities
# ##### Distribution of CVSS Scores for Known Exploited Vulnerabilities - Histogram:
# * Concentration in 7-10 range confirms actively exploited vulnerabilities are high severity, validating CISA's prioritization for patching efforts
# ##### Average CVSS Score of Known Exploited Vulnerabilities by Year - Line Chart:
# * Stable average around 7-8 shows attackers consistently target medium-to-high severity vulnerabilities with no improvement trend over time
# 
# ## Data Cleansing Process
# ##### EUREPOC Attribution Dataset:
# * Removed 1,623 rows with null attribution_year values since the analysis focused exclusively on year based trends. The null values for attribution_month and attribution_day were retained as they were not relevant to our temporal analysis. This cleaning step reduced the dataset size but ensured all records had the essential information needed for year-over-year comparisons and trend analysis.
# ##### Known Exploited Vulnerabilities (KEV) Dataset:
# * The KEV dataset required merging with the National Vulnerability Database (NVD) to obtain CVSS scores, which were not included in the original CISA catalog. I loaded NVD JSON files (2015-2025), extracted CVSS v3.1, v3.0, and v2.0 scores, and performed a left join on cveID to preserve all KEV records. A combined cvss_score column was created by prioritizing the most recent CVSS version available (v3.1 > v3.0 > v2.0), enabling severity analysis of actively exploited vulnerabilities.
# ##### Global Cybersecurity Threats Dataset:
# * This dataset contained minimal null values and required no significant cleaning. I verified data types for temporal fields (Year) and numeric fields (Financial Loss, Number of Affected Users) to ensure proper aggregation and visualization. The dataset was already pretty well structured with consistent categorical values across attack types, industries, countries, and defense mechanisms.
# 
# ## Machine Learning
# ### *<font color = 'red'>We haven't covered machine learning yet but I included since it's in the grading criteria</font>*
# #### What types of machine learning could I use in your project?
# ##### Sources: https://builtin.com/data-science/supervised-machine-learning-classification
# * For this project, supervised learning classification would be most applicable for predicting attack severity levels, attack types, and attacker categories based on historical patterns. Regression models could forecast financial losses and incident resolution times. Unsupervised learning through clustering could identify emerging threat patterns and group similar vulnerabilities or attack campaigns. Time series forecasting would be valuable for predicting future attack trends and vulnerability discovery rates across industries and regions.
# #### What issues do I see in making that happen?
# * The primary issues include data quality problems with missing values and incomplete records, imbalance where certain attack types or threat actors dominate the dataset, and the challenge of obtaining negative examples (vulnerabilities not exploited).
# #### What challenges will I potentially face?
# ##### Sources: https://research.aimultiple.com/model-drift/
# * The main challenges include rapid evolution of cyber threats making historical data quickly outdated. Small datasets and imbalanced classes (rare attack types) lead to overfitting and poor predictions. Attribution data lacks objective ground truth, creating accuracy uncertainty. Models may learn spurious correlations rather than actual causal relationships. 
# 

# In[28]:


# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load datasets
eurepoc_attribution_data = pd.read_csv('./data/eurepoc_attribution_dataset_1.3.csv')
global_security_threats = pd.read_csv('./data/Global_Cybersecurity_Threats_2015_2024.csv')
known_exploited_vulnerabilities = pd.read_csv('./data/known_exploited_vulnerabilities.csv')



# In[3]:


# Display basic statistics of the datasets
display(eurepoc_attribution_data.describe())
display(global_security_threats.describe())
display(known_exploited_vulnerabilities.describe())


# In[4]:


# Display information about the datasets
display(eurepoc_attribution_data.info())
display(global_security_threats.info())
display(known_exploited_vulnerabilities.info())


# In[5]:


# Display first few rows of the datasets
display(eurepoc_attribution_data.head())
display(global_security_threats.head())
display(known_exploited_vulnerabilities.head())


# In[6]:


# Display null value counts for each dataset
display(eurepoc_attribution_data.isnull().sum())
display(global_security_threats.isnull().sum())
display(known_exploited_vulnerabilities.isnull().sum())


# In[7]:


# Europoc attribution data specific null analysis
display(f'Original dataset size: {len(eurepoc_attribution_data)}')
display(f'Rows with null attribution_year: {eurepoc_attribution_data["attribution_year"].isnull().sum()}')

# Remove rows where attribution_year is null
eurepoc_attribution_data = eurepoc_attribution_data[eurepoc_attribution_data['attribution_year'].notna()]
display(f'Dataset size after removing null years: {len(eurepoc_attribution_data)}')

# Verify no null values remain in attribution_year
display(known_exploited_vulnerabilities.isnull().sum())

# Preview of the cleaned data
display(eurepoc_attribution_data.head())


# In[8]:


# Global security threats data specific analysis
# import matplotlib
import matplotlib.pyplot as plt

# Variables for analysis
attacks_by_year = global_security_threats['Year'].value_counts().sort_index()
industry_year = pd.crosstab(global_security_threats['Year'], global_security_threats['Target Industry'])
attack_counts = global_security_threats['Attack Type'].value_counts()

# Count of cyber attacks by year
attacks_by_year_df = attacks_by_year.reset_index()
attacks_by_year_df.columns = ['Year', 'Number of Attacks']
display(attacks_by_year_df)

# Bar plot of cyber attacks by year
display('Bar Plot of Cyber Attacks by Year (2015-2024), trying to determine any trends')
plt.figure(figsize=(12, 6))
attacks_by_year.plot(kind='bar', color='steelblue')
plt.title('Cyber Attacks by Year (2015-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Bar plot of cyber attacks by year and target industry
display('Bar Plot of Cyber Attacks by Year and Target Industry, again trying to determine any trends')
fig, ax = plt.subplots(figsize=(14, 8))
industry_year.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
plt.title('Cyber Attacks by Year and Target Industry (2015-2024)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Attacks', fontsize=12)
plt.legend(title='Target Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

display('Pie Chart of Attack Type Distribution to understand the prevalence of different attack types')
# Pie chart of attack type distribution
fig, ax = plt.subplots(figsize=(10, 8))
ax.pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Attack Types (2015-2024)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# In[9]:


# Europoc attribution data analysis

# Variables for analysis
attacks_by_year = eurepoc_attribution_data['attribution_year'].value_counts().sort_index()
top_origin = eurepoc_attribution_data['initiator_country'].value_counts().head(10)

# Attribution Trends Over Time
display('Line Chart of Cyber Incident Attributions by Year to identify trends in attributions over time')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(attacks_by_year.index, attacks_by_year.values, marker='o', linewidth=2, color='darkblue')
plt.title('Cyber Incident Attributions by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Attributions', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

display('Horizontal Bar Chart of Top 10 Countries Where Attack Originated to see which countries are most frequently attributed as sources of cyber attacks')
fig, ax = plt.subplots(figsize=(10, 6))
# Some names are too long, so we wrap them for better display
import textwrap
top_origin.index = ['\n'.join(textwrap.wrap(str(name), width=20)) for name in top_origin.index]

top_origin.plot(kind='barh', ax=ax, color='crimson')
plt.title('Top 10 Countries Where Attack Origniated', fontsize=16, fontweight='bold')
plt.xlabel('Number of Attributed Attacks', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()



# In[10]:


# Add nvd_cve_dateset, the known exploited vulnerabilities dataset did not contain CVE scores
nvd_full = pd.read_csv('./data/nvd_cve_data_2015_2025.csv')

# Select columns
nvd_selected = nvd_full[['cveID', 'cvss_v31_score', 'cvss_v30_score', 'cvss_v2_score', 'description']]

display('NVD Selected Columns:')
display(nvd_selected.head())

# Merge the datasets
kev_with_scores = pd.merge(known_exploited_vulnerabilities, nvd_selected, on='cveID', how='left')

# Create a combined CVSS score column
kev_with_scores['cvss_score'] = kev_with_scores['cvss_v31_score'].fillna(
    kev_with_scores['cvss_v30_score']
).fillna(kev_with_scores['cvss_v2_score'])

# Display results
display('\nMERGED DATASET')
display(f'Total KEV entries: {len(kev_with_scores)}')
display(f'KEV entries with CVSS scores: {kev_with_scores["cvss_score"].notna().sum()}')

display('\nColumns in merged dataset:')
display(kev_with_scores.columns.tolist())

display('\nSample merged data:')
display(kev_with_scores[['cveID', 'vendorProject', 'product', 'cvss_v31_score', 'cvss_v30_score', 'cvss_v2_score', 'cvss_score']].head(10))

#
kev_with_scores.to_csv('./data/kev_with_cvss_scores.csv', index=False)
display('\n Merged dataset saved!')


# In[ ]:


# Intial analysis on new dataset
merged_data = pd.read_csv('./data/kev_with_cvss_scores.csv')

display('Combined KEV and CVSS Scores Dataset Overview, again trying to understand the data better')
# Distribution of CVSS scores
fig, ax = plt.subplots(figsize=(12, 6))
kev_with_scores['cvss_score'].dropna().hist(bins=30, ax=ax, color='darkred', edgecolor='black')
plt.title('Distribution of CVSS Scores for Known Exploited Vulnerabilities', fontsize=16, fontweight='bold')
plt.xlabel('CVSS Score', fontsize=12)
plt.ylabel('Number of Vulnerabilities', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

kev_with_scores['dateAdded'] = pd.to_datetime(kev_with_scores['dateAdded'])

# Extract year
kev_with_scores['year_added'] = kev_with_scores['dateAdded'].dt.year

avg_cvss_by_year = kev_with_scores.groupby('year_added')['cvss_score'].mean()

display('Average CVSS Score by Year Added to KEV Catalog, to see if there are trends in severity over time')
# Create line chart for yearly average score
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(avg_cvss_by_year.index, avg_cvss_by_year.values, marker='o', linewidth=2, color='darkred', markersize=8)
plt.title('Average CVSS Score of Known Exploited Vulnerabilities by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year Added to Catalog', fontsize=12)
plt.ylabel('Average CVSS Score', fontsize=12)
plt.ylim(8, 10) 
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display the data
display('Average CVSS Score by Year:')
display(avg_cvss_by_year.round(2))

# Additional stats by year
display('\nNumber of Vulnerabilities Added Each Year:')
display(kev_with_scores.groupby('year_added').size())


# ## Machine Learning Implmentation Process 
# (Ask, Prepare, Process, Analyze, Evaluate, Share)
# 
# * This includes:
#     * EDA process that allows for identifying issues
#     * Splitting the dataset into training and test sets
#     * Data cleaning process using sci-kit learn pipelines
#         * Data imputation
#         * Data Scaling and Normalization
#         * Handling of Categorical Data
#     * Testing multiple algorithms and models
#     * Evaluating the different models and choosing one.
# 

# ### EDA Process for Identifying Issues

# In[21]:


display('EDA Process that allows for identifying issues')

# Check the dataset we'll use for ML
display('Dataset Overview:')
display('Total records in kev_with_scores:', len(kev_with_scores))
display('Columns:', kev_with_scores.columns.tolist())

# Check for missing values in key columns
display('Missing Values in Key Columns:')
ml_columns = ['year_added', 'cvss_score']
display(kev_with_scores[ml_columns].isnull().sum())

# Check data types
display('\nData Types:')
display(kev_with_scores[ml_columns].dtypes)

# Statistical summary
display('Statistical Summary:')
display(kev_with_scores[ml_columns].describe())

# Check for any infinite values
display('Check for Infinite Values:')
display('Infinite values in year_added:', np.isinf(kev_with_scores["year_added"]).sum())
display('Infinite values in cvss_score:', np.isinf(kev_with_scores["cvss_score"].dropna()).sum())

# Distribution visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Year distribution
kev_with_scores['year_added'].hist(bins=20, ax=ax1, color='steelblue', edgecolor='black')
ax1.set_xlabel('Year Added', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Year Added', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# CVSS Score distribution
kev_with_scores['cvss_score'].dropna().hist(bins=20, ax=ax2, color='darkred', edgecolor='black')
ax2.set_xlabel('CVSS Score', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of CVSS Scores', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# ### Splitting the Dataset into Training and Test Sets

# In[ ]:


# Split the dataset into training and test sets
display('Splitting dataset into training and test sets')

# Prepare the data - remove rows with missing CVSS scores
ml_data = kev_with_scores[['year_added', 'cvss_score']].dropna()
display('Dataset after removing nulls:', len(ml_data), 'records')

# Separate features (X) and target (y)
X = ml_data[['year_added']]
y = ml_data['cvss_score']

display('Features (X) shape:', X.shape)
display('Target (y) shape:', y.shape)

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

display('Training set size:', len(X_train), 'records')
display('Test set size:', len(X_test), 'records')


# ### Data Cleaning Process Using Scikit-Learn Pipelines
# - **Data Imputation**
# - **Data Scaling and Normalization**
# - **Handling of Categorical Data**

# In[27]:


# Create preprocessing pipelines for data cleaning
display('Data cleaning process using scikit-learn pipelines')

# Create numerical pipeline: imputation + scaling
display('Creating Numerical Pipeline:')
display('SimpleImputer - Fill missing values with median')
display('StandardScaler - Standardize features (mean=0, std=1)')

num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

# Define numerical features
num_features = ['year_added']
display('Numerical features:', num_features)

# Create full preprocessing pipeline using ColumnTransformer
display('Creating Full Pipeline with ColumnTransformer')
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features)
])

# Fit and transform the training data
display('Applying pipeline to training data')
X_train_prepared = full_pipeline.fit_transform(X_train)
display('Data shape:', X_train_prepared.shape)

# Transform the test data (using already fitted pipeline)
display('Applying pipeline to test data')
X_test_prepared = full_pipeline.transform(X_test)
display('Data shape:', X_test_prepared.shape)


# ### Testing Multiple Algorithms and Models
# - **Linear Regression**
# - **Polynomial Regression**

# In[33]:


# Train and test multiple regression models
display('Testing multiple algorithms and models')

# LINEAR REGRESSION MODEL
display('Linear Regression Model')
display('------------------------')
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

# Evaluate on training set
train_predictions = lin_reg.predict(X_train_prepared)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
display('Training RMSE:', train_rmse)

# Evaluate on test set
test_predictions = lin_reg.predict(X_test_prepared)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
display('Test RMSE:', test_rmse)

# POLYNOMIAL REGRESSION MODEL
display('Polynomial Regression Model')
display('----------------------------')
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_reg = make_pipeline(poly_features, LinearRegression())
poly_reg.fit(X_train_prepared, y_train)

# Evaluate on training set
poly_train_predictions = poly_reg.predict(X_train_prepared)
poly_train_mse = mean_squared_error(y_train, poly_train_predictions)
poly_train_rmse = np.sqrt(poly_train_mse)
display('Training RMSE:', poly_train_rmse)

# Evaluate on test set
poly_test_predictions = poly_reg.predict(X_test_prepared)
poly_test_mse = mean_squared_error(y_test, poly_test_predictions)
poly_test_rmse = np.sqrt(poly_test_mse)
display('Test RMSE:', poly_test_rmse)


# ### Evaluating the Models and Choosing One
# Compare model performance and select the best model based on test set RMSE.

# In[30]:


# Evaluate and compare the models to choose the best one
display('Evaluating the models and choosing one')
display('-' * 50)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Polynomial Regression'],
    'Training RMSE': [train_rmse, poly_train_rmse],
    'Test RMSE': [test_rmse, poly_test_rmse]
})

display('Model Performance Comparison:')
display(comparison_df)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['Training RMSE'], width, label='Training RMSE', color='steelblue')
bars2 = ax.bar(x + width/2, comparison_df['Test RMSE'], width, label='Test RMSE', color='coral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Model Performance', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Comparison summary
display('-' * 50)
display('MODEL COMPARISON SUMMARY')
display('-' * 50)

# Calculate differences
rmse_difference = abs(test_rmse - poly_test_rmse)
display('Linear Regression Test RMSE:', test_rmse)
display('Polynomial Regression Test RMSE:', poly_test_rmse)
display('Difference:', rmse_difference)


# ### Results
# 
# Based on the output the Polynomial Regression model has a lower test RMSE compared to the Linear Regression model. This indicates that the Polynomial Regression model captures the data patterns better and is likely to provide more accurate predictions on the data. I believe the Polynomial Regression model is best for my final analysis.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# * https://medium.com/latinxinai/evaluation-metrics-for-regression-models-03f2143ecec2
# * https://realpython.com/linear-regression-in-python/ 
# * https://realpython.com/how-to-use-numpy-arange/ 

# In[31]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

