#!/usr/bin/env python
# coding: utf-8

# # An Analysis of the Cybercrime landscape in an AI World
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# #### <span style = 'color:green'>Understanding whether AI availability has contributed to rising cybercrime rates and attack sophistication.</span>

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# #### <span style = 'color:green'>Is there a measurable correlation between AI accessibility and changes in cybercrime trends?</span>

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# #### <span style = 'color: green'>AI availability has likely contributed to an increase in cybercrime volume and sophistication, as these tools lower technical barriers for attackers and eliminate traditional red flags such as misspellings in phishing emails.</span>

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# * **Cyber Events Database:** The Cyber Events Database consists of publicly available information on cyber events
#     * https://cissm.umd.edu/research-impact/publications/cyber-events-database-home
# * **Global Cybersecurity Threats (2015-2024):** <span style = 'color:red; font-style: italic'>This was determined to be generated data and will not be used extensively for analysis</span> A comprehensive dataset tracking cybersecurity incidents, attack vectors, threat 
#     * https://www.kaggle.com/datasets/atharvasoundankar/global-cybersecurity-threats-2015-2024
# * **AI incident database:** Documenting the times when things go wrong with AI solutions
#     * https://www.kaggle.com/datasets/konradb/ai-incident-database
# * **Epoch AI:** Comprehensive database of over 3200 models tracks key factors driving machine learning progress
#     * https://epoch.ai/data/ai-models 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# <br>
# #### <span style = 'color:green'>ChatGPT's public release was November of 2022 we will use that as a data point to compare metrics before and after wide spread AI availability. The Global Cybersecurity Threats dataset provides volume and attack type trends, the Cyber Events Database shows incident level context on motives and actors, and the AI Incident Database identifies specific cases of AI use allowing us to try and correlate AI availability with changes in cybercrime patterns.</span>

# # Imports and Data Loading

# In[59]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


# In[60]:


# Load datasets

# AI Incident Database
ai_incidents = pd.read_csv('data/AI_incidents_database.csv')

# Global Cybersecurity Threats
cyber_threats = pd.read_csv('data/Global_Cybersecurity_Threats_2015_2024.csv')

# CISSM Cyber Events Database
cyber_events = pd.read_csv('data/CISSM_Cyber_Events_Database_2014_Oct_2025.csv')

# Epoch AI Model Tracking
epoch_ai_models = pd.read_csv('data/epoch_ai_models.csv')


# ---
# # Initial Data Analysis

# #### View first few rows of each dataset

# In[61]:


# Check first few rows of each dataset

display("AI Incidents Database")
display(ai_incidents.head())

display("Global Cybersecurity Threats")
display(cyber_threats.head())

display("CISSM Cyber Events Database")
display(cyber_events.head())

display("Epoch AI Model Tracking")
display(epoch_ai_models.head())


# #### View shape and column info of our datasets

# In[62]:


# Get info about each dataset
display("Dataset Shapes:")
display("AI Incidents:", ai_incidents.shape)
display("Global Cybersecurity Threats:", cyber_threats.shape)
display("CISSM Cyber Events:", cyber_events.shape)
display("Epoch AI Model Tracking:")
display(epoch_ai_models.shape)
display("AI Incidents Database Info")
display(ai_incidents.info())
display("Global Cybersecurity Threats Info")
display(cyber_threats.info())
display("CISSM Cyber Events Database Info")
display(cyber_events.info())
display("Epoch AI Model Tracking Info")
display(epoch_ai_models.info())


# #### Check for null or missing values with our datasets

# In[63]:


# Count null or missing values
display("AI Incidents Database Missing Values")
display(ai_incidents.isnull().sum())
display("Global Cybersecurity Threats Missing Values")
display(cyber_threats.isnull().sum())
display("CISSM Cyber Events Database Missing Values")
display(cyber_events.isnull().sum())
display("Epoch AI Model Tracking Missing Values")
display(epoch_ai_models.isnull().sum())


# ---
# # Data Preparation and Cleaning

# In this section, we prepare each dataset for analysis by performing the following operations:
# 
# | Operation | Reason |
# |-----------|---------|
# | **DateTime conversion** | Enable temporal filtering and time-series analysis |
# | **Year/month extraction** | Allow aggregation by time periods for trend analysis |
# | **Column selection** | Retain only features relevant to our research question |
# | **Duplicate removal** | Ensure data quality and accurate event counts |
# | **Filtering (Epoch AI)** | Focus on publicly accessible language models, which are most relevant to AI-enabled cybercrime|
# | **Column renaming** | Create consistency across datasets for easier merging |

# In[64]:


# Data cleaning and preprocessing

# AI Incidents
# Convert date string to datetime and extract year for time based analysis, we need this to filter by month and year later
ai_incidents['date'] = pd.to_datetime(ai_incidents['date'])
ai_incidents['year'] = ai_incidents['date'].dt.year
ai_incidents['month'] = ai_incidents['date'].dt.month

# Select only columns relevant for analysis
ai_incidents_clean = ai_incidents[[
    'incident_id', 'date', 'year', 'month', 'title', 'description',
    'Alleged deployer of AI system', 'Alleged developer of AI system'
]].copy()

# Global Cybersecurity Threats
# Select relevant columns for trend and impact analysis
cyber_threats_clean = cyber_threats[[
    'Year', 'Country', 'Attack Type', 'Target Industry',
    'Financial Loss (in Million $)', 'Number of Affected Users',
    'Attack Source', 'Security Vulnerability Type'
]].copy()

# CISSM Cyber Events
# Convert event_date to datetime for time based filtering, we need this to filter by month and year later
cyber_events['event_date'] = pd.to_datetime(cyber_events['event_date'])

# Select columns relevant to motive, actor, and event classification
cyber_events_clean = cyber_events[[
    'event_date', 'year', 'month', 'actor_type', 'motive',
    'event_type', 'event_subtype', 'industry', 'country', 'description'
]].copy()
# Remove duplicate records from cyber events
cyber_events_clean = cyber_events_clean.drop_duplicates()

# Epoch AI Models
# Convert publication date to datetime
epoch_ai_models['Publication date'] = pd.to_datetime(epoch_ai_models['Publication date'], errors='coerce')
epoch_ai_models['year'] = epoch_ai_models['Publication date'].dt.year
epoch_ai_models['month'] = epoch_ai_models['Publication date'].dt.month

# Filter to publicly accessible models only (exclude internal/unreleased)
epoch_public_access_types = [
    'API access', 
    'Open weights (unrestricted)', 
    'Open weights (restricted use)', 
    'Hosted access (no API)'
]
epoch_public = epoch_ai_models[
    epoch_ai_models['Model accessibility'].isin(epoch_public_access_types)
].copy()

# Filter to language models (most relevant for AI-enabled cybercrime like phishing)
epoch_language = epoch_public[
    epoch_public['Domain'].str.contains('Language', case=False, na=False)
].copy()

# Select relevant columns for analysis
epoch_ai_clean = epoch_language[[
    'Model', 'Publication date', 'year', 'month',
    'Domain', 'Task', 'Organization', 'Country (of organization)',
    'Model accessibility', 'Parameters', 'Training compute (FLOP)'
]].copy()

# Rename columns for consistency
epoch_ai_clean.columns = [
    'model_name', 'publication_date', 'year', 'month',
    'domain', 'task', 'organization', 'country',
    'accessibility', 'parameters', 'training_compute_flop'
]

# Drop rows with missing publication dates and convert year/month to int
epoch_ai_clean = epoch_ai_clean.dropna(subset=['publication_date'])
epoch_ai_clean['year'] = epoch_ai_clean['year'].astype(int)
epoch_ai_clean['month'] = epoch_ai_clean['month'].astype(int)

display("Data cleaning and preprocessing completed.")
display("Cleaned AI Incidents Dataset")
display(ai_incidents_clean.head())
display("Cleaned Global Cybersecurity Threats Dataset")
display(cyber_threats_clean.head())
display("Cleaned CISSM Cyber Events Dataset")
display(cyber_events_clean.head())
display("Cyber Events after removing duplicates:") 
display(len(cyber_events_clean))
display("Cleaned Epoch AI Models Dataset (Public Language Models)")
display("Total models:", {len(epoch_ai_models)}, "Public models:", {len(epoch_public)}, "Public language models:", {len(epoch_ai_clean)})
display(epoch_ai_clean.head())


# #### After cleaning, we verify the quality of our prepared datasets by checking the shape, viewing info and rechecing for null values

# In[65]:


# Get info and check for missing values in cleaned datasets
display("Dataset Shapes After Cleaning:")
display("AI Incidents:", ai_incidents_clean.shape)
display("Global Cybersecurity Threats:", cyber_threats_clean.shape)
display("CISSM Cyber Events:", cyber_events_clean.shape)

display("AI Incidents Database Info")
display(ai_incidents_clean.info())
display(ai_incidents_clean.isna().sum())

display("Global Cybersecurity Threats Info")
display(cyber_threats_clean.info())
display(cyber_threats_clean.isna().sum())

display("CISSM Cyber Events Database Info")
display(cyber_events_clean.info())
display(cyber_events_clean.isna().sum())

display("Epoch AI Models Info")
display(epoch_ai_clean.info())
display(epoch_ai_clean.isna().sum())


# ---
# # Exploratory Data Analysis (EDA)

# #### Post cleaning validation, recheck the shape, info and null values

# In[66]:


# Begin exploratory data analysis 
display("Begin exploratory data analysis")

# Understand the time span for each dataset
display("Date Ranges")
display(f"AI Incidents: {ai_incidents_clean['year'].min()} - {ai_incidents_clean['year'].max()}")
display(f"Cyber Threats: {cyber_threats_clean['Year'].min()} - {cyber_threats_clean['Year'].max()}")
display(f"Cyber Events: {cyber_events_clean['year'].min()} - {cyber_events_clean['year'].max()}")
display(f"Epoch AI Models: {epoch_ai_clean['year'].min()} - {epoch_ai_clean['year'].max()}")

# Yearly Incident Counts
display("AI Incidents by Year")
display(ai_incidents_clean.groupby('year').size().reset_index(name='count'))

display("Cyber Threats by Year")
display(cyber_threats_clean.groupby('Year').size().reset_index(name='count'))

display("Cyber Events by Year")
display(cyber_events_clean.groupby('year').size().reset_index(name='count'))

display("Epoch AI Models Released by Year")
display(epoch_ai_clean.groupby('year').size().reset_index(name='count'))

# Categories of types of attacks, motives, and actors
display("Cyber Threats - Attack Types")
display(cyber_threats_clean['Attack Type'].value_counts())

display("Cyber Events - Event Types")
display(cyber_events_clean['event_type'].value_counts())

display("Cyber Events - Actor Types")
display(cyber_events_clean['actor_type'].value_counts())

display("Cyber Events - Motives")
display(cyber_events_clean['motive'].value_counts())

# Epoch AI Model characteristics
# Drop rows with missing publication dates
epoch_ai_clean = epoch_ai_clean.dropna(subset=['publication_date'])

display("Epoch AI Models - Accessibility Types")
display(epoch_ai_clean['accessibility'].value_counts())

display("Epoch AI Models - Top Organizations")
display(epoch_ai_clean['organization'].value_counts().head(10))


# 
# #### Defining the Analysis Framework
# To answer my research question, I establish a clear boundary between the **pre-AI era** (2015-2022) and **post-AI era** (2023+), based on ChatGPT's public release in November 2022.
# 
# This allows me to:
# - Compare cyber event patterns before and after widespread AI accessibility
# - Create a classification target for machine learning models
# - Standardize the analysis window (2015-present) across all datasets
# 
# **Note:** The post-AI era has limited data (2023+), which is a limitation of this analysis.

# In[67]:


# Define analysis period and AI era
# Define AI era based on ChatGPT public release (November 2022)
# Pre AI: 2015-2022 / Post-AI: 2023+ I wish we had more relevant data for 2024 but this is what we have to work with

# AI Incidents Dataset
# Filter analysis window and add era column
ai_incidents_clean = ai_incidents_clean[ai_incidents_clean['year'] >= 2015].copy()
ai_incidents_clean['ai_era'] = np.where(ai_incidents_clean['year'] >= 2023, 'post', 'pre')

# Global Cybersecurity Threats
# Add era column
cyber_threats_clean['ai_era'] = np.where(cyber_threats_clean['Year'] >= 2023, 'post', 'pre')

# CISSM Cyber Events Database
# Filter analysis window and add era column
cyber_events_clean = cyber_events_clean[cyber_events_clean['year'] >= 2015].copy()
cyber_events_clean['ai_era'] = np.where(cyber_events_clean['year'] >= 2023, 'post', 'pre')

# Epoch AI Models
# Add era column
epoch_ai_clean = epoch_ai_clean[epoch_ai_clean['year'] >= 2015].copy()
epoch_ai_clean['ai_era'] = np.where(epoch_ai_clean['year'] >= 2023, 'post', 'pre')

# Verify Era Distribution
display("AI Incidents by Era")
display(ai_incidents_clean['ai_era'].value_counts())

display("Cyber Threats by Era")
display(cyber_threats_clean['ai_era'].value_counts())

display("Cyber Events by Era")
display(cyber_events_clean['ai_era'].value_counts())

display("Epoch AI Models by Era")
display(epoch_ai_clean['ai_era'].value_counts())

# Show the acceleration in model releases
display("Public Language Model Releases by Year")
display(epoch_ai_clean.groupby('year').size())


# #### Comparative Analysis: Pre-AI vs Post-AI Era
# With the AI era boundary defined, I compare key metrics across eras to identify potential shifts in cybercrime patterns.
# 
# **Key comparisons:**
# - Financial impact and affected users
# - Attack type distributions
# - Actor types and motives
# - AI model availability and accessibility
# 
# **Note:** During this analysis, I identified that the Global Cybersecurity Threats dataset appears to contain generated data based on uniform distributions. We will rely primarily on the CISSM Cyber Events Database and Epoch AI datasets for my conclusions.

# In[68]:


# Try to understand impact and severity of incidents across eras

# Financial Impact
display("Cyber Threats - Average Financial Loss by Era")
display(cyber_threats_clean.groupby('ai_era')['Financial Loss (in Million $)'].mean().reset_index(name='avg_loss_million'))

# Financial loss seems skewed by outliers, let's look deeper
# After digging into the data, it appears to be generated data for illustration purposes, so we will just show summary statistics and sample values
display("Financial Loss - Summary Statistics")
display(cyber_threats_clean['Financial Loss (in Million $)'].describe())

display("Financial Loss - Sample Values")
display(cyber_threats_clean['Financial Loss (in Million $)'].head(20))

display("Cyber Threats - Average Affected Users by Era")
display(cyber_threats_clean.groupby('ai_era')['Number of Affected Users'].mean().reset_index(name='avg_affected_users'))

display("Cyber Threats - Attack Types by Era")
display(cyber_threats_clean.groupby(['ai_era', 'Attack Type']).size().reset_index(name='count'))

# Event Types & Motives
display("Cyber Events - Event Types by Era")
display(cyber_events_clean.groupby(['ai_era', 'event_type']).size().reset_index(name='count'))

display("Cyber Events - Motives by Era")
display(cyber_events_clean.groupby(['ai_era', 'motive']).size().reset_index(name='count'))

display("Cyber Events - Actor Types by Era")
display(cyber_events_clean.groupby(['ai_era', 'actor_type']).size().reset_index(name='count'))

# AI Model Availability by Era
display("Epoch AI - Model Releases by Era")
display(epoch_ai_clean.groupby('ai_era').size().reset_index(name='model_count'))

display("Epoch AI - Accessibility Types by Era")
display(epoch_ai_clean.groupby(['ai_era', 'accessibility']).size().reset_index(name='count'))

display("Epoch AI - Top Organizations by Era")
display(epoch_ai_clean.groupby(['ai_era', 'organization']).size().reset_index(name='count').sort_values(['ai_era', 'count'], ascending=[True, False]).groupby('ai_era').head(5))


# #### Data Quality Checks (again)
# Before proceeding to correlation analysis, I verify data quality across all datasets by checking for remaining duplicates and missing values. Note: Duplicates were previously removed from CISSM Cyber Events but not yet checked in other datasets.

# In[90]:


# Check for duplicated data
display("Duplicate Records Check")
display(f"AI Incidents duplicates: {ai_incidents_clean.duplicated().sum()}")
display(f"Cyber Events duplicates: {cyber_events_clean.duplicated().sum()}")
display(f"Epoch AI duplicates: {epoch_ai_clean.duplicated().sum()}")


# In[ ]:


# Dig in Cyber Events duplicate values
# Investigate the duplicates
display("Cyber Events - Duplicate Investigation")
display(f"Total records: {len(cyber_events_clean)}")
display(f"Duplicate records: {cyber_events_clean.duplicated().sum()}")
display(f"Unique records: {len(cyber_events_clean) - cyber_events_clean.duplicated().sum()}")

# Look at a sample of duplicates
display("Sample duplicate rows:")
display(cyber_events_clean[cyber_events_clean.duplicated(keep=False)].sort_values(['event_date', 'event_type']).head(10))


# In[71]:


# Check missing values again after cleaning
display("Missing Values Summary")
display("AI Incidents:")
display(ai_incidents_clean.isnull().sum())
display("Cyber Events:")
display(cyber_events_clean.isnull().sum())
display("Epoch AI:")
display(epoch_ai_clean.isnull().sum())


# In[72]:


# Check for outliers for model parameters in Epoch AI dataset
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

epoch_ai_clean['parameters'].dropna().plot(kind='box', ax=axes[0], title='Model Parameters (raw)')
epoch_ai_clean['parameters'].dropna().apply(np.log10).plot(kind='box', ax=axes[1], title='Model Parameters (log10)')

plt.tight_layout()
plt.show()


# In[73]:


# Histogram of events over time and distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Events over time
cyber_events_clean.groupby('year').size().plot(kind='bar', ax=axes[0,0], title='Cyber Events by Year')
epoch_ai_clean.groupby('year').size().plot(kind='bar', ax=axes[0,1], title='AI Model Releases by Year')

# Categorical distributions
cyber_events_clean['event_type'].value_counts().plot(kind='barh', ax=axes[1,0], title='Cyber Event Types')
epoch_ai_clean['accessibility'].value_counts().plot(kind='barh', ax=axes[1,1], title='Model Accessibility Types')

plt.tight_layout()
plt.show()


# In[74]:


# Correlation analysis between AI model releases and cyber events

# Aggregate by year for correlation
yearly_cyber = cyber_events_clean.groupby('year').size().rename('cyber_events')
yearly_ai = epoch_ai_clean.groupby('year').size().rename('ai_models')
yearly_ai_incidents = ai_incidents_clean.groupby('year').size().rename('ai_incidents')

yearly_combined = pd.concat([yearly_cyber, yearly_ai, yearly_ai_incidents], axis=1).dropna()

display("Yearly Aggregated Data")
display(yearly_combined)

display("Correlation Matrix")
display(yearly_combined.corr())

# Scatter matrix visualization
from pandas.plotting import scatter_matrix
scatter_matrix(yearly_combined, figsize=(10, 8), diagonal='hist')
plt.suptitle('Correlation: Cyber Events vs AI Model Releases')
plt.show()


# In[75]:


# Bar Chart Pre vs Post AI Era comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Cyber events by era
cyber_events_clean.groupby('ai_era').size().plot(kind='bar', ax=axes[0], title='Cyber Events by Era', color=['b', 'g'])
axes[0].set_ylabel('Count')

# AI model releases by era
epoch_ai_clean.groupby('ai_era').size().plot(kind='bar', ax=axes[1], title='AI Model Releases by Era', color=['r', 'c'])
axes[1].set_ylabel('Count')

# AI incidents by era
ai_incidents_clean.groupby('ai_era').size().plot(kind='bar', ax=axes[2], title='AI Incidents by Era', color=['y', 'm'])
axes[2].set_ylabel('Count')

plt.show()


# In[76]:


# Scatter plot with regression line

fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=yearly_combined, x='ai_models', y='cyber_events', ax=ax)
# Cyber events to AI model releases correlation
ax.set_title(f'AI Model Releases vs Cyber Events (r = 0.744)')
ax.set_xlabel('Public AI Model Releases')
ax.set_ylabel('Cyber Events')

# Annotate key years
for year in yearly_combined.index:
    ax.annotate(str(year), (yearly_combined.loc[year, 'ai_models'], yearly_combined.loc[year, 'cyber_events']))

plt.show()


# In[77]:


# Distribution chart of time series with ChatGPT release marked
fig, ax1 = plt.subplots(figsize=(12, 6))

# Cyber events line
ax1.set_xlabel('Year')
ax1.set_ylabel('Cyber Events', color='tab:red')
ax1.plot(yearly_combined.index, yearly_combined['cyber_events'], color='tab:red', marker='o', label='Cyber Events')
ax1.tick_params(axis='y', labelcolor='tab:red')

# AI models line (secondary axis)
ax2 = ax1.twinx()
ax2.set_ylabel('AI Model Releases', color='tab:blue')
ax2.plot(yearly_combined.index, yearly_combined['ai_models'], color='tab:blue', marker='s', label='AI Models')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Mark ChatGPT release
ax1.axvline(x=2022, color='g', linestyle='--', linewidth=2, label='ChatGPT Release (Nov 2022)')

plt.title('Cyber Events and AI Model Releases Over Time')
fig.legend(loc='upper left', bbox_to_anchor=(0.09, 0.9))
plt.tight_layout()
plt.show()


# In[78]:


# Attack types pre vs post AI era
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cyber_events_clean[cyber_events_clean['ai_era'] == 'pre']['event_type'].value_counts().plot(
    kind='pie', ax=axes[0], title='Cyber Event Types Pre AI Era', autopct='%1.1f%%')
axes[0].set_ylabel('')

cyber_events_clean[cyber_events_clean['ai_era'] == 'post']['event_type'].value_counts().plot(
    kind='pie', ax=axes[1], title='Cyber Event Types Post AI Era', autopct='%1.1f%%')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# In[79]:


# Dual axis time series with ChatGPT release marked

# Reset index if 'year' is the index
if 'year' not in yearly_combined.columns:
    yearly_combined = yearly_combined.reset_index()

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add cyber events trace
fig.add_trace(
    go.Scatter(
        x=yearly_combined['year'], 
        y=yearly_combined['cyber_events'],
        name="Cyber Events",
        line=dict(color='red', width=3),
        mode='lines+markers',
        marker=dict(size=10),
        hovertemplate='<b>Year:</b> %{x}<br><b>Cyber Events:</b> %{y}<extra></extra>'
    ),
    secondary_y=False,
)

# Add AI models trace
fig.add_trace(
    go.Scatter(
        x=yearly_combined['year'], 
        y=yearly_combined['ai_models'],
        name="AI Model Releases",
        line=dict(color='blue', width=3),
        mode='lines+markers',
        marker=dict(size=10),
        hovertemplate='<b>Year:</b> %{x}<br><b>AI Models:</b> %{y}<extra></extra>'
    ),
    secondary_y=True,
)

# Add vertical line for ChatGPT release
fig.add_vline(x=2022, line_dash="dash", line_color="green", line_width=2,
              annotation_text="ChatGPT Release", annotation_position="top")

# Add shaded region for post-AI era
fig.add_vrect(x0=2023, x1=yearly_combined['year'].max(), 
              fillcolor="lightgreen", opacity=0.2, line_width=0,
              annotation_text="Post-AI Era", annotation_position="top left")

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Cyber Events vs AI Model Releases Over Time</b>',
        font=dict(size=18)
    ),
    xaxis_title="Year",
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    hovermode='x unified',
    template='plotly_white',
    height=500
)

# Set y-axes titles
fig.update_yaxes(title_text="<b>Cyber Events</b>", secondary_y=False, color='red')
fig.update_yaxes(title_text="<b>AI Model Releases</b>", secondary_y=True, color='blue')

fig.show()


# In[80]:


# Scatter plot with regression line using Plotly

# Reset index if needed
if 'year' not in yearly_combined.columns:
    yearly_combined = yearly_combined.reset_index()

# Add ai_era column if it doesn't exist
if 'ai_era' not in yearly_combined.columns:
    yearly_combined['ai_era'] = np.where(yearly_combined['year'] >= 2023, 'post', 'pre')

# Calculate regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    yearly_combined['ai_models'], 
    yearly_combined['cyber_events']
)

# Create scatter plot
fig = px.scatter(
    yearly_combined, 
    x='ai_models', 
    y='cyber_events',
    text='year',
    color='ai_era',
    color_discrete_map={'pre': 'blue', 'post': 'red'},
    labels={
        'ai_models': 'Public AI Model Releases',
        'cyber_events': 'Cyber Events',
        'ai_era': 'AI Era'
    },
    title=f'<b>Correlation: AI Model Releases vs Cyber Events</b><br><sup>r = {r_value:.3f}, p = {p_value:.4f}</sup>'
)

# Add regression line
x_range = np.array([yearly_combined['ai_models'].min(), yearly_combined['ai_models'].max()])
y_pred = slope * x_range + intercept

fig.add_trace(
    go.Scatter(
        x=x_range, 
        y=y_pred,
        mode='lines',
        name=f'Regression (r={r_value:.3f})',
        line=dict(color='gray', dash='dash', width=2)
    )
)

# Update marker styling
fig.update_traces(
    marker=dict(size=14, line=dict(width=2, color='white')),
    textposition='top center',
    selector=dict(mode='markers+text')
)

# Update layout
fig.update_layout(
    template='plotly_white',
    height=500,
    legend=dict(x=0.01, y=0.99),
    hovermode='closest'
)

# Update hover template for scatter points
fig.update_traces(
    hovertemplate='<b>Year:</b> %{text}<br><b>AI Models:</b> %{x}<br><b>Cyber Events:</b> %{y}<extra></extra>',
    selector=dict(mode='markers+text')
)

fig.show()


# In[81]:


# Line Chart of Industries Targeted by Cyber Attacks Over Time

# Aggregate by year and industry
industry_by_year = cyber_events_clean.groupby(['year', 'industry']).size().reset_index(name='count')

# Filter to analysis period (2015+)
industry_by_year = industry_by_year[industry_by_year['year'] >= 2015]

# Truncate industry names to 20 characters and add ... 
industry_by_year['industry_short'] = industry_by_year['industry'].str[:20] + \
    industry_by_year['industry'].str.len().gt(20).map({True: '...', False: ''})

# Get top 8 industries overall (to keep chart readable)
top_industries = cyber_events_clean['industry'].value_counts().head(8).index.tolist()
industry_by_year_top = industry_by_year[industry_by_year['industry'].isin(top_industries)]

# Create interactive line chart
fig = px.line(
    industry_by_year_top,
    x='year',
    y='count',
    color='industry_short',
    markers=True,
    title='<b>Industries Targeted by Cyber Attacks Over Time</b><br><sup>Top 8 most targeted industries (2015-present)</sup>',
    labels={'count': 'Number of Events', 'year': 'Year', 'industry_short': 'Industry'}
)

# Add ChatGPT release marker
fig.add_vline(x=2022, line_dash="dash", line_color="red", line_width=2,
              annotation_text="ChatGPT Release", annotation_position="top left")


fig.update_layout(
    template='plotly_white',
    height=500,
    hovermode='x unified',
    legend=dict(title='Industry', y=0.5)
)

fig.update_traces(line=dict(width=2.5), marker=dict(size=8))

fig.show()


# In[82]:


# Prepare event-level data for classification
# Filter to analysis period (2015+)
cyber_ml = cyber_events_clean[cyber_events_clean['year'] >= 2015].copy()

# Create target variable for ai_era 
cyber_ml['ai_era'] = np.where(cyber_ml['year'] >= 2023, 'post', 'pre')

# Check the data
display("Dataset shape:", cyber_ml.shape)
display("Target distribution:")
display(cyber_ml['ai_era'].value_counts())
cyber_ml.head()


# In[83]:


# Separate features and target
feature_cols = ['event_type', 'actor_type', 'motive', 'industry']

# Drop rows with missing values in our feature columns
cyber_ml_clean = cyber_ml.dropna(subset=feature_cols)

cyber_X = cyber_ml_clean[feature_cols]
cyber_y = cyber_ml_clean['ai_era']

display("Features shape:", cyber_X.shape)
display("Target shape:", cyber_y.shape)


# In[84]:


# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    cyber_X, cyber_y, 
    test_size=0.2, 
    random_state=42, 
    stratify=cyber_y
)

display("Training set size:", len(X_train))
display("Test set size:", len(X_test))
display("Training target distribution:")
display(y_train.value_counts())


# In[85]:


# Define numeric and categorical features
cat_features = ['event_type', 'actor_type', 'motive', 'industry']

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('one-hot-encode', OneHotEncoder(handle_unknown='ignore'))
])

# Full pipeline using ColumnTransformer
full_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_features)
])

# Transform the training data
X_train_prepared = full_pipeline.fit_transform(X_train)
display("Transformed training data shape:", X_train_prepared.shape)


# In[86]:


# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_prepared, y_train)

# Predictions on training data
predictions = log_reg.predict(X_train_prepared)
train_accuracy = accuracy_score(y_train, predictions)
display("Logistic Regression Training Accuracy:", train_accuracy)

# Model 2: Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_prepared, y_train)

predictions = tree_clf.predict(X_train_prepared)
tree_train_accuracy = accuracy_score(y_train, predictions)
display("Decision Tree Training Accuracy:", tree_train_accuracy)

# Model 3: Random Forest
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train_prepared, y_train)

predictions = forest_clf.predict(X_train_prepared)
forest_train_accuracy = accuracy_score(y_train, predictions)
display("Random Forest Training Accuracy:", forest_train_accuracy)


# In[87]:


# Cross validation for Logistic Regression

log_scores = cross_val_score(log_reg, X_train_prepared, y_train, cv=5, scoring='accuracy')
display("Logistic Regression CV Scores:", log_scores)
display("Mean:", log_scores.mean())
display("Std:", log_scores.std())

# Cross validation for Decision Tree
tree_scores = cross_val_score(tree_clf, X_train_prepared, y_train, cv=5, scoring='accuracy')
display("Decision Tree CV Scores:", tree_scores)
display("Mean:", tree_scores.mean())
display("Std:", tree_scores.std())

# Cross validation for Random Forest
forest_scores = cross_val_score(forest_clf, X_train_prepared, y_train, cv=5, scoring='accuracy')
display("Random Forest CV Scores:", forest_scores)
display("Mean:", forest_scores.mean())
display("Std:", forest_scores.std())


# In[88]:


# Evaluate on test set
X_test_prepared = full_pipeline.transform(X_test)

# Logistic Regression on test set
predictions_log = log_reg.predict(X_test_prepared)
log_test_accuracy = accuracy_score(y_test, predictions_log)
display("Logistic Regression Test Accuracy:", log_test_accuracy)

# Decision Tree on test set
predictions_tree = tree_clf.predict(X_test_prepared)
tree_test_accuracy = accuracy_score(y_test, predictions_tree)
display("Decision Tree Test Accuracy:", tree_test_accuracy)

# Random Forest on test set
predictions_forest = forest_clf.predict(X_test_prepared)
forest_test_accuracy = accuracy_score(y_test, predictions_forest)
display("Random Forest Test Accuracy:", forest_test_accuracy)

# Detailed classification report for best model
display("Classification Report (Random Forest):")
display(classification_report(y_test, predictions_forest))

# Summary comparison (like comparing lin_rmse, poly_rmse, tree_rmse in sandbox)
display("="*50)
display("MODEL COMPARISON SUMMARY")
display("="*50)
display(f"{'Model':<25} {'CV Mean':<12} {'CV Std':<12} {'Test Acc':<12}")
display("-"*50)
display(f"{'Logistic Regression':<25} {log_scores.mean():<12.4f} {log_scores.std():<12.4f} {log_test_accuracy:<12.4f}")
display(f"{'Decision Tree':<25} {tree_scores.mean():<12.4f} {tree_scores.std():<12.4f} {tree_test_accuracy:<12.4f}")
display(f"{'Random Forest':<25} {forest_scores.mean():<12.4f} {forest_scores.std():<12.4f} {forest_test_accuracy:<12.4f}")


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# * https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
# * https://www.w3schools.com/python/pandas/ref_df_corr.asp
# * https://docs.scipy.org/doc/scipy/reference/main_namespace.html 
# * https://wesmckinney.com/book/ 
# * https://jakevdp.github.io/PythonDataScienceHandbook/
# * https://github.com/IT4063C-Fall22/Sandbox/blob/e2e/sandbox.ipynb 

# In[89]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

