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

# In[2]:


# Imports
import pandas as pd
import numpy as np


# In[3]:


# Load datasets

# AI Incident Database
ai_incidents = pd.read_csv('data/AI_incidents_database.csv')

# Global Cybersecurity Threats
cyber_threats = pd.read_csv('data/Global_Cybersecurity_Threats_2015_2024.csv')

# CISSM Cyber Events Database
cyber_events = pd.read_csv('data/CISSM_Cyber_Events_Database_2014_Oct_2025.csv')

# Epoch AI Model Tracking
epoch_ai_models = pd.read_csv('data/epoch_ai_models.csv')


# In[4]:


# Check first few rows of each dataset

display("AI Incidents Database")
display(ai_incidents.head())

display("Global Cybersecurity Threats")
display(cyber_threats.head())

display("CISSM Cyber Events Database")
display(cyber_events.head())

display("Epoch AI Model Tracking")
display(epoch_ai_models.head())


# In[5]:


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


# In[6]:


# Count null or missing values
display("AI Incidents Database Missing Values")
display(ai_incidents.isnull().sum())
display("Global Cybersecurity Threats Missing Values")
display(cyber_threats.isnull().sum())
display("CISSM Cyber Events Database Missing Values")
display(cyber_events.isnull().sum())
display("Epoch AI Model Tracking Missing Values")
display(epoch_ai_models.isnull().sum())


# In[ ]:


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
display("Cleaned Epoch AI Models Dataset (Public Language Models)")
display("Total models:", {len(epoch_ai_models)}, "Public models:", {len(epoch_public)}, "Public language models:", {len(epoch_ai_clean)})
display(epoch_ai_clean.head())


# In[13]:


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


# In[15]:


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


# In[18]:


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


# In[20]:


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


# In[21]:


# Check for duplicated data
display("Duplicate Records Check")
display(f"AI Incidents duplicates: {ai_incidents_clean.duplicated().sum()}")
display(f"Cyber Events duplicates: {cyber_events_clean.duplicated().sum()}")
display(f"Epoch AI duplicates: {epoch_ai_clean.duplicated().sum()}")


# In[23]:


# Dig in Cyber Events duplicate values
# Investigate the duplicates
display("Cyber Events - Duplicate Investigation")
display(f"Total records: {len(cyber_events_clean)}")
display(f"Duplicate records: {cyber_events_clean.duplicated().sum()}")
display(f"Unique records: {len(cyber_events_clean) - cyber_events_clean.duplicated().sum()}")

# Look at a sample of duplicates
display("Sample duplicate rows:")
display(cyber_events_clean[cyber_events_clean.duplicated(keep=False)].sort_values(['event_date', 'event_type']).head(10))


# In[22]:


# Check missing values again after cleaning
display("Missing Values Summary")
display("AI Incidents:")
display(ai_incidents_clean.isnull().sum())
display("Cyber Events:")
display(cyber_events_clean.isnull().sum())
display("Epoch AI:")
display(epoch_ai_clean.isnull().sum())


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[12]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

