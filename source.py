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
# * **Global Cybersecurity Threats (2015-2024):** A comprehensive dataset tracking cybersecurity incidents, attack vectors, threat 
#     * https://www.kaggle.com/datasets/atharvasoundankar/global-cybersecurity-threats-2015-2024
# * **AI incident database:** Documenting the times when things go wrong with AI solutions
#     * https://www.kaggle.com/datasets/konradb/ai-incident-database

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# <br>
# #### <span style = 'color:green'>ChatGPT's public release was November of 2022 we will use that as a data point to compare metrics before and after wide spread AI availability. The Global Cybersecurity Threats dataset provides volume and attack type trends, the Cyber Events Database shows incident level context on motives and actors, and the AI Incident Database identifies specific cases of AI use allowing us to try and correlate AI availability with changes in cybercrime patterns.</span>

# In[23]:


# Imports
import pandas as pd
import numpy as np


# In[2]:


# Load datasets

# AI Incident Database
ai_incidents = pd.read_csv('data/AI_incidents_database.csv')

# Global Cybersecurity Threats
cyber_threats = pd.read_csv('data/Global_Cybersecurity_Threats_2015_2024.csv')

# CISSM Cyber Events Database
cyber_events = pd.read_csv('data/CISSM_Cyber_Events_Database_2014_Oct_2025.csv')


# In[3]:


# Check first few rows of each dataset

display("AI Incidents Database")
display(ai_incidents.head())

display("\nGlobal Cybersecurity Threats")
display(cyber_threats.head())

display("\nCISSM Cyber Events Database")
display(cyber_events.head())


# In[18]:


# Get info about each dataset
display("Dataset Shapes After Cleaning:")
display("AI Incidents:", ai_incidents.shape)
display("Global Cybersecurity Threats:", cyber_threats.shape)
display("CISSM Cyber Events:", cyber_events.shape)

display("AI Incidents Database Info")
display(ai_incidents.info())
display("Global Cybersecurity Threats Info")
display(cyber_threats.info())
display("CISSM Cyber Events Database Info")
display(cyber_events.info())


# In[ ]:


# Count null or missing values
display("AI Incidents Database Missing Values")
display(ai_incidents.isnull().sum())
display("Global Cybersecurity Threats Missing Values")
display(cyber_threats.isnull().sum())
display("CISSM Cyber Events Database Missing Values")
display(cyber_events.isnull().sum())


# In[10]:


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

display("Data cleaning and preprocessing completed.")
display("Cleaned AI Incidents Dataset")
display(ai_incidents_clean.head())
display("Cleaned Global Cybersecurity Threats Dataset")
display(cyber_threats_clean.head())
display("Cleaned CISSM Cyber Events Dataset")
display(cyber_events_clean.head())


# In[17]:


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


# In[22]:


# Begin exploratory data analysis 
display("Begin exploratory data analysis")

# Understand the time span for each dataset
display("Date Ranges")
display(f"AI Incidents: {ai_incidents_clean['year'].min()} - {ai_incidents_clean['year'].max()}")
display(f"Cyber Threats: {cyber_threats_clean['Year'].min()} - {cyber_threats_clean['Year'].max()}")
display(f"Cyber Events: {cyber_events_clean['year'].min()} - {cyber_events_clean['year'].max()}")

# Yearly Incident Counts
display("AI Incidents by Year")
display(ai_incidents_clean.groupby('year').size().reset_index(name='count'))

display("Cyber Threats by Year")
display(cyber_threats_clean.groupby('Year').size().reset_index(name='count'))

display("Cyber Events by Year")
display(cyber_events_clean.groupby('year').size().reset_index(name='count'))

# Categories of types of attacks, motives, and actors
display("Cyber Threats - Attack Types")
display(cyber_threats_clean['Attack Type'].value_counts())

display("Cyber Events - Event Types")
display(cyber_events_clean['event_type'].value_counts())

display("Cyber Events - Actor Types")
display(cyber_events_clean['actor_type'].value_counts())

display("Cyber Events - Motives")
display(cyber_events_clean['motive'].value_counts())


# In[32]:


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

# Verify Era Distribution
display("AI Incidents by Era")
display(ai_incidents_clean['ai_era'].value_counts())

display("\nCyber Threats by Era")
display(cyber_threats_clean['ai_era'].value_counts())

display("\nCyber Events by Era")
display(cyber_events_clean['ai_era'].value_counts())


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[7]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

