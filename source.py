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

# In[1]:


# Imports
import pandas as pd


# In[12]:


# Load datasets

# AI Incident Database
ai_incidents = pd.read_csv('data/AI_incidents_database.csv')

# Global Cybersecurity Threats
cyber_threats = pd.read_csv('data/Global_Cybersecurity_Threats_2015_2024.csv')

# CISSM Cyber Events Database
cyber_events = pd.read_csv('data/CISSM_Cyber_Events_Database_2014_Oct_2025.csv')


# In[6]:


# Check first few rows of each dataset

display("AI Incidents Database")
display(ai_incidents.head())

display("\nGlobal Cybersecurity Threats")
display(cyber_threats.head())

display("\nCISSM Cyber Events Database")
display(cyber_events.head())


# In[15]:


# Get info about each dataset
display("AI Incidents Database Info")
display(ai_incidents.info())
display("Global Cybersecurity Threats Info")
display(cyber_threats.info())
display("ISSM Cyber Events Database Info")
display(cyber_events.info())


# In[14]:


# Count null or missing values
display("AI Incidents Database Missing Values")
display(ai_incidents.isnull().sum())
display("Global Cybersecurity Threats Missing Values")
display(cyber_threats.isnull().sum())
display("CISSM Cyber Events Database Missing Values")
display(cyber_events.isnull().sum())


# In[19]:


# Data cleaning and preprocessing

# AI Incidents
# Convert date string to datetime and extract year for time based analysis, we need this to filter by year later
ai_incidents['date'] = pd.to_datetime(ai_incidents['date'])
ai_incidents['year'] = ai_incidents['date'].dt.year
ai_incidents['month'] = ai_incidents['date'].dt.month

# Select only columns relevant for analysis
ai_incidents_clean = ai_incidents[[
    'incident_id', 'date', 'year', 'title', 'description',
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


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

