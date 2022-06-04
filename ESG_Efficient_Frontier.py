#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

from tqdm import tqdm

from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff


# In[3]:


df = pd.read_csv(r'C:\Users\sdama\Desktop\3° ANNO\TESI\DATI\ESG_Data.csv',sep = ";",header = 1, index_col = 0)
pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[4]:


print(df.head())


# In[5]:


# Get annualised returns from converting monthly returns to daily returns
# Afterward multiply to 252 to annualise the returns
# Please note: 252 are the trading days in a year
mus = (1+df.mean()/21)**252 - 1
print(mus)


# In[6]:


#Covariances of annualised returns from monthly returns divifing by 21 and multiplying by 252
cov = (df.cov()/21)*252
print(cov)


# In[7]:


## Creation of random ESG portfolios
# Assests included in each portfolio
n_assets = 4
# Portfolios generated
n_portfolios = 10000

# Start empty list to memorize mean-variance pairs for plotting
mean_variance_pairs = []

np.random.seed(75)
# Loop through and generate lots of random portfolios
for i in range(n_portfolios):
    # Choose assets randomly without replacement
    assets = np.random.choice(list(df.columns), n_assets, replace = False)
    # Choose weights randomly
    weights = np.random.rand(n_assets)
    # Ensure weights sum to 1
    weights = weights/sum(weights)

    # Loop over asset pairs and compute portfolio return and variance
    portfolio_Exp_Variance = 0
    portfolio_Exp_Return = 0
    for i in range(len(assets)):
        portfolio_Exp_Return += weights[i] * mus.loc[assets[i]]
        for j in range(len(assets)):
            # Add variance/covariance for each asset pair
            # Note that when i==j this adds the variance
            portfolio_Exp_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]
            
    # Add the mean/variance pairs to a list for plotting
    mean_variance_pairs.append([portfolio_Exp_Return, portfolio_Exp_Variance])


# In[8]:


# Plot the risk vs. return of randomly generated portfolios
# Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate = 0 # Include risk free rate here

fig = go.Figure()
fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], 
                      marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
                                  showscale=True, 
                                  size=7,
                                  line=dict(width=1),
                                  colorscale="RdBu",
                                  colorbar=dict(title="Sharpe<br>Ratio")
                                 ), 
                      mode='markers'))
fig.update_layout(template='plotly_white',
                  xaxis=dict(title='Annualised Risk (Volatility)'),
                  yaxis=dict(title='Annualised Return'),
                  title='Sample of Random Portfolios',
                  width=850,
                  height=500)
fig.update_xaxes(range=[0, 0.1])
fig.update_yaxes(range=[-0.02,0.07])
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))


# In[9]:


# Random portfolio weights and indexes
# Number of assests in the portfolio
n_assets = 4

mean_variance_pairs = []
weights_list=[]
tickers_list=[]

for i in tqdm(range(10000)):
    next_i = False
    while True:
        #- Choose assets randomly without replacement
        assets = np.random.choice(list(df.columns), n_assets, replace=False)
        #- Choose weights randomly ensuring they sum to one
        weights = np.random.rand(n_assets)
        weights = weights/sum(weights)

        #-- Loop over asset pairs and compute portfolio return and variance
        portfolio_Exp_Variance = 0
        portfolio_Exp_Return = 0
        for i in range(len(assets)):
            portfolio_Exp_Return += weights[i] * mus.loc[assets[i]]
            for j in range(len(assets)):
                portfolio_Exp_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

        # Skip over dominated portfolios
        for R,V in mean_variance_pairs:
            if (R > portfolio_Exp_Return) & (V < portfolio_Exp_Variance):
                next_i = True
                break
        if next_i:
            break

        # Add the mean/variance pairs to a list for plotting
        mean_variance_pairs.append([portfolio_Exp_Return, portfolio_Exp_Variance])
        weights_list.append(weights)
        tickers_list.append(assets)
        break


# In[10]:


len(mean_variance_pairs)


# In[12]:


# Plot the risk vs. return of randomly generated ESG portfolios
# Convert the list from before into an array for easy plotting
mean_variance_pairs = np.array(mean_variance_pairs)

risk_free_rate = 0 

fig = go.Figure()
fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], 
                      marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
                                  showscale=True, 
                                  size=7,
                                  line=dict(width=1),
                                  colorscale="RdBu",
                                  colorbar=dict(title="Sharpe<br>Ratio")
                                 ), 
                      mode='markers',
                      text=[str(np.array(tickers_list[i])) + "<br>" + str(np.array(weights_list[i]).round(2)) for i in range(len(tickers_list))]))
fig.update_layout(template='plotly_white',
                  xaxis=dict(title='Annualised Risk (Volatility)'),
                  yaxis=dict(title='Annualised Return'),
                  title='Sample of ESG Random and Efficient Portfolios',
                  width=850,
                  height=500)
fig.update_xaxes(range=[0, 0.1])
fig.update_yaxes(range=[-0.02,0.1])
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))


# In[ ]:




