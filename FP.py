#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install numpy')
get_ipython().run_line_magic('pip', 'install --no-binary :all: mlxtend')
get_ipython().run_line_magic('pip', 'install chart-studio')


# In[2]:


# importing module
import pandas as pd

# dataset
dataset = pd.read_csv("data.csv")

# printing the shape of the dataset
dataset.shape


# In[3]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


# In[ ]:


# importing module
import numpy as np

# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, dataset.shape[0]):
    for j in range(0, dataset.shape[1]):
        transaction.append(dataset.values[i,j])

# converting to numpy array
transaction = np.array(transaction)
print(transaction)


# In[6]:


#top 5 items
#  Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"]) 

# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
df["incident_count"] = 1 

#  Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)

# Making a New Appropriate Pandas DataFrame for Visualizations  
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

#  Initial Visualizations
df_table.head(5).style.background_gradient(cmap='Greys')


# In[7]:


#tree map
# importing required module
from chart_studio import plotly
import plotly.express as px

# to have a same origin
df_table["all"] = "Top 50 items" 

# creating tree map using plotly
fig = px.treemap(df_table.head(50), path=['all', "items"], values='incident_count',
                  color=df_table["incident_count"].head(50), hover_data=['items'],
                  color_continuous_scale='Greys',
                )
# ploting the treemap
fig.show()


# In[8]:


#encoding
# Transform Every Transaction to Seperate List & Gather Them into Numpy Array
transaction = []
for i in range(dataset.shape[0]):
    transaction.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])

# creating the numpy array of the transactions
transaction = np.array(transaction)

# importing the required module
from mlxtend.preprocessing import TransactionEncoder

# initializing the transactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

# dataset after encoded
dataset.head()


# In[10]:


# select top 30 items
first30 = df_table["items"].head(30).values 

# Extract Top 30
dataset = dataset.loc[:,first30] 

# shape of the dataset
dataset.shape


# In[11]:


#fp
#Importing Libraries
from mlxtend.frequent_patterns import fpgrowth

#running the fpgrowth algorithm
res=fpgrowth(dataset,min_support=0.05, use_colnames=True)

# printing top 10
res.head(10)


# In[12]:


# importing required module
from mlxtend.frequent_patterns import association_rules

# creating asssociation rules
res=association_rules(res, metric="lift", min_threshold=1)

# printing association rules
res


# In[13]:


# Sort values based on confidence
res.sort_values("confidence",ascending=False)


# In[ ]:




