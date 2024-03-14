#!/usr/bin/env python
# coding: utf-8

# # 

# <a id='1'></a><center> <h3 style="background-color:lightgreen; color:black" ><br>Project Title: Crop Production Analysis in India<br></h3>

# ![Crop Production Analysis in India](https://qph.cf2.quoracdn.net/main-qimg-07407e21a39d94abad6b541c42258e21)

# **`INTRODUCTION`**

# The Crop Production Analysis in India project aims to leverage data science techniques to analyze and predict crop production trends in India. The agriculture domain plays a crucial role in the overall supply chain, and advancements in technology, particularly in the realm of the Future Internet, are expected to significantly impact this sector. This project focuses on developing a Business-to-Business collaboration platform within the agri-food sector to enhance collaboration among stakeholders.
# 
# 

# # 

# # ***`Import necessary libraries`***

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# # 

# # ***`Loading Dataset`***

# In[2]:


crop_data = pd.read_csv('C:/Users/stati/OneDrive/Desktop/Crop Production Analysis in India.csv')

crop_data.shape


# # 

# # ***`EDA (Exploratory Data Analysis)`***

# In[3]:


crop_data.head()


# In[4]:


crop_data.info()


# In[5]:


crop_data.isnull().sum()


# In[6]:


data = crop_data.dropna()

data.shape


# In[7]:


data.columns


# In[8]:


data.nunique()


# In[9]:


sum_maxp = data["Production"].sum()

data["percent_of_production"] = data["Production"].map(lambda x:(x/sum_maxp)*100)


# # 

# # ***`Visualization`***

# **`Crop Production Trends Over Years`**

# In[10]:


# Import visualization libraries

import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


# Set the style for seaborn

sns.set(style="whitegrid")


# In[12]:


# Visualizing production trends over crop years using a line plot

plt.figure(figsize=(10, 6))
sns.lineplot(x=data["Crop_Year"], y=data["Production"])
plt.title("Crop Production Trends Over Years")
plt.xlabel("Crop Year")
plt.ylabel("Production")
plt.show()


# **`Total Crop Production by State in India`**

# In[13]:


# Grouping by 'State_Name' and summing 'Production' for each state

state_production = data.groupby('State_Name')['Production'].sum().sort_values(ascending=False)


# In[14]:


# Plotting the bar chart

plt.figure(figsize=(12, 8))
sns.barplot(x=state_production.values, y=state_production.index, palette='viridis')
plt.xlabel('Total Production (in tons)')
plt.title('Total Crop Production by State in India')
plt.show()


# **`Top 10 States by Crop Production in India`**

# In[15]:


# Plotting the bar chart for the top 10 states

top_states_production = state_production.head(10)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_states_production.index, y=top_states_production.values, color='green', orient='v')
plt.ylabel('Total Production (in tons)')
plt.title('Top 10 States by Crop Production in India')
plt.xticks(rotation=90, ha='right')
plt.show()


# **`Pairplot: Area vs. Production`**

# In[16]:


# Pair plot for selected variables

sns.set(style="ticks")
plot = sns.pairplot(data[['Area', 'Production', 'Season']], hue='Season', palette='viridis', height=4)
plot.fig.suptitle('Pair Plot: Area, Production, and Season', y=1.02)
plt.show()


# **`Seasonal Crop Production Analysis`**

# In[17]:


# Count plot for 'Season' vs. 'Production' with a different color palette

plt.figure(figsize=(8, 6))
sns.countplot(x="Season", data=data, palette='viridis')
plt.title('Count Plot: Season vs. Production')
plt.xlabel('Season')
plt.ylabel('Count')
plt.show()


# Grouping by 'Season' and summing 'Production'
data.groupby("Season")["Production"].sum().reset_index()


# # 

# **`Top Crops by Production`**

# In[18]:


# Displaying the top 5 crops by production

data["Crop"].value_counts()[:5]


# In[19]:


# Grouping by 'Crop' and summing 'Production'

top_crop_pro = data.groupby("Crop")["Production"].sum().reset_index().sort_values(by='Production', ascending=False)
top_crop_pro[:5]


# **`Rice Production Analysis`**

# In[20]:


# Subset for Rice, Coconut, and Sugarcane production analysis

rice_df = data[data["Crop"] == "Rice"]


# In[21]:


# Barplot for 'Season' vs. 'Production' for Rice

sns.barplot(x="Season", y="Production", data=rice_df, palette='viridis')
plt.title("Rice Production Across Seasons")
plt.xlabel("Season")
plt.ylabel("Production")
plt.show()


# In[22]:


# Barplot for 'State_Name' vs. 'Production' for Rice

sns.barplot(x="State_Name", y="Production", data=rice_df, palette='viridis')
plt.title("Rice Production Across States")
plt.xlabel("State")
plt.ylabel("Production")
plt.xticks(rotation=90)
plt.show()


# **`Top Rice Producing Districts`**

# In[23]:


# Top rice-producing districts

top_rice_pro_dis = rice_df.groupby("District_Name")["Production"].sum().nlargest(5).reset_index()


# In[24]:


# Calculate the percentage of production for each district

top_rice_pro_dis["percent_of_pro"] = top_rice_pro_dis["Production"] / top_rice_pro_dis["Production"].sum() * 100


# In[25]:


# Barplot for top rice-producing districts

plt.figure(figsize=(9, 6))
sns.barplot(x="District_Name", y="Production", data=top_rice_pro_dis, palette='viridis')
plt.title("Top Rice-Producing Districts")
plt.xlabel("District")
plt.ylabel("Production")
plt.show()


# # 

# **`Rice Production Trends Over Years`**

# In[26]:


# Line plot for 'Crop_Year' vs. 'Production' for Rice

plt.figure(figsize=(10, 5))
sns.lineplot(x="Crop_Year", y="Production", data=rice_df, marker='o', color='blue')
plt.title("Rice Production Over Years")
plt.xlabel("Crop Year")
plt.ylabel("Production")
plt.grid(True)


# # 

# # ***`Data Preprocessing for Modeling`***

# In[27]:


# Dropping unnecessary columns for modeling

data1 = data.drop(["District_Name", "Crop_Year"], axis=1)


# In[28]:


# Creating dummy variables

data_dum = pd.get_dummies(data1)
data_dum[:5]


# # 

# # ***`Modeling: Decision Tree Regressor`***

# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[30]:


# Splitting the data into training and testing sets

from sklearn.model_selection import train_test_split

x = data_dum.drop("Production", axis=1)
y = data_dum[["Production"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[31]:


# Fitting the Decision Tree Regressor model

regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(x_train, y_train)


# In[32]:


# Making predictions

preds = regressor.predict(x_test)


# In[33]:


# Evaluating the model

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print('Mean Squared Error (MSE):', mse)
print('R-squared (R^2) score:', r2)


# # 
