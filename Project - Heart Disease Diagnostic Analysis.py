#!/usr/bin/env python
# coding: utf-8

# <a id='1'></a><center> <h3 style="background-color:lightgreen; color:black" ><br>Project Title: Heart Disease Diagnostic  Analysis<br></h3>

# ![Heart Disease Dignostic Analysis](https://media.istockphoto.com/id/1359314170/photo/heart-attack-and-heart-disease-3d-illustration.jpg?s=612x612&w=0&k=20&c=K5Y-yzsfs7a7CyuAw-B222EMkT04iRmiEWzhIqF0U9E=)

# # ***`Import necessary libraries`***

# In[1]:


# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 

# # ***`Loading Dataset`***

# In[2]:


# Loading Dataset

Heart_Disease = pd.read_csv('C:\\Users\\stati\\OneDrive\\Desktop\\Heart Disease data.csv')

Heart_Disease


# # 

# # ***`EDA (Exploratory Data Analysis)`***

# In[3]:


Heart_Disease.info()


# In[4]:


Heart_Disease.isnull().sum()


# In[5]:


Heart_Disease.duplicated().sum()


# In[6]:


Heart_Disease.drop_duplicates(inplace=True)


# In[7]:


Heart_Disease.duplicated().sum()


# In[8]:


Heart_Disease.shape


# In[9]:


Heart_Disease.info()


# In[10]:


Heart_Disease.describe().T


# In[11]:


Heart_Disease.target.value_counts()


# # 

# # ***`Visualization`***

# **Pie charts for Heart Disease Distribution and Gender Distribution**

# In[12]:


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Heart Disease Distribution
axs[0].pie(Heart_Disease.target.value_counts(),
           labels=['Yes', 'No'], colors=sns.cubehelix_palette(start=2),
           autopct='%1.1f%%', startangle=90)
axs[0].set_title("Heart Disease")

# Gender Distribution
axs[1].pie(Heart_Disease.sex.value_counts(),
           labels=['Male', 'Female'], colors=sns.cubehelix_palette(start=2),
           autopct='%1.1f%%', startangle=90)
axs[1].set_title("Gender")

[ax.axis('off') for ax in axs]
plt.show()


# **Distribution Analysis of Numerical Data**

# In[13]:


sns.set(style="darkgrid")
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

Numerical_feature = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for ax, Numerical_feature in zip(axes.flatten(),Numerical_feature ):
    sns.histplot(Heart_Disease[Numerical_feature], kde=True, color=sns.color_palette("cubehelix")[0], ax=ax)
    ax.set_title(f'Distribution of {Numerical_feature.capitalize()}')
    ax.set(xlabel=Numerical_feature, ylabel='Frequency')
    ax.legend([f"Skew: {Heart_Disease[Numerical_feature].skew():.2f}", 
               f"Kurt: {Heart_Disease[Numerical_feature].kurt():.2f}"], title="Skewness and Kurtosis")

# Remove the last subplot as it's unused
fig.delaxes(axes.flatten()[-1])

plt.tight_layout()
plt.show()


# **Pie Chart of Categorical Data**

# In[14]:


Categorical_Features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

Cat_features = len(Categorical_Features)
Cat_rows, Cat_cols = (Cat_features + 1) // 2, min(2, Cat_features)

fig, axes = plt.subplots(Cat_rows, Cat_cols, figsize=(10, 4*Cat_rows))

for i, feature in enumerate(Categorical_Features):  
    r, c = i // Cat_cols, i % Cat_cols
    values = Heart_Disease[feature].value_counts()
    axes[r, c].pie(values, labels=values.index, autopct='%1.1f%%', colors=sns.cubehelix_palette(start=2))
    axes[r, c].set_title(feature)

plt.tight_layout()
plt.show()


# **Bar chart for Chest Pain Type counts**

# In[15]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1 - Chest Pain Type vs Count
ax1 = Heart_Disease['cp'].value_counts().plot(kind='bar', color=sns.cubehelix_palette(start=2), ax=axes[0])
ax1.set_xticklabels(labels=['type 0', 'type 1', 'type 2', 'type 3'], rotation=0)
ax1.set_title('Chest pain type vs count')

# Chart 2 - Type of chest pain for sex
ax2 = pd.crosstab(Heart_Disease['sex'], Heart_Disease['cp']).plot(kind='bar', color=sns.cubehelix_palette(start=2), ax=axes[1])
ax2.set_xticklabels(labels=['Female', 'Male'], rotation=0)
ax2.set_title('Type of chest pain for sex')

plt.show()


# **Correlation Heatmap of Numerical Variables**

# In[16]:


plt.figure(figsize=(10, 8))
sns.heatmap(Heart_Disease.corr(), annot=True, cmap=
            sns.cubehelix_palette(start=2), fmt=".2f", linewidths=.5).set(
    title="Correlation Heatmap of Dataset")
plt.show()


# **The Relationship Between Categorical Variables and Heart Disease (Target)**

# In[17]:


sns.set(style="darkgrid")

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))

Categorical_Features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for i, cat in enumerate(Categorical_Features):
    row = i // 2
    col = i % 2
    sns.countplot(x=cat, hue='target', data=Heart_Disease, ax=axes[row, col], palette=sns.cubehelix_palette(start=2))
    axes[row, col].set_title(f"Target vs. {cat}")

fig.suptitle("Count Plot of Target vs. Categorical Values", fontweight='bold')
plt.tight_layout()
plt.show()


# # 

# # ***`Modeling`***

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[19]:


# Splitting data into features (x) and target (y)

x = Heart_Disease.drop("target", axis=1)

y = Heart_Disease["target"]


# In[20]:


# Splitting the data into train and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=50)


# In[21]:


# Data Normalization using Min-Max Method

x = MinMaxScaler().fit_transform(x)


# # 

# # ***`Model Implementation`***

# **`Logistic Regression`**

# In[22]:


LR_classifier = LogisticRegression(max_iter=1000, random_state=1, 
                                  solver='liblinear', penalty='l1').fit(
    x_train, y_train)

y_pred_LR = LR_classifier.predict(x_test)


# In[23]:


LR_Accuracy = accuracy_score(y_pred_LR, y_test)

print('Logistic Regression Accuracy:'+'\033[1m {:.2f}%'.format(LR_Accuracy*100))


# **`K-Nearest Neighbour (KNN)`**

# In[24]:


KNN_Classifier = KNeighborsClassifier(n_neighbors=3).fit(
    x_train, y_train)

y_pred_KNN = KNN_Classifier.predict(x_test)


# In[25]:


KNN_Accuracy = accuracy_score(y_pred_KNN, y_test)

print('K-Nearest Neighbour Accuracy:'+'\033[1m {:.2f}%'.format(KNN_Accuracy*100))


# **`Support Vector Machine (SVM)`**

# In[26]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[27]:


SVM_Classifier = SVC(kernel='linear', max_iter=5000, C=10, probability=True).fit(
    x_train_scaled, y_train)

y_pred_SVM = SVM_Classifier.predict(x_test_scaled)


# In[28]:


SVM_Accuracy = accuracy_score(y_pred_SVM, y_test)

print('Support Vector Machine Accuracy:', '\033[1m{:.2f}%'.format(SVM_Accuracy * 100))


# **`Decision Tree`**

# In[29]:


DT_Classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', min_samples_split=5,
                                       splitter='random', random_state=1).fit(
    x_train, y_train)

y_pred_DT = DT_Classifier.predict(x_test)


# In[30]:


DT_Accuracy = accuracy_score(y_pred_DT, y_test)

print('Decision Tree Accuracy:', '\033[1m{:.2f}%'.format(DT_Accuracy * 100))


# **`Random Forest`**

# In[31]:


RF_Classifier = RandomForestClassifier(n_estimators=1000, random_state=1, 
                                       max_leaf_nodes=20, min_samples_split=15).fit(
    x_train, y_train)

y_pred_RF = RF_Classifier.predict(x_test)


# In[32]:


RF_Accuracy = accuracy_score(y_pred_RF, y_test)

print('Random Forest Accuracy:'+'\033[1m {:.2f}%'.format(RF_Accuracy*100))


# # 

# # ***`Model Comparison`***

# In[33]:


compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbors',
                                  'Support Vector Machine', 'Decision Tree', 'Random Forest'], 
                        'Accuracy': [
                            LR_Accuracy * 100, KNN_Accuracy * 100, SVM_Accuracy * 100, 
                            DT_Accuracy * 100, RF_Accuracy * 100]})


# In[34]:


# Sorting and styling the comparison table

compare_sorted = compare.sort_values(by='Accuracy', 
                                     ascending=False).reset_index(drop=True)
compare_styled = compare_sorted.style.background_gradient(
    cmap='Blues', subset=['Accuracy']).set_properties(**{'font-family': 'Segoe UI'})


# In[35]:


compare_styled

