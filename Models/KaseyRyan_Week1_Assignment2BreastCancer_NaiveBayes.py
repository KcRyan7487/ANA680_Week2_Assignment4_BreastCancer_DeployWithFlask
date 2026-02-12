#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


# After some initial confusion due to the first link on the assignment leading to adifferent incorrect file, I was able to figure out how to bring it in
# Basically had to use the second link to go straight to the source and examine the files from the downloaded zip. There are 8 total but we only need 1
# Or I guess technically we need 2 (the .data file is the main one but the .names file has some read-me type junk and the column headers in a list)
# Just manually extracting the column headers and will apply them here as headers to the data only csv rows from the main breast-cancer-wisconsin.data file

import pandas as pd
# Below taken from the breast-cancer-wisconsin.names file from the zip from the download direct from the site linked in the 2nd link in the assignment
column_names = [
    "id",
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"
]

# Further updates... while exporting these from .ipynb in the notebook to actual .py files for each model (which I THINK is what the assignment wants)
# I was running into issues due to my instance of VSCode running in a different directory versus the Jupyter Notebook
# Furthermore the method I used in VSCode to get it running from start to end without error (using __file__ to build a relative data path from the model file)
# Does not work on the Jupyter notebook side. I want the code to work in both platforms seamlessly and so modifying this a bit further
from pathlib import Path
try:
    # Running as a script (.py)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]   #Step up a level outside of my models folder to the project's root folder
    DATA_PATH = PROJECT_ROOT / "Data" / "breast-cancer-wisconsin.data"

except NameError:
    # It'll throw an error on __file__ if running in a notebook (.ipynb) so then just set it the way it was working before
    DATA_PATH = Path("Data/breast-cancer-wisconsin.data")

data = pd.read_csv(
    DATA_PATH,
    header=None,
    names=column_names
)

data.head()


# In[3]:


#No need for the IDs, these shouldn't be present in a real model as they won't provide meaningful explanatory power and worse could cause problems etc
data = data.drop(columns=["id"])


# In[4]:


# the missing values are ? in the .data file but let's change them to true NA's instead
data = data.replace("?", pd.NA)

# Now let's get an idea of how many rows there are with 
na_counts = data.isna().sum()
na_counts
# looks like the only feature which is sometimes missing values (Rarely) is bare_nuclei


# In[5]:


rows_with_any_na = data.isna().any(axis=1).sum()
total_rows = data.shape[0]

rows_with_any_na, total_rows
#Only 16 out of 699 rows (aligns with the 16 expectation since only one feature was missing values and it had 16 missing)


# In[6]:


#Okay it's very few so we'll just drop the 16 rows with NAs then
data = data.dropna()
data.shape



# In[7]:


# separate into features (x) and class labels (y)
X = data.drop(columns=["class"])
y = data["class"]
X.shape, y.value_counts()



# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
X_train.shape, X_test.shape


# In[9]:


#Just important eveyrthing we'll need for all 8 of the modules here, so that we can make minimal changes accross the 8 files the assignment asks for
#(Assignment instructions want 1 file per model so I'll copy paste the code for 1 accross all of them and make only the minimal changes required to swap models)
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional / environment-dependent
# from xgboost import XGBClassifier
warnings.filterwarnings("ignore")



# In[10]:


# Build the model (and I set a variable so that the accuracy statements at the end can write out which model is evaluated as well)
model_name = "Naive Bayes"
model = GaussianNB()

#Fit/train the model and run the predictions on the test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Actual: Benign (2)", "Actual: Malignant (4)"],
    columns=["Predicted: Benign (2)", "Predicted: Malignant (4)"]
)

print(f"{model_name} Accuracy:", accuracy)
print(f"{model_name} Confusion Matrix:")
print(conf_matrix_df)






# In[ ]:




