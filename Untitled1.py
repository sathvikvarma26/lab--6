#!/usr/bin/env python
# coding: utf-8

# In[3]:


# In[1]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatasheet-1.xlsx")
df


# In[4]:


import matplotlib.pyplot as plt


feature1 = df['embed_1']
feature2 = df['embed_2']

plt.scatter(feature1, feature2)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Scatter Plot of Feature1 vs Feature2')
plt.show()


# In[5]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Select one feature as the independent variable and the other as the dependent variable
independent_feature = np.array(df['embed_1']).reshape(-1, 1)
dependent_feature = np.array(df['embed_2'])

# Create a Linear Regression model
model = LinearRegression()
model.fit(independent_feature, dependent_feature)

# Predict the values
predicted_values = model.predict(independent_feature)

# Calculate the mean squared error
mse = mean_squared_error(dependent_feature, predicted_values)
print(f"Mean Squared Error: {mse:.2f}")


# In[6]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Select one feature as the independent variable and the other as the dependent variable
independent_feature = np.array(df['embed_1']).reshape(-1, 1)
dependent_feature = np.array(df['embed_2'])

# Create a Linear Regression model
model = LinearRegression()
model.fit(independent_feature, dependent_feature)

# Predict the values
predicted_values = model.predict(independent_feature)

# Calculate the mean squared error
mse = mean_squared_error(dependent_feature, predicted_values)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the linear regression line
plt.scatter(independent_feature, dependent_feature, label='Data points')
plt.plot(independent_feature, predicted_values, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()


# In[7]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a DataFrame from the data

# Create the linear regression model with Feature1 as the independent variable
model_feature1 = LinearRegression()
model_feature1.fit(df[['embed_1']], df['embed_2'])  # Feature2 as the dependent variable

# Predict the values using the model
predictions_feature1 = model_feature1.predict(df[['embed_1']])

# Calculate the mean squared error for Feature1 as the independent variable
mse_feature1 = mean_squared_error(df['embed_2'], predictions_feature1)
print(f"Mean Squared Error (Feature1 as independent variable): {mse_feature1:.2f}")

# Create the linear regression model with Feature2 as the independent variable
model_feature2 = LinearRegression()
model_feature2.fit(df[['embed_2']], df['embed_1'])  # Feature1 as the dependent variable

# Predict the values using the model
predictions_feature2 = model_feature2.predict(df[['embed_2']])

# Calculate the mean squared error for Feature2 as the independent variable
mse_feature2 = mean_squared_error(df['embed_1'], predictions_feature2)
print(f"Mean Squared Error (Feature2 as independent variable): {mse_feature2:.2f}")


# In[9]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_excel("embeddingsdatasheet-1.xlsx")
# Load your dataset
# Assuming 'X' contains your feature vectors and 'y' contains the corresponding labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the logistic regression model
logistic_model = LogisticRegression(solver='liblinear', C=1.0, penalty='l2', max_iter=100)

# Train the model on the training data
logistic_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:




