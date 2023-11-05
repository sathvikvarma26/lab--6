import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatasheet-1.xlsx")
df
import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_excel("embeddingsdatasheet-1.xlsx")

#Loading 2 features having numeric values
feature_a = df['embed_0']
feature_b = df['embed_1']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(feature_a, feature_b, color='blue', alpha=0.5)

# Add labels and title
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.title('Scatter Plot of embed_0 vs embed_1')

# Show plot
plt.grid(True)
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("embeddingsdatasheet-1.xlsx")

# Assume 'embed_1' is the independent variable
independent_variable = df['embed_1']
dependent_variable = df['embed_0']

# Reshape the data as sklearn's LinearRegression model expects 2D array
independent_variable = independent_variable.values.reshape(-1, 1)
dependent_variable = dependent_variable.values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()
model.fit(independent_variable, dependent_variable)

# Predict the values
predicted_values = model.predict(independent_variable)

# Calculate mean squared error
mse = mean_squared_error(dependent_variable, predicted_values)
print(f'Mean Squared Error: {mse}')

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(independent_variable, dependent_variable, color='blue', alpha=0.5)

# Plot the regression line
plt.plot(independent_variable, predicted_values, color='red')

# Add labels and title
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.title('Linear Regression Model: embed_0 vs embed_1')

# Show plot
plt.grid(True)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Load the dataset
df = pd.read_excel("embeddingsdatasheet-1.xlsx")

# Assuming X_train, y_train, X_test, y_test are your training and test sets

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model on the training data
logistic_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
predictions = logistic_model.predict(X_test)

# Calculate accuracy by comparing predicted labels to actual labels in the test set
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of Logistic Regression on the test set: {accuracy * 100:.2f}%")
