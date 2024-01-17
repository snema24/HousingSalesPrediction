#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

housing_data = pd.read_csv("kc_house_data.csv")
X_new = housing_data.drop('date', axis = 1)  # Features
X = X_new.drop('price', axis = 1)  # Features
y = housing_data['price'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

len(y_train)
housing_data


# In[4]:


housing_data.info()


# In[3]:


housing_data.isnull().sum()


# In[6]:


housing_data['sqft_living'].describe()


# In[18]:


housing_data['sqft_lot'].describe()


# In[19]:


housing_data['price'].corr(housing_data['sqft_living'])


# In[20]:


housing_data['price'].corr(housing_data['sqft_lot'])


# In[5]:


plt.subplot(2, 3, 1)
plt.hist(housing_data["condition"], bins=30, color='blue', alpha=0.7)
plt.title("Counts of condition rating")

plt.subplot(2, 3, 2)
plt.hist(housing_data["bedrooms"], bins=30, color='green', alpha=0.7)
plt.title('Counts of bedrooms')

plt.subplot(2, 3, 3)
plt.hist(housing_data["floors"], bins=30, color='red', alpha=0.7)
plt.title("Counts of floors")

plt.subplot(2, 3, 4)
plt.hist(housing_data["bathrooms"], bins=30, color='purple', alpha=0.7)
plt.title('Counts of bathrooms')

plt.subplot(2, 3, 5)
plt.hist(housing_data["sqft_living"], bins=30, color='orange', alpha=0.7)
plt.title('Counts of sqft_living')

plt.subplot(2, 3, 6)
plt.hist(housing_data["sqft_lot"], bins=30, color='cyan', alpha=0.7)
plt.title("Counts of sqft_lot")


plt.tight_layout()


# In[49]:


plt.hist(housing_data["price"], bins=30, color='cyan', alpha=0.7)
plt.title("Counts of price in 100 thousands")


# In[48]:


plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.scatter(x = housing_data["sqft_living"], y = housing_data["price"], s = 10)
plt.xlabel("sqft_living")
plt.ylabel("price in 100 thousands")
plt.title("Square foot living vs Price")

plt.subplot(2,1,2)
plt.scatter(x = housing_data["sqft_lot"], y = housing_data["price"], s = 10)
plt.title("Square foot lot vs Price")
plt.xlabel("sqft_lot")
plt.ylabel("price in 100 thousands")

plt.tight_layout()


# In[23]:


plt.figure(figsize=(15, 10))  # Set the figure size

plt.subplot(2,1,1)
plt.boxplot(housing_data['sqft_living'], vert=False)  # vert=False for horizontal boxplot
plt.title('Boxplot for Square foot living')
plt.xlabel('Square foot living')
plt.yticks([])  # Hide y-axis ticks

plt.tight_layout()


# In[20]:


from sklearn import metrics
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)
y_test_pred = regr.predict(X_test)
print(f'R-Squared: {r2_score(y_train, y_train_pred)}')
print(f'R-Squared: {r2_score(y_test, y_test_pred)}')


# In[13]:


from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#100000, 1000000]

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# Initialize lists to store coefficients and R2 values
coefficients = []
r2_values = []

for alpha in alphas:
    # Train the Lasso regression model
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    
    # Get the coefficients
    coefficients.append(lasso.coef_)
    
    # Make predictions on the test set
    y_pred = lasso.predict(X_test)
    
    # Calculate the R2 value
    r2 = r2_score(y_test, y_pred)
    r2_values.append(r2)

# Print the coefficients and R2 values for each alpha
for alpha, coef, r2 in zip(alphas, coefficients, r2_values):
    print(f"Alpha: {alpha}, Coefficients: {coef}, R2: {r2}")
    


# In[16]:


from sklearn.preprocessing import PolynomialFeatures
cubic = PolynomialFeatures(3, include_bias = False)
X_train_quadratic = cubic.fit_transform(X_train)
X_test_quadratic = cubic.transform(X_test)

# Fit a linear regression model to the quadratic features
model = linear_model.LinearRegression()
model.fit(X_train_quadratic, y_train)

# Predict on both training and testing data
y_train_pred = model.predict(X_train_quadratic)
y_test_pred = model.predict(X_test_quadratic)

# Calculate R2 values
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"R2 on the training set: {r2_train}")
print(f"R2 on the testing set: {r2_test}")


# In[9]:


from sklearn.ensemble import GradientBoostingRegressor 
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(r2_train)
print(r2_test)


# In[ ]:




