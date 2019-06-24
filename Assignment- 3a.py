#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with one variable

# # 1. EDA

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv(r'C:\Users\user\Documents\Summer Analytics\week-3\data.txt', header = None)
data.head()


# In[16]:


x = data.iloc[:,0]
y = data.iloc[:,1]


# In[17]:


plt.scatter(x,y)
plt.title("Profit vs Population")
plt.xlabel('Population in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


# This shows there is a linear relationship between the variables

# # 2. Initializing the parameters

# In[15]:


theta = np.zeros(2)


# In[19]:


X = np.ones(shape=(len(x), 2))
X[:, 1] = x


# In[20]:


predictions = X @ theta


# # Cost Function

# In[27]:


def cost(theta, X, y):
    predictions = X @ theta
    squared_errors = np.square(predictions - y)
    return np.sum(squared_errors) / (2 * len(y))


# In[28]:


print('The initial cost is:', cost(theta, X, y))


# # Gradient Descent 
# 

# In[30]:


def gradient_descent(X, y, alpha, num_iters):
    num_features = X.shape[1]               
    theta = np.zeros(num_features)          
    for n in range(num_iters):
        predictions = X @ theta             
        errors = predictions - y
        gradient = X.transpose() @ errors
        theta -= alpha * gradient / len(y) 
    return theta                            


# # Best fit line

# In[34]:



theta = gradient_descent(X, y, 0.02, 600)  
predictions = X @ theta                     
plt.plot(X[:, 1], predictions, linewidth=2)  


# In[36]:


def gradient_descent(X, y, alpha, num_iters):
    cost_history = np.zeros(num_iters)          
    num_features = X.shape[1]               
    theta = np.zeros(num_features)
    for n in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = X.transpose() @ errors
        theta -= alpha * gradient / len(y)
        cost_history[n] = cost(theta, X, y)     
    return theta, cost_history       


# In[37]:


plt.figure()
num_iters = 1200
learning_rates = [0.01, 0.015, 0.02]
for lr in learning_rates:
    _, cost_history = gradient_descent(X, y, lr, num_iters)
    plt.plot(cost_history, linewidth=2)
plt.title("Gradient descent with different learning rates")
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.legend(list(map(str, learning_rates)))
plt.axis([0, num_iters, 4, 6])
plt.show()


# # Profit for population of 70,000

# In[38]:


theta, _ = gradient_descent(X, y, 0.02, 600)    
test_example = np.array([1, 7])                 
prediction = test_example @ theta              
print('For population = 70,000, we predict a profit of $', prediction * 10000);


# # Predictions with sklearn Library

# In[39]:


from sklearn.linear_model import LinearRegression


# In[43]:



data = pd.read_csv(r'C:\Users\user\Documents\Summer Analytics\week-3\data.txt', header = None)
data.columns = ['population','profit']
data.head()


# In[47]:


regr = LinearRegression()
train_x = np.asanyarray(data[['population']])
train_y = np.asanyarray(data[['profit']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[48]:


plt.scatter(data.population, data.profit,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("population")
plt.ylabel("profit")


# In[53]:


profit = regr.coef_[0][0]*3.5 + regr.intercept_[0]
print('profit for 35,000 population:$',profit*10000)


# In[54]:


profit = regr.coef_[0][0]*7 + regr.intercept_[0]
print('profit for 70,000 population:$',profit*10000)


# In[ ]:




