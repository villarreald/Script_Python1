#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

df = pd.read_csv (sys.argv[1])
print (df)


# In[10]:


import matplotlib.pyplot as plt

x = df.x

y = df.y

plt.scatter(x,y)
plt.title('Regrex Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig("scatter.png")


# In[11]:


import numpy as np
from sklearn.linear_model import LinearRegression
X = df.x.to_numpy()
X = X.reshape(-1, 1)
y = df.y.to_numpy()
y = y.reshape(-1, 1)
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
y_predict = reg.predict(X)


# In[14]:


plt.scatter(x,y)
plt.plot(X,y_predict)
plt.title('Regrex Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig("fit.png")


# In[ ]:




