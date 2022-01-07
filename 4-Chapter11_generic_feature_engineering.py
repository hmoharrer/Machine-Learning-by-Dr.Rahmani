#!/usr/bin/env python
# coding: utf-8

# In[3]:



#Best practice 12 - page 361- 362
from sklearn.preprocessing import Binarizer

X = [[4], [1], [3], [0],[0.5],[5],[1.6]]
binarizer = Binarizer(threshold=2.9)
X_new = binarizer.fit_transform(X)
print(X_new)


from sklearn.preprocessing import PolynomialFeatures

X = [[2, 4],
     [1, 3],
     [3, 2],
     [0, 3]]
poly = PolynomialFeatures(degree=2)
X_new = poly.fit_transform(X)
print(X_new)


# In[ ]:




