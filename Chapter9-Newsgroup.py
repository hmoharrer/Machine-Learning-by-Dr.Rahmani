#!/usr/bin/env python
# coding: utf-8

# In[36]:



from sklearn.datasets import fetch_20newsgroups


groups = fetch_20newsgroups()
groups.keys()
groups['target_names']
groups.target


import numpy as np
np.unique(groups.target)



import seaborn as sns
sns.displot(groups.target)
import matplotlib.pyplot as plt
plt.show()



# In[37]:



groups.data[7]


# In[38]:


groups.target[7]


# In[39]:


groups.target_names[groups.target[7]]


# In[ ]:




