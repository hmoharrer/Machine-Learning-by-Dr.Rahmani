#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession             .builder             .appName("test")             .getOrCreate()


# In[3]:



df = spark.read.csv("people.csv", header=True, sep=';')
df.show()


# In[4]:


df.count()


# In[5]:


df.printSchema()


# In[6]:





df.select("name").show()
df.select(["name", "job"]).show()






# In[7]:


df.filter(df['age'] > 31).show()


# In[8]:


from pyspark.sql.functions import monotonically_increasing_id
df.withColumn('index', monotonically_increasing_id())

