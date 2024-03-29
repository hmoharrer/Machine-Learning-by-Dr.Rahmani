#!/usr/bin/env python
# coding: utf-8

# In[2]:



from pyspark.sql import SparkSession


spark = SparkSession    .builder    .appName("CTR")    .getOrCreate()





from pyspark.sql.types import StructField, StringType, StructType, IntegerType

schema = StructType([
    StructField("id", StringType(), True),
    StructField("click", IntegerType(), True),
    StructField("hour", IntegerType(), True),
    StructField("C1", StringType(), True),
    StructField("banner_pos", StringType(), True),
    StructField("site_id", StringType(), True),
    StructField("site_domain", StringType(), True),
    StructField("site_category", StringType(), True),
    StructField("app_id", StringType(), True),
    StructField("app_domain", StringType(), True),
    StructField("app_category", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("device_ip", StringType(), True),
    StructField("device_model", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("device_conn_type", StringType(), True),
    StructField("C14", StringType(), True),
    StructField("C15", StringType(), True),
    StructField("C16", StringType(), True),
    StructField("C17", StringType(), True),
    StructField("C18", StringType(), True),
    StructField("C19", StringType(), True),
    StructField("C20", StringType(), True),
    StructField("C21", StringType(), True),
])



df = spark.read.csv("train3.csv", schema=schema, header=True)


df = df.drop('id').drop('hour').drop('device_id').drop('device_ip')

df = df.withColumnRenamed("click", "label")




df_train, df_test = df.randomSplit([0.7, 0.3], 42)

df_train.cache()

df_test.cache()



categorical = df_train.columns
categorical.remove('label')
print(categorical)



cat_inter = ['C14', 'C15']

concat = '+'.join(categorical)
interaction = ':'.join(cat_inter)
formula = "label ~ " + concat + '+' + interaction

print(formula)

from pyspark.ml.feature import RFormula
interactor = RFormula(
    formula=formula,
    featuresCol="features",
    labelCol="label").setHandleInvalid("keep")

interactor.fit(df_train).transform(df_train).select("features").show()

from pyspark.ml.classification import LogisticRegression

classifier = LogisticRegression(maxIter=20, regParam=0.000, elasticNetParam=0.000)

stages = [interactor, classifier]

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=stages)


model = pipeline.fit(df_train)

predictions = model.transform(df_test)


predictions.cache()

predictions.show()


from pyspark.ml.evaluation import BinaryClassificationEvaluator

ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", metricName = "areaUnderROC")
print(ev.evaluate(predictions))


spark.stop()

