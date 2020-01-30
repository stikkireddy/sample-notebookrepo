# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC # Koalas: Pandas on Spark
# MAGIC 
# MAGIC ## Predicting Response Time for the San Francisco Fire Department
# MAGIC 
# MAGIC *Data Science Workflow:*
# MAGIC 0. Read Data
# MAGIC 0. Exploratory Data Analysis (EDA)
# MAGIC 0. Featurization
# MAGIC 
# MAGIC <br>
# MAGIC <img src="https://files.training.databricks.com/images/fire-koala.jpg" style="height: 200px; border: 1px solid #ddd; border-radius: 15px 15px 15px 15px;"/>
# MAGIC <img src="https://files.training.databricks.com/images/fire-koala.jpg" style="height: 200px; border: 1px solid #ddd; border-radius: 15px 15px 15px 15px;"/>
# MAGIC <img src="https://files.training.databricks.com/images/fire-koala.jpg" style="height: 200px; border: 1px solid #ddd; border-radius: 15px 15px 15px 15px;"/>
# MAGIC 
# MAGIC Data is available from [Open Data SF](https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3).

# COMMAND ----------

# DBTITLE 1,Reading in CSV (Pandas)
import pandas as pd

data = pd.read_csv("/dbfs/databricks-datasets/sai-summit-2019-sf/fire-calls.csv", header=0)
data = data[["CallType", "Neighborhood", "NumAlarms", "OrigPriority", "UnitType", "Delay"]]
display(data.head())

# COMMAND ----------

# DBTITLE 1,Reading in CSV (Koalas)
from databricks import koalas

data = koalas.read_csv("dbfs:/databricks-datasets/sai-summit-2019-sf/fire-calls.csv", header=0)
data = data[["CallType", "Neighborhood", "NumAlarms", "OrigPriority", "UnitType", "Delay"]]
display(data.head())

# COMMAND ----------

# DBTITLE 1,Most Common Call Types? (Unified Syntax)
data["CallType"].value_counts()

# COMMAND ----------

# DBTITLE 1,Create Temporary View (to use in SQL below)
data.to_spark().createOrReplaceTempView("FireCalls")

# COMMAND ----------

# DBTITLE 1,Average Time Delay by Call Type? (SQL)
# MAGIC %sql
# MAGIC SELECT CallType, avg(Delay) as avgDelay
# MAGIC FROM FireCalls
# MAGIC GROUP BY CallType
# MAGIC ORDER BY avgDelay DESC

# COMMAND ----------

# DBTITLE 1,Feature Engineering: One Hot Encoding
# MAGIC %md
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ohe-calltype.jpg"/>

# COMMAND ----------

display(koalas.get_dummies(data))