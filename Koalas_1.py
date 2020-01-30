# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://raw.githubusercontent.com/databricks/koalas/master/Koalas-logo.png" width="220"/>
# MAGIC </div>
# MAGIC 
# MAGIC The Koalas project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark. By unifying the two ecosystems with a familiar API, Koalas offers a seamless transition between small and large data.
# MAGIC 
# MAGIC **Goals of this notebook:**
# MAGIC * Demonstrate the similarities of the Koalas API with the pandas API
# MAGIC * Understand the differences in syntax for the same DataFrame operations in Koalas vs PySpark
# MAGIC 
# MAGIC [Koalas Docs](https://koalas.readthedocs.io/en/latest/index.html)
# MAGIC 
# MAGIC [Koalas Github](https://github.com/databricks/koalas)
# MAGIC 
# MAGIC **Requirements:**
# MAGIC * `DBR 6.0 ML`
# MAGIC * `koalas==0.25.0` 
# MAGIC 
# MAGIC **Data:**
# MAGIC * *[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014*

# COMMAND ----------

# MAGIC %md
# MAGIC We will be using the [UCI Machine Learning Repository 
# MAGIC Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing) throughout this demo.

# COMMAND ----------

# MAGIC %sh wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip -O /tmp/bank.zip

# COMMAND ----------

# MAGIC %sh unzip -o /tmp/bank.zip -d /tmp/bank

# COMMAND ----------

file_path = "/bank-full.csv"
dbutils.fs.cp("file:/tmp/bank/bank-full.csv", file_path)
dbutils.fs.head(file_path)

# COMMAND ----------

# MAGIC %fs ls bank-full.csv

# COMMAND ----------

# MAGIC %md ### Loading the dataset as a Spark Dataframe

# COMMAND ----------

df = (spark.read
           .option("inferSchema", "true")
           .option("header", "true")
           .option("delimiter", ";")
           .option("quote", '"')
           .csv(file_path))

display(df)

# COMMAND ----------

# MAGIC %md ### Loading the dataset as a pandas Dataframe

# COMMAND ----------

import pandas as pd

csv_path = "/dbfs/bank-full.csv"

# Read in using pandas read_csv
pdf = pd.read_csv(csv_path, header=0, sep=";", quotechar='"')
display(pdf.head())

# COMMAND ----------

# MAGIC %md ### Loading the dataset as a Koalas Dataframe 

# COMMAND ----------

# Import Koalas
import databricks.koalas as ks
import warnings
warnings.filterwarnings("ignore")

# Read in using Koalas read_csv
kdf = ks.read_csv(file_path, header=0, sep=";", quotechar='"')

display(kdf.head())

# COMMAND ----------

# Converting to Koalas Dataframe from Spark DataFrame

# Creating a Koalas DataFrame from PySpark DataFrame
# kdf = ks.DataFrame(df)

# # Alternative way of creating a Koalas DataFrame from PySpark DataFrame
kdf = df.to_koalas()

# COMMAND ----------

# MAGIC %md
# MAGIC Note that calling `.head()` in Koalas may not return return the same results as pandas here. Unlike pandas, the data in a Spark dataframe is not ordered - it has no intrinsic notion of index. When asked for the head of a DataFrame, Spark will just take the requested number of rows from a partition. Do not rely on it to return specific rows, instead use `.loc` or `iloc`.

# COMMAND ----------

# MAGIC %md ### Indexing Rows

# COMMAND ----------

pdf.iloc[:3]

# COMMAND ----------

kdf.iloc[3]

# COMMAND ----------

# MAGIC %md Using a scalar integer for row selection is not allowed in Koalas, instead we must supply either a *slice* object or boolean condition.

# COMMAND ----------

kdf.iloc[:4]

# COMMAND ----------

display(df.limit(4))

# COMMAND ----------

# Koalas Dataframe -> PySpark DataFrame
display(kdf.to_spark())

# COMMAND ----------

# Getting the number of rows and columns in PySpark
print((df.count(), len(df.columns)))

# COMMAND ----------

# Getting the number of rows and columns in Koalas
print(kdf.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Descriptive Stats

# COMMAND ----------

kdf.head()

# COMMAND ----------

kdf.describe()

# COMMAND ----------

# MAGIC %md ### Column Manipulation

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's have a look at some column operations. Suppose we want to create a new column, where each last contact duration (the `duration` column) is 100 greater than the original duration entry. We will call this new column `duration_new`.

# COMMAND ----------

# Creating a column with PySpark
from pyspark.sql.functions import col

df = df.withColumn("duration_new", col("duration") + 100)
display(df)

# COMMAND ----------

# Creating a column with Koalas
kdf["duration_new"] = kdf["duration"] + 100
display(kdf.head())

# COMMAND ----------

# MAGIC %md ###Filtering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's now count the number of instances where `duration_new` is greater than or equal to 300.

# COMMAND ----------

# Filtering with PySpark
df_filtered  = df.filter(col("duration_new") >= 300)
print(df_filtered.count())

# COMMAND ----------

# Filtering with Koalas
kdf_filtered = kdf[kdf.duration_new >= 300]
print(kdf_filtered.shape[0])

# COMMAND ----------

# MAGIC %md ### Value Counts

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Suppose we want to have a look at the number of clients for each unique job type.

# COMMAND ----------

# To get value counts of the different job types with PySpark
display(df.groupby("job").count().orderBy("count", ascending=False))

# COMMAND ----------

# Value counts in Koalas
kdf["job"].value_counts()

# COMMAND ----------

# MAGIC %md ###GroupBy 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Let's compare group by operations in PySpark versus Koalas. We will create two DataFrames grouped by education, to get the average `age` and maximum `balance` for each education group.

# COMMAND ----------

# Get average age per education group using PySpark
df_grouped_1 = (df.groupby("education")
                .agg({"age": "mean"})
                .select("education", col("avg(age)").alias("avg_age")))

display(df_grouped_1)

# COMMAND ----------

# Get the maximum balance for each education group using PySpark
df_grouped_2 = (df.groupby("education")
                .agg({"balance": "max"})
                .select("education", col("max(balance)").alias("max_balance")))

display(df_grouped_2)

# COMMAND ----------

# Get the average age per education group in Koalas
kdf_grouped_1 = kdf.groupby("education", as_index=False).agg({"age": "mean"})

# Rename our columns
kdf_grouped_1.columns = ["education", "avg_age"]
display(kdf_grouped_1)

# COMMAND ----------

# Get the maximum balance for each education group in Koalas
kdf_grouped_2 = kdf.groupby("education", as_index=False).agg({"balance": "max"})
kdf_grouped_2.columns = ["education", "max_balance"]
display(kdf_grouped_2)

# COMMAND ----------

# MAGIC %md ### Joins

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Let's now look at doing an inner join between our grouped DataFrames, on the `education` attribute.

# COMMAND ----------

# Joining the grouped DataFrames on education using PySpark
df_edu_joined = df_grouped_1.join(df_grouped_2, on="education", how="inner")
display(df_edu_joined)

# COMMAND ----------

# Joining the grouped DataFrames on education using Koalas
kdf_edu_joined = kdf_grouped_1.merge(kdf_grouped_2, on="education", how="inner")
display(kdf_edu_joined)

# COMMAND ----------

# MAGIC %md ### Writing Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Finally, let's save our joined DataFrames as Parquet files. [Parquet](https://parquet.apache.org/) is an efficient and compact file format to read and write faster.

# COMMAND ----------

# Saving the Spark DataFrame as a Parquet file.
spark_out_path = "/dbfs/bank_grouped_pyspark.parquet"

df_edu_joined.write.mode("overwrite").parquet(spark_out_path)

# COMMAND ----------

# Saving the Koalas DataFrame as a Parquet file.
koalas_out_path = "/dbfs/bank_grouped_koalas.parquet"

kdf.to_parquet(koalas_out_path, mode="overwrite")
