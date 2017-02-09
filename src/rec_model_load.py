import pyspark as ps
from pyspark.sql.types import *
from pyspark.ml.recommendation import *

import pyspark.ml.evaluation
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.evaluation import *
import numpy as np
import pandas as pd

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .master("yarn")
    .appName("artistrecommender")
    .getOrCreate())

#    .config("spark.executor.memory", "100g")

## Data frame loads

artist_follower_df = spark.read.csv('s3a://salstuff.com/data/artist_follower.csv')
artist_df = spark.read.csv('s3a://salstuff.com/data/artist_meta.csv')
follower_df = spark.read.csv('s3a://salstuff.com/data/followers.csv')

artist_follower_df.createOrReplaceTempView('artist_follower')
artist_df.createOrReplaceTempView('artists')
follower_df.createOrReplaceTempView('followers')

## Artist & Follower column name generation

artist_follower_df = spark.sql("""
SELECT distinct
    /*
    _c0 as exception
    ,_c1 as artist_num
    ,_c2 as artist_follower_num
    ,*/
    _c3 as artist_alias
    ,_c4 as follower_alias
FROM artist_follower where _c2 != 'artist_follower_num'
""").persist()

artist_follower_df.createOrReplaceTempView('artist_follower')

#0 means the artist is not a duplicate record
artist_df = spark.sql("""
SELECT distinct
    _c0 as index
    ,_c1 as id
    ,_c2 as artist_name
    ,_c3 as sc_alias
    ,_c4 as city
    ,_c5 as country
    ,_c6 as followers_count
    ,_c7 as followings_count
    ,_c8 as last_modified
    ,_c9 as playlist_count
    ,_c10 as plan
    ,_c11 as public_favorites_count
    ,_c12 as track_count
    ,_c13 as count
FROM artists where _c2 != 'artist_name'
AND _c14 = 0
AND _c2 != 'Daft Punk'
""").persist()

artist_df.createOrReplaceTempView('artists')

follower_df = spark.sql("""
SELECT distinct
    --_c0 as exception
    --,_c1 as artist_num
    --,_c2 as artist_follower_num
    _c3 as artist_alias
    ,_c4 as follower_alias
    ,_c5 as comments_count
    ,_c6 as followers_count
    ,_c7 as followings_count
    --,_c8 as last_modified
    ,_c9 as likes_count
    ,_c10 as plan
    ,_c11 as playlist_count
    ,_c12 as public_favorites_count
    ,_c13 as reposts_count
    ,_c14 as track_count
    ,_c15 as uri
    ,_c16 as username
FROM followers where _c0 != 'exception'
""").persist()

follower_df.createOrReplaceTempView('followers')

## Artist & Follower id generation

custom_artist_id_df = spark.sql("""
select distinct
    row_number() over(order by sc_alias) as artist_id
    ,sc_alias as artist_alias
from artists
group by sc_alias
""").persist()

custom_artist_id_df.createOrReplaceTempView('custom_artist_id')

custom_follower_id_df = spark.sql("""
select distinct
    row_number() over(order by follower_alias) as follower_id
    ,follower_alias
from followers
group by follower_alias
""").persist()

custom_follower_id_df.createOrReplaceTempView('custom_follower_id')

artist_follower_ids_df = spark.sql("""
SELECT DISTINCT
    caid.artist_id
    ,cfid.follower_id
    ,af.artist_alias
    ,af.follower_alias
    ,1 as count
FROM artist_follower af
JOIN custom_artist_id caid on af.artist_alias = caid.artist_alias
JOIN custom_follower_id cfid on af.follower_alias = cfid.follower_alias
sort by af.artist_alias
""").persist()

als = ALS(rank=5, maxIter=5, seed=0, regParam=1, implicitPrefs=True, userCol="follower_id", itemCol="artist_id", ratingCol="count", nonnegative=True)

# training, test = artist_follower_ids_df.randomSplit([0.8, 0.2], seed = 1)

model = als.fit(artist_follower_ids_df)

# predictions = model.transform(test)
#
# predictions.registerTempTable("predictions")

# print spark.sql("SELECT * FROM predictions WHERE NOT ISNAN(prediction) ORDER BY prediction DESC").show()

# paramGrid = ParamGridBuilder()\
#     .addGrid(als.rank, [5, 10, 20])\
#     .addGrid(als.regParam, [.01, .1, 1])\
#     .build()
#
# evaluator = RegressionEvaluator(metricName="rmse", labelCol="count")
#
# cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator)
#
# cvModel = cv.fit(training)
#
# best = cvModel.bestModel
#
# regParam = (best
#             ._java_obj     # Get Java object
#             .parent()      # Get parent (ALS estimator)
#             .getRegParam()) # Get maxIter
#
# rank = best.rank
#
# print 'regParam: ', regParam, 'rank: ', rank
#
# items = best.itemFactors

items = model.itemFactors

df = items.toPandas()

df.to_csv('bestModelItems.csv')

print df.head()
