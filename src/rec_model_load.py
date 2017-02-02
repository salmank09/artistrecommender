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
# from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
# from scipy.spatial.distance import cdist
import pandas as pd

# import time

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark.sql import SparkSession
#
# def computeRmse(model, data, n):
#     """
#     Compute RMSE (Root Mean Squared Error).
#     """
#     predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
#     predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
#       .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
#       .values()
#     return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

spark = (SparkSession.builder
    .master("yarn")
    # .config("spark.driver.cores", 2)
    # .config("spark.driver.memory", "16g")
    # .config("spark.executor.memory", "16g")
    # .config("spark.executor.instances", "4")
    # .config("spark.executor.")
    .appName("artistrecommender")
    .getOrCreate())

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
""")

custom_artist_id_df.createOrReplaceTempView('custom_artist_id')

custom_follower_id_df = spark.sql("""
select distinct
    row_number() over(order by follower_alias) as follower_id
    ,follower_alias
from followers
group by follower_alias
""")

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
""")

artist_follower_ids_df.createOrReplaceTempView('artist_follower')
#
# ranks = [5, 10, 15]
# lambdas = [.1, 1, 10]
# numIters = [5, 10, 20]
# bestModel = None
# bestValidationRmse = float("inf")
# bestRank = 0
# bestLambda = -1.0
# bestNumIter = -1

als = ALS(rank=5, implicitPrefs=True, userCol="follower_id", itemCol="artist_id", ratingCol="count", nonnegative=True)

#model = als_model.fit(artist_follower_ids_df)

training, test = artist_follower_ids_df.randomSplit([0.8, 0.2])

model = als.fit(training)

train, validation = training.randomSplit([.8, .2])

# predictions = model.transform(test)
#
# predictions.registerTempTable("predictions")

#print spark.sql("SELECT * FROM predictions WHERE NOT ISNAN(prediction) ORDER BY prediction DESC").show()

#print spark.sql("SELECT max(prediction), min(prediction) FROM predictions WHERE NOT ISNAN(prediction) ORDER BY prediction DESC").show()

# paramGrid = ParamGridBuilder()\
#     .addGrid(als.rank, [5, 10])\
#     .addGrid(als.regParam, [.1, 1, 10])\
#     .addGrid(als.maxIter, [5,10,20])\
#     .build()
#
# paramGrid = ParamGridBuilder()\
#     .addGrid(als.rank, [5, 10, 20])\
#     .addGrid(als.regParam, [.1, 1, 10])\
#     .build()

paramGrid = ParamGridBuilder()\
    .addGrid(als.rank, [5, 6])\
    .build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="count")

cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator)

# cvModel = cv.fit(training)
#
# best = cvModel.bestModel

# numTraining = train.count()
# numValidation = validation.count()
# numTest = test.count()
#
# for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
#     als = ALS(rank = rank, maxIter = numIter, regParam=lmbda, implicitPrefs=True,
#             userCol="follower_id", itemCol="artist_id", ratingCol="count", nonnegative=True)
#
#     model = als.fit(train)
#
#     validationRmse = computeRmse(model, validation, numValidation)
#
#     print "RMSE (validation) = %f for the model trained with " % validationRmse + \
#           "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
#
#     if (validationRmse < bestValidationRmse):
#         bestModel = model
#         bestValidationRmse = validationRmse
#         bestRank = rank
#         bestLambda = lmbda
#         bestNumIter = numIter
#
# testRmse = computeRmse(bestModel, test, numTest)
#
# # evaluate the best model on the test set
# print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
#   + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)
#
# items = best.itemFactors
#
# df = items.toPandas()
#
# df.to_csv('bestModelItems.csv')
#
# items_mat = np.array(list(df['features']))
#
# print df.head()
#
# np.savetxt('items_mat.csv', items_mat)
