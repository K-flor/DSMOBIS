from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

if __name__ == "__main__":
    spark = SparkSession.builder.appName("BinomialLogisticRregression").master("local[*]").getOrCreate()
    # Load training data
    training = spark.read.format("").load("")

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)

    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    # We can also use the multinomial family for binary classification
    mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

    # Fit the model
    mlrModel = mlr.fit(training)

    # Print the coefficients and intercepts for logistic regression with multinomial family
    print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
    print("Multinomial intercepts: " + str(mlrModel.interceptVector))

    # Extract the summary from the returned LogisticRegressionModel instance trained
    # in the earlier example
    trainingSummary = lrModel.summary

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)