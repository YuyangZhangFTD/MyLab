from pyspark import SparkConf, SparkContext

sc = SparkContext(
    conf=SparkConf().setMaster("local").setAppName("BPR").set(
        "spark.executor.memory", "10g"))

path = "input/ml-100k/u.data"

# rating = sc.textFile(path).map(lambda x: x.split('\t')).map(lambda x: map(int, x[:2]))
rating = sc.textFile(path).map(lambda x: x.split('\t'))
print(rating.first())
