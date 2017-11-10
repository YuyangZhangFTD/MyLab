from pyspark import SparkConf, SparkContext

from PySparkLab.distbpr import optimizeMF as distbpr

conf = (SparkConf().setMaster("local")
        .setAppName("BPR")
        .set("spark.executor.memory", "10g"))

sc = SparkContext(conf=conf)

if __name__ == '__main__':
    PREFIX = './'

    ratings = sc.textFile(
        "%s/input/ml-100k/u.data" % PREFIX
    ).map(
        lambda line: line.split("\t")
    ).map(
        lambda x: map(int, x[:2])
    )

    # userMat, prodMat = bpr(ratings, 10, 10)
    userMat, prodMat = distbpr(ratings, 10, 10)

    print(type(userMat))
