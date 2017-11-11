import numpy as np
from pyspark import SparkConf, SparkContext

from PySparkLab.bpr import optimizeMF as bpr

# from PySparkLab.distbpr import optimizeMF as distbpr

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
    userMat, itemMat = bpr(ratings, 100, num_iter=20, num_neg_samples=10)

    np.savetxt("userMatrix.txt", userMat, delimiter=',')
    np.savetxt("itemMatrix.txt", itemMat, delimiter=',')
