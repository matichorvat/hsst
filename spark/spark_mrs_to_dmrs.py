import logging
import argparse

from pyspark import SparkContext, SparkConf, StorageLevel

from pydmrs.mrs_to_dmrs.mrs_to_dmrs import mrs_to_dmrs


def read_file(sc, file_location, num_partitions):
    return sc.textFile(file_location, num_partitions, use_unicode=True)


def convert(input_file, output_file, sc, num_partitions):
    data = read_file(sc, input_file, num_partitions)
    convert = data.map(lambda x: mrs_to_dmrs(x, True))
    convert.saveAsTextFile(output_file)


if __name__ == "__main__":

    appName = 'MRS to DMRS mapping'

    parser = argparse.ArgumentParser(description='Map MRS to DMRS graphs.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--partitions', type=int, default=100)
    parser.add_argument('input_filepath')
    parser.add_argument('output_filepath')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    num_partitions = args.partitions

    conf = SparkConf().setAppName(appName).set('spark.executor.extraJavaOptions', '-Dfile.encoding=UTF-8')
    sc = SparkContext(conf=conf)

    convert(args.input_filepath, args.output_filepath, sc, num_partitions)
