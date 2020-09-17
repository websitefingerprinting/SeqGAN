from os.path import join, abspath, dirname, pardir
import logging
import configparser

BASE_DIR = abspath(join(dirname(__file__), pardir))
confdir   = join(BASE_DIR, 'src/conf.ini')
outputdir = join(BASE_DIR, 'data/')
modeldir  = join(BASE_DIR, 'models/')
gendir  = join(BASE_DIR, 'dump/')
syndir = join(BASE_DIR, 'synthesized/')
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger


def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)
    return dict(cf['default'])


