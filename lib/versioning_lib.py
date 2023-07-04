import hashlib
import logging
import os
import shutil
import time
from logging.handlers import RotatingFileHandler
from pprint import pprint

import git
import numpy
# from filelock import FileLock


def setup_logger(logger_name, log_file, level=logging.INFO):
    """
    Return a log object associated to <log_file> with specific formatting and rotate handling.
    """

    log = logging.getLogger(logger_name)

    if not getattr(log, 'handler_set', None):
        logging.Formatter.converter = time.gmtime
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

        rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
        rotatehandler.setFormatter(formatter)

        # TODO: Debug stream handler for development
        # streamHandler = logging.StreamHandler()
        # streamHandler.setFormatter(formatter)
        # l.addHandler(streamHandler)

        log.addHandler(rotatehandler)
        log.setLevel(level)
        log.handler_set = True

    elif not os.path.exists(log_file):
        logging.Formatter.converter = time.gmtime
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

        rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
        rotatehandler.setFormatter(formatter)

        # TODO: Debug stream handler for development
        # streamHandler = logging.StreamHandler()
        # streamHandler.setFormatter(formatter)
        # l.addHandler(streamHandler)

        log.addHandler(rotatehandler)
        log.setLevel(level)
        log.handler_set = True

    return log


def generate_md5(dataset_filename):
    """
    Calculate md5 digest.
    :param dataset_filename:
    :return MD5 digest:
    """

    md5_hash = hashlib.md5()

    with open(dataset_filename, 'rb') as dataset_file:
        dataset_content = dataset_file.read()

    md5_hash.update(dataset_content)
    digest = md5_hash.hexdigest()

    return digest


def rename_md5_dataset(dataset_filename, dataset_basename=None, digest=None, return_digest=False):
    """
    Rename dataset filename adding md5.
    :param dataset_filename:
    :param dataset_basename: the basename to add md5, default value is the dataset_filename basename.
    :param digest: arbitrary digest value.
    :param return_digest:
    :return md5_dataset_filename:
    """

    digest = generate_md5(dataset_filename) if digest is None else digest

    dataset_dirname = os.path.dirname(dataset_filename)
    dataset_basename = os.path.splitext(os.path.basename(dataset_filename))[0] if dataset_basename is None else dataset_basename
    dataset_extension = os.path.splitext(os.path.basename(dataset_filename))[1]

    if not dataset_dirname:
        dataset_dirname = '.'

    md5_dataset_filename = '{}/{}_{}{}'.format(dataset_dirname, dataset_basename, digest[0:8], dataset_extension)

    os.rename(dataset_filename, md5_dataset_filename)

    return md5_dataset_filename if not return_digest else digest


def full_print(*args, **kwargs):
    opt = numpy.get_printoptions()
    numpy.set_printoptions(linewidth=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


def full_log(logger_object, msg_to_log):
    opt = numpy.get_printoptions()
    numpy.set_printoptions(linewidth=numpy.inf)
    logger_object.info(msg_to_log)
    numpy.set_printoptions(**opt)


def log_versioning_info(dataset_md5, output_path, return_timestamp=True, level=logging.INFO):
    """
    Saves the versioning information in a logger. The logger is created by this function with a fixed schema name.
    Logger is returned to use it in the whole code.
    :param dataset_md5: MD5 string for identify the dataset
    :param output_path: results directory
    :param logger: a custom logger could be passed to this function
    :param return_timestamp: boolean, if True function returns logger and timestamp, default is true
    :param level: logging level, default is logging.INFO
    :return: logger object and (optionally) timestamp
    """
    # home = os.path.expanduser('~')
    # with FileLock('%s/.versioning_lock' % home):
    # Saving the timestamp
    timestamp = str(int(time.time()))
    # Creating the logger object
    log_absolute_filename = '%s/%s_code_and_config_versioning.log' % (os.getcwd(), timestamp)
    logger = setup_logger('%s_versioning_logger' % timestamp,  log_absolute_filename, level)
    # Getting the current git repository
    repo = git.Repo(search_parent_directories=True)
    # Getting the SHA-1 of the last commit
    repo_sha1 = repo.head.commit.hexsha

    # Logging versioning info
    full_log(logger, 'SHA-1 COMMIT %s' % repo_sha1)
    full_log(logger, 'TIMESTAMP: %s' % timestamp)
    full_log(logger, 'DATASET MD5 HASH: %s' % dataset_md5)
    full_log(logger, 'OUTPUT PATH: %s' % output_path)

    # Adding head of log_file and committing changes.
    repo.git.add(log_absolute_filename)
    repo.git.commit(m='Experiment started: %s' % timestamp)
    logger.timestamp = timestamp

    return logger, timestamp if return_timestamp else logger


def close_versioning_info(logger_object, timestamp=0, absolute_results_filenames=[], upload_results=False):
    """
    Function closes the logger_object releasing related handlers. Each log file will be
    committed. If a list of file_basenames is passed, all these files are added and committed.
    :param logger_object:
    :param timestamp:
    :param absolute_results_filenames: list of string, results absolute filenames.
    :param upload_results: boolean, if True, results files will be uploaded on the remote reporitory.
    :return:
    """
    # Getting the current git repository
    # home = os.path.expanduser('~')
    # with FileLock('%s/.versioning_lock' % home):
    repo = git.Repo(search_parent_directories=True)
    full_log(logger_object, 'VERSIONING CLOSED.')
    handlers = logger_object.handlers

    for handler in handlers:
        # Adding to the repository the just created log_file
        log_absolute_filename = handler.baseFilename
        repo.git.add(log_absolute_filename)

    # For each absolute path, we add related files to the repository.
    for i, absolute_filename in enumerate(absolute_results_filenames):
        if upload_results:
            if not absolute_filename.startswith('/'):
                print('Warning: are you passing the an absolute path? %s' % absolute_filename)
            repo.git.add(absolute_filename)
        full_log(logger_object, "RESULTS FILEPATH [%s]: %s" % (i, absolute_filename))

    for handler in handlers:
        # Destroying logger object
        handler.close()
        logger_object.removeHandler(handler)

    # Logger object holds the timestamp added by the log_versioning_info function.
    # If it is not set, will be used the passed one (or default).
    try:
        repo.git.commit(m='Experiment completed: %s' % logger_object.timestamp)
    except:
        repo.git.commit(m='Experiment completed: %s' % timestamp)
