#!/home/miproj/mphil.acs.oct2012/mh693/code/hsst/venv/bin/python

import os
import logging
import argparse
import itertools
from collections import OrderedDict
from ConfigParser import SafeConfigParser, NoSectionError, NoOptionError

from hsst.utility import utility
from hsst.decoding.alignment.AlignmentDecoder import AlignmentDecoder
from hsst.decoding.alignment.AlignmentOpenFST import AlignmentOpenFST


def get_config_option(config, section, option, opt_type=None):

    if opt_type is None:
        opt_type = basestring

    try:
        if opt_type == basestring:
            return config.get(section, option)
        elif opt_type == int:
            return config.getint(section, option)
        elif opt_type == float:
            return config.getfloat(section, option)
        elif opt_type == bool:
            return config.getboolean(section, option)

    except (NoSectionError, NoOptionError):
        if opt_type == bool:
            return False
        else:
            return None


def parse_local_prune_conditions(conditions_string):
    split_string = conditions_string.split(',')
    conditions = itertools.izip_longest(*[itertools.islice(split_string, i, None, 4) for i in range(4)])
    parsed_conditions = list()

    for condition in conditions:
        symbol, coverage_n, size_n, threshold_n = condition

        if symbol != 'X':
            raise Exception('Symbols other than X not supported.')

        parsed_conditions.append((symbol, int(coverage_n), int(size_n), int(threshold_n)))

    return sorted(parsed_conditions, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Decode a dataset.')
    parser.add_argument('-ra', '--range')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('config')

    args = parser.parse_args()

    config = SafeConfigParser()
    config.read(args.config)

    run_num = get_config_option(config, 'DEFAULT', 'run_num')

    # Set up logging
    if args.log:
        log_dir = get_config_option(config, 'Logging', 'log_dir')

        if log_dir:
            utility.make_sure_path_exists(log_dir)

        log_filename = os.path.join(log_dir, str(run_num) + '.log')
        logging.basicConfig(
            filename=args.log, format='%(asctime)s:%(levelname)s:%(message)s',
            level=logging.DEBUG if args.verbose else logging.INFO
        )

    else:
        logging.basicConfig(
            format='%(asctime)s:%(levelname)s:%(message)s',
            level=logging.DEBUG if args.verbose else logging.INFO
        )

    memory_limit = get_config_option(config, 'Settings', 'memory_limit', int)

    if memory_limit is not None and memory_limit > 0:
        utility.set_memory_limit(memory_limit)

    # Reading basic info for the decoding task
    task = get_config_option(config, 'General', 'task')
    job = get_config_option(config, 'General', 'job')
    target_lang = get_config_option(config, 'General', 'target_lang')

    if task is None or job is None or target_lang is None or run_num is None:
        raise Exception('Missing information from config file')

    if job == 'test':
        testing_mode = True
    elif job == 'tune':
        testing_mode = False
    else:
        raise Exception('Incorrect job information from config file')

    # Set up dataset range if working on a subset
    # Check whether a subset was specified
    if args.range:
        try:
            dataset_range = (int(args.range.split('-')[0]), int(args.range.split('-')[1]))

            # Augment the run number if we are operating only on a subset of the dataset
            run_num += '.' + str(dataset_range[0]) + '.' + str(dataset_range[1])

        except (IndexError, ValueError):
            raise Exception('Incorrect specification of range. Needs to have form start:end.')

        logging.info('Decoding a subset of a dataset file: %s-%s' % dataset_range)

    else:
        dataset_range = None

    # Load Dataset to decode
    logging.info('Loading dataset')
    dataset = utility.load_dataset(
        get_config_option(config, 'Data', 'dataset'),
        subset_range=dataset_range
    )

    logging.info('Loaded %d sentences' % len(dataset))

    # Set up the decoder
    logging.info('Loading decoder')
    decoder = AlignmentDecoder(
        testing_mode,
        os.path.join(get_config_option(config, 'Settings', 'temporary_storage'), str(run_num) + '/'),
        filter_min=get_config_option(config, 'Settings', 'filter_min', int),
        filter_max=get_config_option(config, 'Settings', 'filter_max', int),
        timeout=get_config_option(config, 'Settings', 'timeout', int),
        write_rtn=get_config_option(config, 'Output', 'write_rtn'),
        hiero_peeping=get_config_option(config, 'Integration', 'hiero_peeping', bool),
        expanded_integration=get_config_option(config, 'Integration', 'expanded_integration', bool),
        heuristic_intersection=get_config_option(config, 'Integration', 'heuristic_intersection', bool),
        hiero_only=get_config_option(config, 'Settings', 'hiero_only', bool),
        gzip=get_config_option(config, 'Settings', 'gzip', bool),
        max_cell_coverage=get_config_option(config, 'Settings', 'max_cell_coverage', int),
        carg_rules=get_config_option(config, 'Settings', 'carg_rules', bool)
    )

    # Load feature weights
    feature_weights_dict = OrderedDict((name, float(value)) for name, value in config.items('Features'))

    # Set up OpenFST interface
    rule_id_offset = get_config_option(config, 'Integration', 'rule_id_offset', int)
    openfst = AlignmentOpenFST(
        get_config_option(config, 'Data', 'target_wmap'), None,
        rule_id_offset if rule_id_offset is not None else 0
    )

    # Run decoder
    decoder(
        dataset,
        get_config_option(config, 'Data', 'grammar_pattern'),
        get_config_option(config, 'Data', 'coverage_pattern'),
        feature_weights_dict,
        openfst,
        os.path.join(get_config_option(config, 'Data', 'translation_lats')),
        hiero_dir=get_config_option(config, 'Integration', 'hiero_dir'),
        hiero_lats=os.path.join(get_config_option(config, 'Integration', 'hiero_lats')),
        prune_reference_shortest_path=get_config_option(config, 'Settings', 'prune_reference_shortest_path', int),
        outdir=get_config_option(config, 'Output', 'outdir')
    )
