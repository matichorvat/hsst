#!/home/miproj/mphil.acs.oct2012/mh693/code/hsst/venv/bin/python

import os
import logging
import argparse
from collections import OrderedDict
from ConfigParser import SafeConfigParser, NoSectionError, NoOptionError

from hsst.decoding.Alilats2Splats import Alilats2Splats


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Alilats to splats (a2s) tool.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    subparsers = parser.add_subparsers(dest='mode')

    parser_main = subparsers.add_parser('main')
    parser_main.add_argument('rule_id_offset', default=0, type=int)
    parser_main.add_argument('range')
    parser_main.add_argument('feature_weights')
    parser_main.add_argument('a2s_config')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, format='%(asctime)s:%(levelname)s:%(message)s',
                            level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                            level=logging.DEBUG if args.verbose else logging.INFO)

    config = SafeConfigParser()
    config.read(args.a2s_config)

    if args.range:
        try:
            dataset_range = (int(args.range.split('-')[0]), int(args.range.split('-')[1]))

        except (IndexError, ValueError):
            raise Exception('Incorrect specification of range. Needs to have form start:end.')

    else:
        dataset_range = None

    run_num = get_config_option(config, 'DEFAULT', 'run_num')
    tmp_dir = get_config_option(config, 'Settings', 'temporary_storage')
    tmp_dir = os.path.join(tmp_dir, run_num)

    openfst_bin = get_config_option(config, 'Software', 'openfst_bin')
    a2s_bin = get_config_option(config, 'Software', 'a2s_bin')
    lexmap_bin = get_config_option(config, 'Software', 'lexmap_bin')

    a2s_nthreads = get_config_option(config, 'Settings', 'nthreads', int)
    a2s_nthreads = a2s_nthreads if a2s_nthreads is not None else 1

    hiero_grammar = get_config_option(config, 'Data', 'hiero_grammar')
    hsst_rule_app_pattern = get_config_option(config, 'Data', 'hsst_rule_app_pattern')
    hiero_lm = get_config_option(config, 'Data', 'hiero_lm')
    joint_lm = get_config_option(config, 'Data', 'joint_lm')
    hiero_alilats = get_config_option(config, 'Data', 'hiero_alilats')
    joint_alilats = get_config_option(config, 'Data', 'joint_alilats')
    joint_veclats = get_config_option(config, 'Output', 'joint_veclats')

    hiero_feature_weights = get_config_option(config, 'Settings', 'hiero_feature_weights')
    hsst_feature_weights = OrderedDict((name, float(value)) for name, value in config.items('Features') if name != 'run_num')

    # Offset by one as feature positions start at 1. Offset by additional one for LM feature in case of HSST_ONLY
    hiero_feature_num = 1 + len(hiero_feature_weights.split(',')) if hiero_feature_weights else 1

    # Initialize object
    a2s = Alilats2Splats(
        openfst_bin,
        a2s_bin,
        hsst_rule_app_pattern,
        joint_lm,
        tmp_dir,
        hiero_grammar=hiero_grammar,
        hiero_language_model=hiero_lm,
        hiero_feature_num=hiero_feature_num,
        lexmap_bin=lexmap_bin,
        rule_id_offset=args.rule_id_offset,
        nthreads=a2s_nthreads
    )

    # Run a2s
    a2s(
        joint_alilats,
        joint_veclats,
        args.feature_weights,
        hsst_feature_weights,
        dataset_range,
        hiero_alilats=hiero_alilats
    )

