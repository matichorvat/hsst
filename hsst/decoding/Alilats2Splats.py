import os
import re
import ast
import logging
import subprocess

from hsst.utility import utility
from hsst.decoding import helpers
from hsst.decoding.alignment.AlignmentOpenFST import AlignmentOpenFST


class Alilats2Splats(object):

    def __init__(self, openfst_bin, a2s_bin, hsst_rule_app_pattern, language_model, tmp_dir, hiero_grammar=None,
                 hiero_language_model=None, hiero_feature_num=0, nthreads=1, lexmap_bin=None, rule_id_offset=0):
        """
        Initialize the Alilats2Splats object
        :param openfst_bin: Path to the OpenFST bin directory
        :param a2s_bin: Path to the HiFST alilats2splats tool
        :param hsst_rule_app_pattern: HSST tune set rule app path pattern
        :param language_model: Path to the joint language model
        :param tmp_dir: Path to temporary directory
        :param hiero_grammar: Path to the hiero tune set grammar file
        :param hiero_language_model: Path to the hiero tune set language model in mmap format
        :param hiero_feature_num: Number of hiero features
        :param nthreads: Number of threads to run applylm tool with
        :param lexmap_bin: Path to the HiFST lexmap tool. Specify if HSST/joint alilats need to be converted to lex semiring.
        :return:
        """

        logging.info('Initializing alilats2splats object.')

        self.openfst_bin = openfst_bin
        self.a2s_bin = a2s_bin
        self.lexmap_bin = lexmap_bin

        self.rule_id_offset = rule_id_offset
        self.hsst_rule_app_pattern = hsst_rule_app_pattern

        self.language_model = language_model
        self.tmp_dir = tmp_dir

        self.hiero_grammar = hiero_grammar
        self.hiero_language_model = hiero_language_model
        self.hiero_feature_num = hiero_feature_num

        self.nthreads = nthreads
        self.openfst = AlignmentOpenFST(None, None)

    def __call__(self, joint_alilats, joint_vectorlats, joint_feature_weights, hsst_feature_weights, sent_range,
                 hiero_alilats=None):
        """
        :param joint_alilats: Path to join ALILATS directory
        :param joint_vectorlats: Path to vector lattices output directory
        :param joint_feature_weights: Hiero and HSST feature weights in a comma-separated string
        :param hsst_feature_weights: HSST feature weights in an OrderedDict
        :param sent_range: Tuple indicating start and end of sentence range
        :param hiero_alilats: Path to Hiero ALILATS directory
        :return:
        """

        logging.info('Starting alilats2splats on sentence range %s.' % (sent_range,))

        # Create flower lattice
        flower_lattice_string = self.create_flower_lattice(
            joint_feature_weights,
            hsst_feature_weights,
            sent_range,
            hiero_alilats=hiero_alilats
        )

        if self.lexmap_bin:
            joint_alilats = self.convert_alilats_semiring(joint_alilats, sent_range)

        # Run alilats2splats tool with augmented flower lattice
        self.alilats2splats(flower_lattice_string, joint_alilats, joint_vectorlats, joint_feature_weights, sent_range)

    def create_flower_lattice(self, joint_feature_weights, hsst_feature_weights, sent_range, hiero_alilats=None):
        """
        Create the flower lattice with arcs for HSST rules.
        :param joint_feature_weights: Hiero and HSST feature weights in a comma-separated string
        :param hsst_feature_weights: HSST feature weights in an OrderedDict
        :param sent_range: Tuple indicating start and end of sentence range
        :param hiero_alilats: Path to Hiero ALILATS directory
        :return:
        """

        logging.info('Creating flower lattice.')

        if hiero_alilats is not None:
            hiero_feature_weights = ','.join(joint_feature_weights.split(',')[0:self.hiero_feature_num])
            base_flower_lattice_string = self.get_hiero_flower_lattice(hiero_alilats, hiero_feature_weights, sent_range)

        else:
            base_flower_lattice_string = '0'

        # Extract HSST rule features from rule application grammar files
        rule_features = self.extract_rule_features(sent_range, hsst_feature_weights)

        # Augment hiero flower lattice with HSST rule_id:feature_vector arcs
        joint_flower_fst_lattice_string = self.augment_flower_lattice_with_hsst(
            base_flower_lattice_string,
            rule_features
        )

        return joint_flower_fst_lattice_string

    def get_hiero_flower_lattice(self, hiero_alilats, hiero_feature_weights, sent_range):
        """
        Create and read in Hiero flower lattice representing mappings between rule ids and feature values.
        :param hiero_alilats: Path to Hiero ALILATS directory
        :param hiero_feature_weights: Hiero feature weights in a comma-separated string
        :param sent_range: Tuple indicating start and end of sentence range
        :return: Flower lattice FST as string
        """

        logging.info('Running alilats2splats to obtain hiero flower lattice.')

        tmp_hiero_veclats = os.path.join(self.tmp_dir, 'tmp_veclats', '?.fst.gz')
        tmp_flower_lattice = os.path.join(self.tmp_dir, 'hiero_flower.fst')

        # Run alilats2splats to create the hiero flower lattice
        get_hiero_flower_fst_command = self.construct_command(
            hiero_alilats,
            tmp_hiero_veclats,
            self.hiero_grammar,
            self.hiero_language_model,
            hiero_feature_weights,
            sent_range,
            ruleflower_fst_store=tmp_flower_lattice
        )

        run_subprocess(get_hiero_flower_fst_command)

        # Read in flower lattice
        flower_lattice_string = run_subprocess('{}/fstprint {}'.format(self.openfst_bin, tmp_flower_lattice))

        return flower_lattice_string

    def convert_alilats_semiring(self, joint_alilats, sent_range):
        """
        Convert HSST or joint lattices to lexicographical semiring.
        :param joint_alilats: Path to joint ALILATS directory.
        :param sent_range: Sentence range tuple
        :return: New path to joint ALILATS directory directory
        """

        logging.info('Converting joint alilats from tropical to lexicographical semiring.')

        join_latx_lex_dir = os.path.join(self.tmp_dir, 'ALILATS_LEX')
        utility.make_sure_path_exists(join_latx_lex_dir)
        joint_alilats_lex = join_latx_lex_dir + '/?.lex.fst'

        run_subprocess('{0} --input={1} --output={2} --action=std2lex -r {3}:{4}'.format(self.lexmap_bin,
                                                                              joint_alilats,
                                                                              joint_alilats_lex,
                                                                              sent_range[0],
                                                                              sent_range[1],
                                                                              ))

        return joint_alilats_lex

    def alilats2splats(self, joint_flower_lattice_string, joint_alilats, joint_vectorlats, joint_feature_weights,
                       sent_range):
        """
        Run alilats2splats with modified flower lattice.
        :param joint_flower_lattice_string: Augmented flower lattice represented as a string
        :param joint_alilats: Path to join ALILATS directory
        :param joint_vectorlats: Path to vector lattices output directory
        :param joint_feature_weights: Hiero and HSST feature weights in a comma-separated string
        :param sent_range: Tuple indicating start and end of sentence range
        :return:
        """

        logging.info('Running alilats2splats with augmented flower lattice on joint alilats.')

        # Write joint flower lattice to disk and compile
        flower_lattice_str_path = os.path.join(self.tmp_dir, 'joint_flower.fst.txt')
        flower_lattice_fst_path = os.path.join(self.tmp_dir, 'joint_flower.fst')

        with open(flower_lattice_str_path, 'wb') as o:
            o.write(joint_flower_lattice_string)

        run_subprocess('{0}/fstcompile --arc_type=tropicalsparsetuple {1} | {0}/fstarcsort --sort_type=ilabel > {2}'.format(
            self.openfst_bin, flower_lattice_str_path, flower_lattice_fst_path))

        logging.info('Joint features: %s' % (joint_feature_weights,))

        # Run alilats2splats with the new flower lattice
        alilats2splats_command = self.construct_command(
            joint_alilats,
            joint_vectorlats,
            flower_lattice_fst_path,
            self.language_model,
            joint_feature_weights,
            sent_range
        )

        run_subprocess(alilats2splats_command)

    def construct_command(self, alilats_dir, vectorlats_dir, ruleflower_fst_path, lm_path, feature_weights,
                          sent_range, ruleflower_fst_store=None):
        """
        Generate command line for running the alilats2splats tool.
        :param alilats_dir: Path to ALILATS directory
        :param vectorlats_dir: Path to vector lattices output directory
        :param ruleflower_fst_path: Grammar or flower lattice path
        :param lm_path: Path to the language model
        :param feature_weights:
        :param sent_range: Tuple indicating start and end of sentence range
        :param ruleflower_lattice_store:
        :return:
        """

        command = """{0} -v \
                        --sparseweightvectorlattice.loadalilats={1} \
                        --sparseweightvectorlattice.store={2} \
                        --ruleflowerlattice.load={3} \
                        --ruleflowerlattice.filterbyalilats \
                        --sparseweightvectorlattice.stripspecialepsilonlabels=yes \
                        --lm.load={4} \
                        --featureweights={5} \
                        --range={6}:{7} \
                        --nthreads={8}
                        """.format(self.a2s_bin,
                                   alilats_dir,
                                   vectorlats_dir,
                                   ruleflower_fst_path,
                                   lm_path,
                                   feature_weights,
                                   sent_range[0],
                                   sent_range[1],
                                   self.nthreads
                                   )

        if ruleflower_fst_store is not None:
            command = command.strip() + " --ruleflowerlattice.store={}".format(ruleflower_fst_store)

        return command

    def extract_rule_features(self, sent_range, hsst_feature_weights):
        """
        Extract feature vectors for given rule ids.
        :param sent_range: Tuple indicating start and end of sentence range
        :param hsst_feature_weights: HSST feature weights in an OrderedDict
        :return: Dictionary rule_id:list of tuples (feature_id, feature_value)
        """

        sent_ids = range(sent_range[0], sent_range[1] + 1)

        # Get a list of rule_app grammar filepaths in sent_range
        rule_app_directory = os.path.dirname(self.hsst_rule_app_pattern)
        filename_match = re.compile(os.path.basename(self.hsst_rule_app_pattern).replace('?', '[0-9]+'))
        filepaths = [os.path.join(rule_app_directory, filename) for filename in os.listdir(rule_app_directory) if filename_match.match(filename) and int(filename.split('.')[0]) in sent_ids]

        logging.info('Extracting HSST rule features for {} sentences.'.format(len(filepaths)))

        rules = dict()

        for filepath in filepaths:
            sentence_rules = self.extract_rules_for_single_sentence(filepath)
            rules.update(sentence_rules)

        logging.info('Computing rule features for {} rules.'.format(len(rules)))

        rule_features = dict()

        for rule_id in rules:

            integration = self.hiero_grammar is not None

            features = helpers.compute_hsst_features(
                ast.literal_eval(rules[rule_id][1]),
                target_side=rules[rule_id][0],
                integration=integration
            )

            ordered_features = helpers.order_hsst_features(features, hsst_feature_weights)

            rule_features[rule_id] = ordered_features

        return rule_features

    def extract_rules_for_single_sentence(self, filepath):
        """
        Extract rules from a single sentence grammar file.
        :param filepath: Path to the rule app grammar file
        :return: rule_id:rule_features dictionary
        """

        rules = dict()
        with open(filepath) as f:
            for line in f:
                line = line.strip()

                assert len(line.split('\t')) == 3

                rule_id, target, features = line.split('\t')
                rules[int(rule_id) + self.rule_id_offset] = (target, features)

        return rules

    def augment_flower_lattice_with_hsst(self, base_flower_lattice_string, rule_features):
        """
        Augment the flower lattice string by adding HSST rule arcs.
        :param base_flower_lattice_string: FST represented as a string
        :param rule_features: Dictionary rule_id:list of tuples (feature_id, feature_value)
        :return: Augmented FST represented as a string
        """

        logging.info('Augmenting flower lattice with HSST rules.')

        arc_split = base_flower_lattice_string.strip().split('\n')
        arcs = arc_split[:-1]

        for rule_id, feature_vector in sorted(rule_features.items(), key=lambda x: int(x[0])):
            arcs.append(self.construct_arc_string(rule_id, feature_vector))

        arcs.append(arc_split[-1])
        augmented_flower_lattice_string = '\n'.join(arcs)

        return augmented_flower_lattice_string

    def construct_arc_string(self, rule_id, feature_vector):
        """
        Construct the FST arc string for the given rule.
        :param rule_id: Rule id
        :param feature_vector: List of tuples (feature_id, feature_value)
        :return: Formatted arc string
        """

        feature_vector_string = '0,'
        feature_vector_string += ','.join(
            '{},{}'.format(self.hiero_feature_num + feature_id, feature_value) \
            for feature_id, feature_value in feature_vector
        )

        arc_string = '0\t0\t{0}\t{0}\t{1},'.format(rule_id, feature_vector_string)

        return arc_string


def run_subprocess(command):
    """
    Run a command as a subprocess.
    :param command: Command string
    :return:
    """

    logging.debug(command)

    # Run the command
    proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if 'error' in err.lower() or 'fatal' in err.lower() or 'abort' in err.lower() or 'larger dimensionality than the parameters' in err.lower():
        logging.error('Error running command {0}.\nError: {1}'.format(command, err))
        raise Exception('Error running command {0}.\nError: {1}'.format(command, err))

    return out