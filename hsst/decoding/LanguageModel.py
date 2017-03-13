import os
import logging
import subprocess


class LanguageModel(object):

    def __init__(self, lm_filename, application_script, semiring='tropical', nthreads=8):
        """
        Initialize the Language Model object
        :param lm_filename: Path to the language model in ARPA format
        :param application_script: Path to the HiFST apply_lm tool
        :param semiring: Semiring used in FSA/FSTs
        :param nthreads: Number of threads to run applylm tool with
        :return:
        """

        self.lm_filename = lm_filename
        self.lm_application_script = application_script

        if semiring == 'tropical':
            self.semiring = 'stdarc'
        elif semiring == 'lex':
            self.simiring = 'lexstdarc'
        else:
            raise NotImplementedError('Semiring %s not implemented.' % semiring)

        self.nthreads = nthreads

    def __call__(self, input_dir, output_dir, sent_range, lm_feature_weight):
        """
            Apply a language model to a directory of lattices using the HiFST applylm tool.
            :param language_model: Language model object
            :param input_dir: Path to the input directory
            :param output_dir: Path to the output directory
            :param sent_range: A range tuple with start and end of fst ids
            :param lm_feature_weight: Language model feature weight
            :return:
            """

        # Compile the applylm command
        command = self.construct_command(os.path.join(input_dir, '?.fst.gz'), os.path.join(output_dir, '?.fst'),
                                         sent_range, lm_feature_weight)

        logging.info('%s' % command)

        # Run the apply_lm command which generates binary fsts
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        logging.debug(err)

        if 'error' in err.lower() or 'abort' in err.lower():
            logging.error('Error applying language model.\nOutput: {0}\nError: {1}'.format(out, err))
            raise Exception('Error applying language model.\nOutput: {0}\nError: {1}'.format(out, err))

        logging.debug('applylm output: %s' % out)

    def construct_command(self, input_pattern, output_dir, fst_range, lm_feature_weight):
        """
        Generate command line for running the apply_lm tool.
        :param input_pattern: Input directory path with filename pattern
        :param output_dir: Output directory path
        :param fst_range: Range tuple
        :param lm_feature_weight: Language Model feature weight
        :return: Formatted string command line
        """

        return """{0} \
                --lm.load={1} \
                --semiring={2} \
                --lm.featureweights={3} \
                --lattice.load={4} \
                --lattice.store={5} \
                --range={6}:{7} \
                --nthreads={8}
                """.format(self.lm_application_script,
                           self.lm_filename,
                           self.semiring,
                           lm_feature_weight,
                           input_pattern,
                           output_dir,
                           fst_range[0],
                           fst_range[1],
                           self.nthreads
                           )
