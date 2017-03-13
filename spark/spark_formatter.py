import re
import sys
import gzip
import logging
import argparse
import itertools
import xml.etree.ElementTree as xml

from pyspark import SparkContext, SparkConf, StorageLevel

from hsst.preprocessing.alignment import align
from hsst.utility.representation import dumps, Side, Alignment, Sentence, SentenceCollection


def open_file(filename):
    if filename is not None:
        if filename.endswith('.gz'):
            return gzip.open(filename, 'rb')
        else:
            return open(filename, 'rb')
    else:
        return None


def close_file(file_handle):
    if file_handle is not None:
        file_handle.close()


def split_line_file(content):
    return content.split('\n')


def split_dmrs_file(content):
    content_split = content.split('<dmrs')
    content_filter = filter(lambda x: x.strip() != '', content_split)
    content_fixed = [('<dmrs' + x).strip() for x in content_filter]
    return content_fixed


mttk_regex = re.compile('SENT:.+')


def split_mttk_align_file(content):
    sentence_chunks = mttk_regex.split(content.strip())
    if sentence_chunks[0].strip() == '':
        sentence_chunks = sentence_chunks[1:]
    #sentence_chunks = filter(lambda x: x.strip() != '', sentence_chunks)
    return [split_mttk_sentence_chunk(chunk) for chunk in sentence_chunks]


def split_mttk_sentence_chunk(chunk):
    lines = filter(lambda x: x.strip() != '', chunk.split('\n'))
    return sorted([tuple(int(x) for x in line.split(' ')[1:]) for line in lines if len(line.split(' ')) == 3])


def split_giza_align_file(content):
    return [process_giza_sentence_line(line) for line in content.split('\n')]


def process_giza_sentence_line(line):
    sentence_parts = line.split('{##}')

    if sentence_parts[0].startswith('ALIGN_ERR'):
        return []

    # Reversing alignments because they need to be (source, target)
    return sorted([tuple(int(x) for x in alignment.split('-')[::-1]) for alignment in sentence_parts[2].strip().split(' ')])


class Formatter(object):
    """
    Preprocess a dataset into a Python object appropriate for outputting
    in various formats
    """
    
    def __init__(self,
                 sid=None,
                 src_plain=None,
                 src_tok=None,
                 src_idx=None,
                 src_parse=None,
                 trg_plain=None,
                 trg_tok=None,
                 trg_idx=None,
                 trg_parse=None,
                 src_trg_align=None,
                 giza=False
                 ):

        self.giza = giza

        try:
            self.sid = open_file(sid)
            self.src_plain = open_file(src_plain)
            self.src_tok = open_file(src_tok)
            self.src_idx = open_file(src_idx)
            self.src_parse = open_file(src_parse)
            self.trg_plain = open_file(trg_plain)
            self.trg_tok = open_file(trg_tok)
            self.trg_idx = open_file(trg_idx)
            self.trg_parse = open_file(trg_parse)
            self.src_trg_align = open_file(src_trg_align)

        except IOError:
            logging.exception("Error opening file")
            raise
        except:
            logging.exception("Unexpected error")
            raise
        
        self.sid_list = list()
        self.src_plain_list = list()
        self.src_tok_list = list()
        self.src_idx_list = list()
        self.src_parse_list = list()
        self.trg_plain_list = list()
        self.trg_tok_list = list()
        self.trg_idx_list = list()
        self.trg_parse_list = list()
        self.src_trg_align_list = list()

    def close_files(self):
        try:
            close_file(self.sid)
            close_file(self.src_plain)
            close_file(self.src_tok)
            close_file(self.src_idx)
            close_file(self.src_parse)
            close_file(self.trg_plain)
            close_file(self.trg_tok)
            close_file(self.trg_idx)
            close_file(self.trg_parse)
            close_file(self.src_trg_align)

        except:
            logging.exception("Unexpected error closing file.")
            raise

    def read_files(self):
        if self.sid is not None:
            self.sid_list = split_line_file(self.sid.read().decode('utf-8').strip())
            
        if self.src_plain is not None:
            self.src_plain_list = split_line_file(self.src_plain.read().decode('utf-8').strip())

        if self.src_tok is not None:
            self.src_tok_list = split_line_file(self.src_tok.read().decode('utf-8').strip())

        if self.src_idx is not None:
            self.src_idx_list = split_line_file(self.src_idx.read().decode('utf-8').strip())

        if self.src_parse is not None:
            self.src_parse_list = split_dmrs_file(self.src_parse.read().decode('utf-8').strip())

        if self.trg_plain is not None:
            self.trg_plain_list = split_line_file(self.trg_plain.read().decode('utf-8').strip())

        if self.trg_tok is not None:
            self.trg_tok_list = split_line_file(self.trg_tok.read().decode('utf-8').strip())

        if self.trg_idx is not None:
            self.trg_idx_list = split_line_file(self.trg_idx.read().decode('utf-8').strip())

        if self.trg_parse is not None:
            self.trg_parse_list = split_dmrs_file(self.trg_parse.read().decode('utf-8').strip())

        if self.src_trg_align is not None:
            alignment_content = self.src_trg_align.read().decode('utf-8').strip()

            if self.giza:
                self.src_trg_align_list = split_giza_align_file(alignment_content)
            else:
                self.src_trg_align_list = split_mttk_align_file(alignment_content)


def parse_sentence(src_untok, src_tok, src_idx, src_dmrs, trg_untok, trg_tok, trg_idx, trg_dmrs, sid, src_trg_align, ignore_alignments=False):

    src_side = parse_side(src_untok, src_tok, src_idx, src_dmrs)
    trg_side = parse_side(trg_untok, trg_tok, trg_idx, trg_dmrs)

    alignment = None
    if not ignore_alignments:
        alignment = parse_alignment(src_side, trg_side, src_trg_align)

    if len(sid.split(' ')) == 3:
        dataset, global_sid, dataset_sid = sid.split(' ')
        orig_id = '%s %s' % (dataset, dataset_sid)

    elif len(sid.split('\t')) == 2:
        global_sid, orig_id = sid.split('\t')

    else:
        raise Exception('Wrong sid format: ' + sid)

    return Sentence(int(global_sid), src_side, trg_side, alignment, orig_id=orig_id)


def parse_side(untok, tok, idx, dmrs):

    if all([x is None for x in [untok, tok, idx, dmrs]]):
        return None

    preprocessed_untok = untok

    if tok is not None:
        preprocessed_tok = tok.split(' ')
    else:
        preprocessed_tok = None

    if idx is not None:
        try:
            preprocesed_idx = [int(x) for x in idx.split(' ')]
        except ValueError:
            logging.exception('IDX integer conversion failed')
            raise
    else:
        preprocesed_idx = None

    if dmrs is not None:
        try:
            parser = xml.XMLParser(encoding='utf-8')
            preprocessed_dmrs = xml.fromstring(dmrs.encode('utf8'), parser=parser) 
        except xml.ParseError:
            sys.stderr.write(dmrs.encode('utf-8'))
            raise

    else:
        preprocessed_dmrs = None

    return Side(preprocessed_untok, preprocessed_tok, preprocesed_idx, preprocessed_dmrs)


def parse_alignment(src_side, trg_side, alignment):

    if src_side is None or trg_side is None:
        return None

    parsed_alignment = align(src_side, trg_side, alignment)
    sttt, sgtt, sttg, sgtg = parsed_alignment

    return Alignment(sttt, sgtt, sttg, sgtg)


def format_sentence(src_untok, src_tok, src_idx, src_dmrs, trg_untok, trg_tok, trg_idx, trg_dmrs, sid, src_trg_align, ignore_alignments=False):
    sentence = parse_sentence(
        src_untok,
        src_tok,
        src_idx,
        src_dmrs,
        trg_untok,
        trg_tok,
        trg_idx,
        trg_dmrs,
        sid,
        src_trg_align,
        ignore_alignments
    )

    sent_col = SentenceCollection([sentence])
    return dumps(sent_col)


def check_length(*args):
    lengths = [len(x) for x in args if len(x) > 0]
    if not all(x == lengths[0] for x in lengths):
        raise Exception('Formatter input lists are of different length (%s).' % ', '.join(str(x) for x in lengths))


def format_dataset(formatter, output_file, sc, num_partitions, ignore_alignments=False):
    formatter.read_files()
    formatter.close_files()

    check_length(
        formatter.src_plain_list,
        formatter.src_tok_list,
        formatter.src_idx_list,
        formatter.src_parse_list,
        formatter.trg_plain_list,
        formatter.trg_tok_list,
        formatter.trg_idx_list,
        formatter.trg_parse_list,
        formatter.sid_list,
        formatter.src_trg_align_list
    )

    data = sc.parallelize(
        itertools.izip_longest(
            formatter.src_plain_list,
            formatter.src_tok_list,
            formatter.src_idx_list,
            formatter.src_parse_list,
            formatter.trg_plain_list,
            formatter.trg_tok_list,
            formatter.trg_idx_list,
            formatter.trg_parse_list,
            formatter.sid_list,
            formatter.src_trg_align_list,
            fillvalue=None
        ),
        num_partitions
    )

    formatted = data.map(lambda x: format_sentence(*x, ignore_alignments=ignore_alignments))
    formatted.saveAsTextFile(output_file)


if __name__ == "__main__":

    appName = 'Dataset formatter'

    parser = argparse.ArgumentParser(description='Format dataset.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--partitions', type=int, default=100)
    parser.add_argument('--sid')
    parser.add_argument('--src_parse')
    parser.add_argument('--src_plain')
    parser.add_argument('--src_tok')
    parser.add_argument('--src_idx')
    parser.add_argument('--trg_parse')
    parser.add_argument('--trg_plain')
    parser.add_argument('--trg_tok')
    parser.add_argument('--trg_idx')
    parser.add_argument('--src_trg_align')
    parser.add_argument('--giza', action='store_true',
                        help='Use GIZA++ alignment file format. Default is MTTK.')
    parser.add_argument('--tune_test', action='store_true',
                        help='Indicated that the dataset is for tuning/testing and no alignments will be stored.')
    parser.add_argument('output_filepath')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    num_partitions = args.partitions
    
    format_params = {
        'sid': args.sid,
        'src_parse': args.src_parse,
        'src_plain': args.src_plain,
        'src_tok': args.src_tok,
        'src_idx': args.src_idx,
        'trg_parse': args.trg_parse,
        'trg_plain': args.trg_plain,
        'trg_tok': args.trg_tok,
        'trg_idx': args.trg_idx,
        'src_trg_align': args.src_trg_align,
        'giza': args.giza
    }

    conf = SparkConf().setAppName(appName).set('spark.executor.extraJavaOptions', '-Dfile.encoding=UTF-8')
    sc = SparkContext(conf=conf)

    format_dataset(Formatter(**format_params), args.output_filepath, sc, num_partitions, ignore_alignments=args.tune_test)
