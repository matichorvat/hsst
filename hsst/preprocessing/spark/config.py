PATH = '/home/blue1/mh693/software/phd/hsst/preprocessing/spark/'
PERL_NORMALIZE_CMD = PATH + 'tools/perl_scripts/normalize-punctuation.perl'
AACHEN_TOKENIZER_CMD = PATH + 'tools/corpusTools.28set2005/intel/applyTokenizer ' + PATH + 'tools/corpusTools.28set2005/en.tokmap'
POST_AACHEN_TOKENIZER_CMD = PATH + 'tools/scripts/postAachanTokeniser.sh'
POST_POST_AACHEN_TOKENIZER_CMD = PATH + 'tools/perl_scripts/engtok-1.0/tokenizer.pl'
MOSES_TOKENIZER_CMD = PATH + 'tools/perl_scripts/tokenize.perl'
MOSES_TRUECASER_CMD = PATH + 'tools/recaser/truecase.perl'
MOSES_TRUECASER_DE_MODEL = PATH + 'tools/recaser/truecase-model.eva.de'

AACHEN_TOKENIZER_TIMEOUT = 1.0
