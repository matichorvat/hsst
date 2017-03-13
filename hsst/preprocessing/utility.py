import re
from xml.sax.saxutils import unescape as un
import langdetect


def clean_utf8(line):
    # both clean-utf8.pl and clean-utf8-wmt07.v2.pl
    line = re.sub(ur'\xc2\x20', u'\x20', line)
    line = re.sub(ur'\xc3\x20', u'\xc3\xa0', line)
    line = re.sub(ur'\xc5\x93', u'\xc3\xba', line)
    line = re.sub(ur'\xc5\xa1', u'\x69', line)
    line = re.sub(ur'\xe2\x80\xa2\x20', u'', line)
    line = re.sub(ur'\xc2\xa0', u'\x20', line)

    # clean-utf8-wmt07.v2.pl specific
    line = re.sub(ur'\xe2\x80\x99', u'\x27', line)
    line = re.sub(ur'\xce\x91', u'\x41', line)
    line = re.sub(ur'\xe2\x80\xa6', u'...', line)
    line = re.sub(ur' \xe2\x80\x93 ', u'\xe2\x80\x92', line)

    return line


def unescape(line):
    return un(line)


def squeeze(line):
    return re.sub(ur'\s+', ' ', line).strip()


def filter_if_empty(line_par):
    return len(line_par[0].strip()) > 0 and len(line_par[1].strip()) > 0


def filter_by_length(line_par, length=100):
    return len(line_par[0].split(' ')) <= length and len(line_par[1].split(' ')) <= length


def filter_by_fertility(line_par, fertility=2.4):
    l1 = len(line_par[0])
    l2 = len(line_par[1])

    reject = test_fertility(l1, l2, 1, 5) or test_fertility(l1, l2, 2, 8) or test_fertility(l1, l2, 3, 9) or (l1 > 3 and l1 > l2 * fertility) or (l2 > 3 and l2 > l1 * fertility)
    return not reject


def test_fertility(l1, l2, lmin, lmax):
    return (l1 == lmin and l2 > lmax) or (l2 == lmin and l1 > lmax)


def filter_if_match(line_par, match_str):
    return line_par[0] != match_str and line_par[1] != match_str


def filter_by_language(line, expected_lang, treshold=0.999995):

    if len(line.split(' ')) < 10:
        return True

    #try:
    langs = langdetect.detect_langs(line.strip().decode('utf-8') if not isinstance(line, unicode) else line.strip())

    if len(langs) == 0:
        return True

    if langs[0].lang != expected_lang and langs[0].prob > treshold:
        return False
    else:
        return True

    #except:
    #    return True
