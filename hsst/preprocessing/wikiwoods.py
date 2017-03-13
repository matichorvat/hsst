import re


init_line_pattern = re.compile(r"\[([0-9]+)\] \([0-9]+ of [0-9]+\) \{[0-9]+\} `(.+)'")
token_line_pattern = re.compile(r'<([0-9]+):([0-9]+)>, [0-9]+, "(.+)", [0-9]+')
char_span_pattern = re.compile(r"<([0-9]+):([0-9]+)>")
ace_jp_pattern = re.compile(r"SENT: (.+)")

punctuation = ".!?,;-:"


def split_wikiwoods_file(filename):
    # Split the input file into sentence chunks
    sentence_chunks = re.split('\x04', filename.read().decode('utf8'))
    sentence_chunks = filter(lambda x: x.strip() != '', sentence_chunks)
    mrs_list = [process_sentence_chunk(sentence_chunk) for sentence_chunk in sentence_chunks]
    return mrs_list


def process_sentence_chunk(segment):
    # Process each sentence segment file to produce DMRS style XML

    segment_chunks = segment.split('\n\n')
    header_chunk = segment_chunks[0]
    token_chunk = segment_chunks[1]
    mrs_chunk = segment_chunks[5]

    first_line_match = init_line_pattern.search(header_chunk)
    ident = first_line_match.group(1)
    sentence_surface = first_line_match.group(2)

    # Extract the tokenized version of the surface and create character to token mapping
    tokenized_surface = []
    punctuation_tokens = []
    char_token_mapping = {}
    char_span_start_dict = {}
    char_span_end_dict = {}

    for line in token_chunk.split('\n'):
        if not line.startswith('<') and not line.startswith('>'):
            token_match = token_line_pattern.search(line)
            char_span = (int(token_match.group(1)), int(token_match.group(2)))
            char_span_start_dict[char_span[0]] = char_span
            char_span_end_dict[char_span[1]] = char_span
            token = token_match.group(3)
            char_token_mapping[char_span] = len(tokenized_surface)
            if token in punctuation:
                punctuation_tokens.append(len(tokenized_surface))
            tokenized_surface.append(token)

    # Replace character spans with token spans in the MRS chunk
    mrs_chunk = char_span_pattern.sub(
        lambda match: char_span_to_token_span_map(match, char_token_mapping, char_span_start_dict, char_span_end_dict,
                                                  punctuation_tokens),
        mrs_chunk)

    return mrs_chunk


def char_span_to_token_span_map(char_span_match, char_token_mapping, char_span_start_dict, char_span_end_dict,
                                punctuation_tokens):
    # Replace a single character span with a token span

    char_span = (int(char_span_match.group(1)), int(char_span_match.group(2)))

    if char_span in char_token_mapping:
        # If the exact character span is in the character to token span mapping, use that
        return "<%d:%d>" % (char_token_mapping[char_span], char_token_mapping[char_span])
    else:
        # If it isn't, it means that the character span spans multiple character spans
        # Find the starting and end character span and convert them to start and end token span
        # and use that

        start_token_index = char_token_mapping[char_span_start_dict[char_span[0]]]
        end_token_index = char_token_mapping[char_span_end_dict[char_span[1]]]

        # In case that the last token is a punctuation mark, separate it from the span
        if end_token_index in punctuation_tokens:
            end_token_index -= 1

        return "<%d:%d>" % (start_token_index, end_token_index)