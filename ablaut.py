from argparse import ArgumentParser
from collections import namedtuple, defaultdict, Counter
from itertools import combinations
import logging
from math import factorial
from pprint import pprint
from statistics import mean
from typing import Iterable, Dict, Set, Tuple, List, Collection

import Levenshtein as lev
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

TokenMatch = namedtuple('TokenMatch', ['token1', 'token2',
                                       'vector1', 'vector2',
                                       'string_similarity',
                                       'vector_similarity'])


WORD_START = '*'
WORD_END = '#'
DIFF_BOUNDARY = '-'

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    return args


def get_matches(model, string_threshold=0.75, vector_threshold=0.65):
    counts = (entry.count for entry in model.vocab.values())
    mean_count = mean(counts)

    vocab = sorted({word.casefold() for word, entry 
                    in model.vocab.items()
                    if entry.count > mean_count * 2})

    num_combinations = factorial(len(vocab)) // 2 
    num_combinations //= factorial(len(vocab) - 2)


    for token1, token2 in tqdm(combinations(vocab, 2),
                               total=num_combinations):
        if token1.isalpha() and token2.isalpha():
            string_similarity = lev.ratio(token1, token2)

            if string_similarity >= string_threshold:
                try:
                    vector_similarity = model.similarity(token1, token2)
                except KeyError:
                    # casefolded token doesn't exist in the vocab
                    # skip
                    continue

                if vector_similarity >= vector_threshold:
                    vector1, vector2 = model[token1], model[token2]

                    yield TokenMatch(token1, token2,
                                     vector1, vector2,
                                     string_similarity,
                                     vector_similarity)


def get_stem_and_diffs(w1: str, w2: str) -> Tuple[str, List[str], List[str]]:
    """
    Given two strings, returns the 'stem' (common part)
    and the respective diffs between the two strings.
    >>> get_stem_and_diffs('suggest', 'suggests')
    ('suggest', ['-'], ['-s'])

    >>> get_stem_and_diffs('erie', 'eerie') # doctest: +SKIP
    ('erie', ['-'], ['e-'])

    :param w1: str
    :param w2: str
    :return:
    """



    # sames and diffs can be non-concatenative
    sames = []
    diffs1 = []
    diffs2 = []

    # NOTE: opcodes can be
    # equal, delete, insert, replace
    opcodes = lev.opcodes(w1, w2)

    # we need to keep track of the current diff
    # because opcode names can differ
    # but, essentially, every operation should be treated
    # as 'replace'
    current_diff1 = []
    current_diff2 = []
    for n, (opcode, w1_start, w1_end,
            w2_start, w2_end) in enumerate(opcodes):
        if opcode == 'equal':

            if current_diff1:
                # prepend the boundary
                # in case this diff is non-concatenative
                current_diff1.append(DIFF_BOUNDARY)
                diff1 = ''.join(current_diff1)
                diffs1.append(diff1)
                current_diff1.clear()

            if current_diff2:
                current_diff2.append(DIFF_BOUNDARY)
                diff2 = ''.join(current_diff2)
                diffs2.append(diff2)
                current_diff2.clear()

            same = w1[w1_start:w1_end]

            # if a diff comes between 'equal' operations
            # then it's a non-concatenative diff
            # we need to reflect that by adding
            # DIFF_BOUNDARY on either end

            # NOTE for sanity:
            # an equal block can never follow another equal block

            sames.append(same)

        else:
            # not 'equal'

            diff1 = w1[w1_start:w1_end]
            diff2 = w2[w2_start:w2_end]

            # if this is not the beginning of a word
            # prepend the boundary character
            if n != 0:
                # only if it hasn't already been prepended
                if not current_diff1:
                    diff1 = DIFF_BOUNDARY + diff1

                if not current_diff2:
                    diff2 = DIFF_BOUNDARY + diff2

            current_diff1.append(diff1)
            current_diff2.append(diff2)

    if current_diff1:
        # append the boundary
        # in case this diff is non-concatenative
        diff1 = ''.join(current_diff1)
        diffs1.append(diff1)

    if current_diff2:
        diff2 = ''.join(current_diff2)
        diffs2.append(diff2)

    # if one of the strings ends with the other
    # (e.g. 'erie' and 'eerie')
    # then Levenshtein opcodes will return ('equal', ... 'insert')
    # we don't want this behaviour
    # because it would complicate things down the line
    # so we need to treat every case like this as a prefix

    # the easiest solution is to reverse the input strings
    # and then reverse the 'same' and diffs back in the output

    if w1.endswith(w2) or w2.endswith(w1):
        same = ''.join(sames)
        # FIXME

    else:
        same = DIFF_BOUNDARY.join(sames)

    return same, diffs1, diffs2


def adjust_stem(stem1: str, stem2: str,
                affixes: Tuple[Collection[str], Collection[str]]):
    """
    >>> affixes = (['-ng', '-ve'], ['-ing', '-s'])
    >>> result = adjust_stem('suggesti', 'suggest', affixes)
    >>> result == ('suggest', 'suggesti', {'-ing', '-ive'})
    True

    :param stem1: str
        The first stem to adjust.
    :param stem2: str
        The second stem to adjust.
    :param affixes: tuple with two collections of strings
        A tuple containing a list of 'affixes' associated
        with each stem
    :return:
    """
    # we assume that shorter stems are 'better'

    # TODO: fix 'junk' affixes retained from earlier 'bad' stems

    better_stem, worse_stem = sorted((stem1, stem2),
                                     key=len)

    # get the index of the better and worse stem
    # in the stems tuple
    better_index = 0 if better_stem == stem1 else 1
    worse_index = int(not better_index)

    # get the difference between the two
    stem, diffs1, diffs2 = get_stem_and_diffs(better_stem, worse_stem)

    # the better stem should be the shorter one
    try:
        assert stem == better_stem
    except AssertionError:
        logging.info(f'expected {better_stem}, got {stem} (other stem: {worse_stem})')
        raise

    # there should be only one diff
    assert len(diffs2) == 1
    # the first character is DIFF_BOUNDARY
    diff = diffs2[0][1:]

    # get the affixes associated with the 'bad' stem
    worse_affixes = affixes[worse_index]

    # adjust these affixes by adding the part
    # that had been wrongly counted as part of the stem
    # e.g. suggesti-ng -> suggest-ing
    adjusted_affixes = set()

    # if it's a suffix, it begins with - (DIFF_BOUNDARY)
    # if it's a prefix, it ends with -
    # if it's an infix (or some non-concatenative morpheme)
    # it will have - on either side
    for affix in worse_affixes:
        affix_parts = affix.split(DIFF_BOUNDARY)

        # infixes shouldn't change
        if affix_parts[0] == '' and affix_parts[-1] == '':
            pass

        # suffix
        elif affix_parts[0] == '':
            significant_part = affix_parts[1]
            significant_part = diff + significant_part
            affix_parts[1] = significant_part

        # prefix
        elif affix_parts[-1] == '':
            significant_part = affix_parts[-1]
            significant_part = significant_part + diff
            affix_parts[-1] = significant_part

        # restore the diff boundary
        adjusted_affix = DIFF_BOUNDARY.join(affix_parts)
        adjusted_affixes.add(adjusted_affix)

    return better_stem, worse_stem, adjusted_affixes


def is_substem(stem1, stem2):
    """
    Returns True if either stem is a substem of the other.
    A stem is a substem if the other stem starts or ends with it.
    >>> is_substem('suggest', 'suggesti')
    True
    >>> is_substem('suggesti', 'suggests')
    False

    :param stem1: str
    :param stem2: str
    :return: bool
    """
    if len(stem1) > len(stem2):
        return is_substem(stem2, stem1)

    return stem2.startswith(stem1) or stem2.endswith(stem1)


def get_cossim(vector1, vector2):
    cossim = cosine_similarity([vector1], [vector2])
    return cossim.ravel()[0]


def stems_match(stem1, stem2,
                stem2vec, threshold=0.75):

    # we need to build a 'ladder' of tests
    # ascending in order of their computational complexity
    shorter = len(stem1) < len(stem2)
    if shorter:
        substem = is_substem(stem1, stem2)
        if substem:
            strings_close = lev.ratio(stem1, stem2) >= threshold
            if strings_close:
                vectors_close = get_cossim(stem2vec[stem1], stem2vec[stem2])
                if vectors_close:
                    return True

    return False



def adjust_stems(stems_to_affixes: Dict[str, Set[str]],
                 stem2vec: Dict[str, Collection]):
    """
    >>> stems_to_affixes = {'suggest': {'-ing', '-s'}, # doctest: +NORMALIZE_WHITESPACE
    ... 'suggesti': {'-ng', '-ve'}}
    >>> stem2vec = {'suggest': [1], 'suggesti': [1]}
    >>> stems_to_affixes = adjust_stems(stems_to_affixes, stem2vec) # doctest: +ELLIPSIS
    ... # doctest: +ELLIPSIS
    >>> stems_to_affixes == {'suggest': {'-s', '-ing', '-ive'}}
    True

    :param stems_to_affixes:
    :param stem2vec:
    :return:
    """

    logging.info('Adjusting stems...')
    logging.info(f'Starting with {len(stems_to_affixes)}...')

    has_changed = True

    while has_changed:
        adjusted_s2a = stems_to_affixes.copy()
        old_size = len(adjusted_s2a)
        has_changed = False

        for candidate in stems_to_affixes:

            # check if there are stems that are similar
            # get all stems that are shorter
            # and have a high Levenshtein ratio
            # with this stem
            shorter_stems = [stem for stem in adjusted_s2a
                             if stems_match(stem, candidate, stem2vec)]

            if shorter_stems:
                has_changed = True

                for stem in shorter_stems:
                    affixes = stems_to_affixes[stem], stems_to_affixes[candidate]

                    good_stem, worse_stem, adjusted_affixes = adjust_stem(stem, candidate,
                                                                          affixes)
                    new_stem = good_stem
                    this_signature = adjusted_affixes
                    del adjusted_s2a[candidate]
                    adjusted_s2a[new_stem].update(this_signature)
                    break

        stems_to_affixes = adjusted_s2a
        new_size = len(adjusted_s2a)
        logging.info(f'Reduced to {new_size} (from {old_size}).')
        if has_changed:
            logging.info('Iterating again...')


    logging.info(f'Reduced to {len(stems_to_affixes)} stems.')

    return stems_to_affixes


def build_signatures(matches: Iterable[TokenMatch]):
    logging.info('Building signatures from matches...')

    stem2vec = {}

    # stems is a dictionary
    # where every stem points to an ID (index)
    # of a signature
    # if a more optimal (shorter) stem is found
    # all of its signatures can be adjusted
    # to reflect the changed stem
    stems_to_affixes = defaultdict(set)

    for match in matches:
        same, diffs1, diffs2 = get_stem_and_diffs(match.token1, match.token2)
        this_signature = diffs1 + diffs2
        assert isinstance(this_signature, list)

        combined_vector = (match.vector1 + match.vector2) / 2
        new_stem = same

        # add the stem and its signature to stems
        stems_to_affixes[new_stem].update(this_signature)

        stem2vec[new_stem] = combined_vector

    stems_to_affixes = adjust_stems(stems_to_affixes, stem2vec)

    # affixes is a dictionary
    # where every individual affix
    # points to an id of a
    affixes_to_stems = defaultdict(set)
    for stem, affixes in stems_to_affixes.items():
        affixes_to_stems[str(affixes)].add(stem)

    stem_counter = Counter({stem: len(affixes)
                            for stem, affixes in stems_to_affixes.items()})
    affix_counter = Counter({affixes: len(stems)
                             for affixes, stems in affixes_to_stems.items()})

    return stems_to_affixes, affixes_to_stems, stem_counter, affix_counter





def main():
    args = get_args()

    logging.info(f'Loading the model from {args.model}')
    model = Word2Vec.load(args.model)
    matches = get_matches(model)

    s2a, a2s, s_counter, a_counter = build_signatures(matches)

    with open('stems2affixes.txt', 'w') as f:
        pprint(s2a, stream=f)

    with open('affixes2stems.txt', 'w') as f:
        pprint(a2s, stream=f)

    with open('stems_counts.txt', 'w') as f:
        pprint(s_counter.most_common(), stream=f)

    with open('affix_counts.txt', 'w') as f:
        pprint(a_counter.most_common(), stream=f)




    # with open('matches.txt', 'w') as f:
    #     for match in matches:
    #         print(match, file=f)
    #         print(file=f)
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()

    logging.basicConfig(format='%{message}s', level=logging.INFO)
    # TODO: fix
    logging.info = print

    main()
