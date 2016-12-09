from argparse import ArgumentParser
from collections import namedtuple, defaultdict, Counter
from difflib import SequenceMatcher
from enum import Enum
from itertools import combinations
import logging
from math import factorial
from pathlib import Path
from pprint import pprint
from statistics import mean
from typing import Iterable, Dict, Set, Tuple, List, Collection

import Levenshtein as lev
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tqdm import tqdm

TokenMatch = namedtuple('TokenMatch', ['token1', 'token2',
                                       'vector1', 'vector2',
                                       'string_similarity',
                                       'vector_similarity'])


class Affix(Enum):
    Prefix = 0
    Suffix = 1
    Infix = 2
    Circumfix = 3
    Transfix = 4


WORD_START = '*'
WORD_END = '#'
DIFF_BOUNDARY = '-'
STEM_BOUNDARY = '_'

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    return args


def are_similar(string_similarity, vector_similarity,
                vector_weight=2, string_weight=1,
                threshold=0.7):

    # compute the similarity
    # as a weighted mean of the Levenshtein ratio
    # and vector cosine similarity
    # at a ratio of 2:1

    similarity_num = (vector_similarity * vector_weight +
                      string_similarity * string_weight)

    similarity_denom = vector_weight + string_weight
    similarity = similarity_num / similarity_denom
    return similarity >= threshold


def get_matches(model, string_threshold=0.75, combined_threshold=0.7):
    counts = (entry.count for entry in model.vocab.values())
    mean_count = mean(counts)

    vocab = sorted({word.casefold() for word, entry 
                    in model.vocab.items()
                    if entry.count > mean_count * 2})

    num_combinations = factorial(len(vocab)) // 2 
    num_combinations //= factorial(len(vocab) - 2)

    seq_matcher = SequenceMatcher()
    for token1, token2 in tqdm(combinations(vocab, 2),
                               total=num_combinations):
        if token1.isalpha() and token2.isalpha():

            # initial check to make sure
            # we are not comparing completely implausible things
            if lev.ratio(token1, token2) >= 0.5:

                # we will now use Sequence Matcher
                # because its algorithm is often better
                # for purposes of picking up non-concatenative similarity
                # (e.g Semitic)

                # SequenceMatcher caches the second sequence
                # and, with combinations, the first token will occur more often
                if seq_matcher.b != token1:
                    seq_matcher.set_seq2(token1)

                seq_matcher.set_seq1(token2)

                string_similarity = seq_matcher.quick_ratio()

                if string_similarity >= string_threshold:
                    try:
                        vector_similarity = model.similarity(token1, token2)
                    except KeyError:
                        # casefolded token doesn't exist in the vocab
                        # skip
                        continue

                    passes_threshold = are_similar(string_similarity,
                                                   vector_similarity,
                                                   threshold=combined_threshold)

                    if passes_threshold:
                        vector1, vector2 = model[token1], model[token2]

                        yield TokenMatch(token1, token2,
                                         vector1, vector2,
                                         string_similarity,
                                         vector_similarity)


def get_same_and_diffs(w1: str, w2: str) -> Tuple[str, List[str], List[str]]:
    """
    Given two strings, returns the 'stem' (common part)
    and the respective diffs between the two strings.
    >>> get_same_and_diffs('suggest', 'suggests')
    ('suggest', ['-'], ['-s'])

    >>> get_same_and_diffs('erie', 'eerie')
    ('erie', ['-'], ['e-'])
    >>> get_same_and_diffs('q_t_l', 'q_tlit')  # doctest: +SKIP
    ('q_t_l', ['---'], ['---', '-i-'])

    :param w1: str
    :param w2: str
    :return:
    """

    seq_matcher = SequenceMatcher(a=w1, b=w2)

    # sames and diffs can be non-concatenative
    sames = []
    diffs1 = []
    diffs2 = []

    # NOTE: opcodes can be
    # equal, delete, insert, replace
    # opcodes = lev.opcodes(w1, w2)
    opcodes = seq_matcher.get_opcodes()

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
                # or prefix
                if current_diff1[-1] != DIFF_BOUNDARY:
                    current_diff1.append(DIFF_BOUNDARY)

                diff1 = ''.join(current_diff1)
                diffs1.append(diff1)
                current_diff1.clear()

            if current_diff2:
                if current_diff2[-1] != DIFF_BOUNDARY:
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

    same = DIFF_BOUNDARY.join(sames)  # test

    return same, diffs1, diffs2


# def adjust_stem(stem1: str, stem2: str,
#                 affixes: Tuple[Collection[str], Collection[str]]):
#     """
#     >>> affixes = (['-ng', '-ve'], ['-ing', '-s'])
#     >>> result = adjust_stem('suggesti', 'suggest', affixes)
#     >>> result == ('suggest', 'suggesti', {'-ing', '-ive'})
#     True
#
#     :param stem1: str
#         The first stem to adjust.
#     :param stem2: str
#         The second stem to adjust.
#     :param affixes: tuple with two collections of strings
#         A tuple containing a list of 'affixes' associated
#         with each stem
#     :return:
#     """
#     # we assume that shorter stems are 'better'
#
#     # TODO: fix 'junk' affixes retained from earlier 'bad' stems
#
#     better_stem, worse_stem = sorted((stem1, stem2),
#                                      key=len)
#
#     # get the index of the better and worse stem
#     # in the stems tuple
#     better_index = 0 if better_stem == stem1 else 1
#     worse_index = int(not better_index)
#
#     # get the difference between the two
#     stem, diffs1, diffs2 = get_same_and_diffs(better_stem, worse_stem)
#
#     # the better stem should be the shorter one
#     try:
#         assert stem == better_stem
#     except AssertionError:
#         logging.info(f'expected {better_stem}, got {stem} (other stem: {worse_stem})')
#         raise
#
#     # there should be only one diff
#     assert len(diffs2) == 1
#     # the first character is DIFF_BOUNDARY
#     diff = diffs2[0][1:]
#
#     # get the affixes associated with the 'bad' stem
#     worse_affixes = affixes[worse_index]
#
#     # adjust these affixes by adding the part
#     # that had been wrongly counted as part of the stem
#     # e.g. suggesti-ng -> suggest-ing
#     adjusted_affixes = set()
#
#     # if it's a suffix, it begins with - (DIFF_BOUNDARY)
#     # if it's a prefix, it ends with -
#     # if it's an infix (or some non-concatenative morpheme)
#     # it will have - on either side
#     for affix in worse_affixes:
#         affix_parts = affix.split(DIFF_BOUNDARY)
#
#         # infixes shouldn't change
#         if affix_parts[0] == '' and affix_parts[-1] == '':
#             pass
#
#         # suffix
#         elif affix_parts[0] == '':
#             significant_part = affix_parts[1]
#             significant_part = diff + significant_part
#             affix_parts[1] = significant_part
#
#         # prefix
#         elif affix_parts[-1] == '':
#             significant_part = affix_parts[-1]
#             significant_part = significant_part + diff
#             affix_parts[-1] = significant_part
#
#         # restore the diff boundary
#         adjusted_affix = DIFF_BOUNDARY.join(affix_parts)
#         adjusted_affixes.add(adjusted_affix)
#
#     return better_stem, worse_stem, adjusted_affixes


def adjust_stem(stem1: str, stem2: str,
                words: Tuple[Set[str], Set[str]]):
    """
    >>> words = ({'suggesting', 'suggestive'},
    ... {'suggesting', 'suggests'})
    >>> result = adjust_stem('suggesti', 'suggest', words)
    >>> result == ('suggest', 'suggesti',
    ...  {'suggesting', 'suggestive', 'suggests'})
    True

    :param stem1: str
        The first stem to adjust.
    :param stem2: str
        The second stem to adjust.
    :param words: tuple with two collections of strings
        A tuple containing a list of 'affixes' associated
        with each stem
    :return:
    """
    # we assume that shorter stems are 'better'

    better_stem, worse_stem = sorted((stem1, stem2),
                                     key=len)

    # get the index of the better and worse stem
    # in the stems tuple
    better_index = 0 if better_stem == stem1 else 1
    worse_index = int(not better_index)

    # get the difference between the two
    stem, diffs1, diffs2 = get_same_and_diffs(better_stem, worse_stem)

    # the better stem should be the shorter one
    try:
        assert stem == better_stem
    except AssertionError:
        logging.info(f'expected {better_stem}, got {stem} (other stem: {worse_stem})')
        raise

    # combine the words
    combined_words = words[better_index]
    combined_words.update(words[worse_index])

    return better_stem, worse_stem, combined_words


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


def get_vector_mean(*vectors: Iterable[np.ndarray]) -> np.ndarray:
    vector_sum = sum(vectors)
    return vector_sum / len(vectors)


def stems_match(stem1, stem2,
                stem2vecs, threshold=0.75):
    """
    >>> stem1 = 'suggest'
    >>> stem2 = 'suggesti'
    >>> stem2vecs = {stem1: [np.array([1])], stem2: [np.array([1])]}
    >>> stems_match(stem1, stem2, stem2vecs)
    True
    >>> stem1 = 'become'
    >>> stem2 = 'became'
    >>> stem2vecs = {stem1: [np.array([0.85])]}

    :param stem1:
    :param stem2:
    :param stem2vecs:
    :param threshold:
    :return:
    """

    # we need to build a 'ladder' of tests
    # ascending in order of their computational complexity
    # they could also be equal to account for things like 'come'/'came'

    shorter = len(stem1) <= len(stem2)
    if shorter and stem1 != stem2:
        substem = is_substem(stem1, stem2)
        if substem:
            string_similarity = lev.ratio(stem1, stem2)
            if string_similarity >= threshold:

                # FIXME: temporary, until we detect outliers
                mean1 = get_vector_mean(*stem2vecs[stem1])
                mean2 = get_vector_mean(*stem2vecs[stem2])

                vector_similarity = get_cossim(mean1, mean2)
                pass_threshold = are_similar(string_similarity,
                                             vector_similarity,
                                             threshold=threshold)
                if pass_threshold:
                    return True

    return False


# def adjust_stems(stems_to_affixes: Dict[str, Set[str]],
#                  stem2vec: Dict[str, Collection]):
#     """
#     >>> stems_to_affixes = {'suggest': {'-ing', '-s'}, # doctest: +NORMALIZE_WHITESPACE
#     ... 'suggesti': {'-ng', '-ve'}}
#     >>> stem2vec = {'suggest': [1], 'suggesti': [1]}
#     >>> stems_to_affixes = adjust_stems(stems_to_affixes, stem2vec) # doctest: +ELLIPSIS
#     ... # doctest: +ELLIPSIS
#     >>> stems_to_affixes == {'suggest': {'-s', '-ing', '-ive'}}
#     True
#
#     :param stems_to_affixes:
#     :param stem2vec:
#     :return:
#     """
#
#     logging.info('Adjusting stems...')
#     logging.info(f'Starting with {len(stems_to_affixes)}...')
#
#     has_changed = True
#
#     while has_changed:
#         adjusted_s2a = stems_to_affixes.copy()
#         old_size = len(adjusted_s2a)
#         has_changed = False
#
#         for candidate in stems_to_affixes:
#
#             # check if there are stems that are similar
#             # get all stems that are shorter
#             # and have a high Levenshtein ratio
#             # with this stem
#             shorter_stems = [stem for stem in adjusted_s2a
#                              if stems_match(stem, candidate, stem2vec)]
#
#             if shorter_stems:
#                 has_changed = True
#
#                 for stem in shorter_stems:
#                     affixes = stems_to_affixes[stem], stems_to_affixes[candidate]
#
#                     good_stem, worse_stem, adjusted_affixes = adjust_stem(stem, candidate,
#                                                                           affixes)
#                     new_stem = good_stem
#                     this_signature = adjusted_affixes
#                     del adjusted_s2a[candidate]
#                     adjusted_s2a[new_stem].update(this_signature)
#                     break
#
#         stems_to_affixes = adjusted_s2a
#         new_size = len(adjusted_s2a)
#         logging.info(f'Reduced to {new_size} (from {old_size}).')
#         if has_changed:
#             logging.info('Iterating again...')
#
#     logging.info(f'Reduced to {len(stems_to_affixes)} stems.')
#
#     return stems_to_affixes


# def build_signatures(matches: Iterable[TokenMatch]):
#     logging.info('Building signatures from matches...')
#
#     stem2vec = {}
#
#     # stems is a dictionary
#     # where every stem points to an ID (index)
#     # of a signature
#     # if a more optimal (shorter) stem is found
#     # all of its signatures can be adjusted
#     # to reflect the changed stem
#     stems_to_affixes = defaultdict(set)
#
#     for match in matches:
#         same, diffs1, diffs2 = get_same_and_diffs(match.token1, match.token2)
#         this_signature = diffs1 + diffs2
#         assert isinstance(this_signature, list)
#
#         combined_vector = (match.vector1 + match.vector2) / 2
#         new_stem = same
#
#         # add the stem and its signature to stems
#         stems_to_affixes[new_stem].update(this_signature)
#
#         stem2vec[new_stem] = combined_vector
#
#     stems_to_affixes = adjust_stems(stems_to_affixes, stem2vec)
#
#     # affixes is a dictionary
#     # where every individual affix
#     # points to an id of a
#     affixes_to_stems = defaultdict(set)
#     for stem, affixes in stems_to_affixes.items():
#         affixes_to_stems[str(affixes)].add(stem)
#
#     stem_counter = Counter({stem: len(affixes)
#                             for stem, affixes in stems_to_affixes.items()})
#     affix_counter = Counter({affixes: len(stems)
#                              for affixes, stems in affixes_to_stems.items()})
#
#     return stems_to_affixes, affixes_to_stems, stem_counter, affix_counter

def get_stems(matches: Iterable[TokenMatch]):
    logging.info('Building sets of stems and words...')

    # stems is a dictionary
    # where every stem points to an ID (index)
    # of a signature
    # if a more optimal (shorter) stem is found
    # all of its signatures can be adjusted
    # to reflect the changed stem
    stem2words = defaultdict(set)

    stem2vecs = defaultdict(list)

    # a set to store which tokens' vectors
    # we have already stored
    # to make sure we don't fill stem2vecs with duplicates
    # this will be stored as tuples (stem, token)
    # because we want to account for vectors
    # occurring with different stems
    stored_vecs = set()

    for match in matches:
        same, diffs1, diffs2 = get_same_and_diffs(match.token1, match.token2)
        # this_signature = diffs1 + diffs2
        # assert isinstance(this_signature, list)

        new_stem = same

        # add the stem and its signature to stems
        stem2words[new_stem].update((match.token1, match.token2))

        if (new_stem, match.token1) not in stored_vecs:
            stem2vecs[new_stem].append(match.vector1)
        if (new_stem, match.token2) not in stored_vecs:
            stem2vecs[new_stem].append(match.vector2)
    
    return stem2words, stem2vecs


def consolidate_stems(stems2words: Dict[str, Set[str]],
                      stem2vecs: Dict[str, Collection]):
    """
    >>> stems_to_words = {'suggest': {'suggesting', 'suggests'},
    ... 'suggesti': {'suggesting', 'suggestive'}} # doctest: +ELLIPSIS
    >>> stem2vecs = {'suggest': [np.array([1])], 'suggesti': [np.array([1])]}
    >>> stems_to_affixes = consolidate_stems(stems_to_words, stem2vecs) # doctest: +ELLIPSIS
    ... # doctest: +ELLIPSIS
    >>> stems_to_affixes == {'suggest': {'suggests', 'suggesting', 'suggestive'}}
    True

    :param stems2words:
    :param stem2vec:
    :return:
    """

    logging.info('Adjusting stems...')
    logging.info(f'Starting with {len(stems2words)}...')

    has_changed = True

    while has_changed:
        adjusted_s2a = stems2words.copy()
        old_size = len(adjusted_s2a)
        has_changed = False

        for candidate in stems2words:

            # check if there are stems that are similar
            # get all stems that are shorter
            # and have a high Levenshtein ratio
            # with this stem
            shorter_stems = [stem for stem in adjusted_s2a
                             if stems_match(stem, candidate, stem2vecs)]

            if shorter_stems:
                has_changed = True

                for stem in shorter_stems:
                    words = stems2words[stem], stems2words[candidate]

                    good_stem, worse_stem, words = adjust_stem(stem, candidate,
                                                               words)
                    new_stem = good_stem

                    # update the words and the associated stems

                    # new_vector = get_vector_mean(stem2vec[stem], stem2vec[candidate])
                    adjusted_s2a[new_stem].update(words)

                    # update the vector average for the stem
                    # stem2vec[new_stem] = new_vector

                    del adjusted_s2a[candidate]
                    # del stem2vec[candidate]

                    break

        stems2words = adjusted_s2a
        new_size = len(adjusted_s2a)
        logging.info(f'Reduced to {new_size} (from {old_size}).')
        if has_changed:
            logging.info('Iterating again...')

    logging.info(f'Reduced to {len(stems2words)} stems.')

    return stems2words


def get_affix_type(affix: str) -> Affix:
    """
    >>> affix_type = get_affix_type(f's{DIFF_BOUNDARY}') # doctest: +ELLIPSIS
    ... # doctest: +ELLIPSIS
    >>> affix_type == Affix.Prefix
    True
    >>> affix_type = get_affix_type(f'{DIFF_BOUNDARY}s')
    >>> affix_type == Affix.Suffix
    True
    >>> affix_type = get_affix_type(f'{DIFF_BOUNDARY}s{DIFF_BOUNDARY}')
    >>> affix_type == Affix.Infix
    True
    >>> affix_type = get_affix_type(f's{DIFF_BOUNDARY}s')
    >>> affix_type == Affix.Circumfix
    True
    >>> db = DIFF_BOUNDARY
    >>> affix = f'{db}a{db}u{db}'  # transfix (e.g. Arabic)
    >>> affix_type = get_affix_type(affix)
    >>> affix_type == Affix.Transfix
    True
    >>> affix = '{db}'
    >>> get_affix_type(affix)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError:

    :param affix: str
    :return: Affix
    """

    split_affix = affix.split(DIFF_BOUNDARY)
    # print(affix, split_affix)

    # transfix is the special case
    # most likely, you will have more than 3 splits
    # because there will be more than two diff boundaries
    # e.g. '-a-u-' will turn into ['', 'a', '', 'u', '']
    if len(split_affix) < 2:
        raise ValueError(f'"{affix}" matches no pattern.')

    elif len(split_affix) > 3:
        return Affix.Transfix
    else:
        first, last = split_affix[0], split_affix[-1]

        if first and not last:
            # prefix
            return Affix.Prefix
        elif last and not first:
            # suffix
            return Affix.Suffix
        elif not first and not last:
            # infix
            return Affix.Infix
        elif first and last:
            return Affix.Circumfix
        else:
            raise ValueError(f'"{affix}" matches no pattern.')


def build_signatures(stems2words: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Builds signatures from stems and their associated words.
    :param stems2words:
    :return:
    """

    stems2affixes = defaultdict(set)

    for stem, words in stems2words.items():

        prefixes = []  # \s+DIFF_BOUNDARY
        suffixes = []  # DIFF_BOUNDARY\s+
        circumfixes = []  # \s+DIFF_BOUNDARY\s+
        infixes = []  # DIFF_BOUNDARY\s+DIFF_BOUNDARY

        # NOTE: a circumfix is a prefix+suffix combination
        # that seems to only occur together

        # sanity check
        assert stem not in stems2affixes

        # NOTE: check for diff boundary?

        for word in words:
            same, diffs1, diffs2 = get_same_and_diffs(stem, word)

            # assert same == stem, f'Stem is not the same (expected {stem}, got {same})'

            # TODO: build a list of all diffs
            # and iterate over them to prevent duplicates
            if diffs1:
                stems2affixes[stem].add(DIFF_BOUNDARY.join(diffs1))
            if diffs2:
                stems2affixes[stem].add(DIFF_BOUNDARY.join(diffs2))

    return stems2affixes


def get_stems_and_signatures(matches: Iterable[TokenMatch]):
    logging.info('Building signatures from matches...')

    stem2words, stem2vec = get_stems(matches)

    # reduce the stems to the most robust ones
    # and combine the words from each stem in the process
    stem2words = consolidate_stems(stem2words, stem2vec)

    # build the signatures for each stem
    stems_to_affixes = build_signatures(stem2words)

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

    output_path = Path('output')

    logging.info(f'Loading the model from {args.model}')
    model = Word2Vec.load(args.model)
    matches = get_matches(model)

    s2a, a2s, s_counter, a_counter = get_stems_and_signatures(matches)

    s2a_path = Path(output_path, 'stems2affixes.txt')
    a2s_path = Path(output_path, 'affixes2stems.txt')
    s_counts_path = Path(output_path, 'stem_counts.txt')
    a_counts_path = Path(output_path, 'affix_counts.txt')

    with s2a_path.open('w') as f:
        pprint(s2a, stream=f)

    with a2s_path.open('w') as f:
        pprint(a2s, stream=f)

    with s_counts_path.open('w') as f:
        pprint(s_counter.most_common(), stream=f)

    with a_counts_path.open('w') as f:
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
