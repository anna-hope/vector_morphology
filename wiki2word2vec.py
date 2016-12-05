from argparse import ArgumentParser
from itertools import tee
import logging
from os import cpu_count
import tarfile

from bs4 import BeautifulSoup
from gensim.models.word2vec import Word2Vec
import nltk
import nltk.data
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def get_sents_from_xml(xml_obj, sent_detector):
    contents = xml_obj.read()
    soup = BeautifulSoup(contents, 'lxml')
    for string in soup.stripped_strings:
        sents = sent_detector.tokenize(string)
        for sent in sents:
            yield nltk.word_tokenize(sent)


def get_sents_from_corpus(fp):
    logging.info(f'loading the corpus {fp}')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    with tarfile.open(fp) as corpus:
        for member in tqdm(corpus.getmembers()):
            with corpus.extractfile(member) as f:
                file_sents = get_sents_from_xml(f, sent_detector)
                for sent in file_sents:
                    yield sent


def train_model(corpus_fp, model_fp):
    sents = get_sents_from_corpus(corpus_fp)
    sents1, sents2 = tee(sents, 2)

    logging.info('Training the model...')
    model = Word2Vec(sg=1, workers=cpu_count(),
                     max_vocab_size=int(4e7))
    model.build_vocab(sents1)
    model.train(sents2)
    model.init_sims(replace=True)
    logging.info(f'Saving the model to {model_fp}...')
    model.save(model_fp)
    logging.info('Done.')


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--corpus', required=True)
    arg_parser.add_argument('-mp', '--model-path', required=True)
    args = arg_parser.parse_args()
    train_model(args.corpus, args.model_path)
   

if __name__ == '__main__':
    main()
