# Vector Morphology

Vector morphology is an experimental project that aims to use word vector
embeddings to discover the morphology of a language in an unsupervised way.

While this project attempts to find an accurate description of a language's
morphemes as a whole, it was started in particular to discover
non-concatenative morphological patterns (such as [strong
verbs](https://en.wikipedia.org/wiki/Germanic_strong_verb)). While this
project already succeeds with some examples, it is still a work in progress,
and therefore much work needs to be done to refine the output.

## Why word vectors?

[Word vectors](https://en.wikipedia.org/wiki/Word2vec) provide an important
semantic dimension to algorithms that try to find morphological patterns based
on strings, and help eliminate noise (for instance, false positive 'patterns'
like *string-strong*). By comparing not only string tokens, but their
associated vectors, we can more accurately say that a given pair of words
likely belongs to the same paradigm.

## Where do you get the vectors?

For testing this project, a [gensim](http://radimrehurek.com/gensim/)
skip-gram model was trained on the 
[Wikicorpus](http://www.cs.upc.edu/~nlp/wikicorpus/). It is not clear whether a
skip-gram model's accuracy exceeds that of a continuous bag of words for
the purposes of this project. What is clear (and, perhaps, obvious), is that
the corpus on which the model is trained needs to be quite large -- the larger,
the more accurate the results will be.


Just like the rest of this project, this README is still early and rough. If
the project proves viable, this description will be expanded with more details
on how the code works and how one should use it.
