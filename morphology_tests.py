import unittest
import doctest
import morphology


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(morphology))
    return tests

