#!/usr/bin/env python3

"""
Utility/convenience script to perform heuristic splitting on dense text. Uses a
prefix tree for *very* fast lookups. Using PyPy, you can cut the initialisation
stage from around 0.736s to 0.487s (around 51.13%).
"""

import sys
import argparse
import textwrap
import string
import time
import re
import pathlib

from collections import defaultdict

letter_set = set(string.ascii_lowercase + "_")

DATA_DIR = pathlib.Path(__file__).parent / "data"

WORD_REGEX = r"[^.?!:;]+|[.?!:;]+"
ALLOWED_CHARS = set(string.ascii_lowercase + ".?!:;")

def strip_stream(plain):
    return "".join(re.findall("[a-z]", plain.read().lower()))

def strip_punc(word):
    """
    Strip punctuation from word
    """
    return "".join(ch for ch in word.lower() if ch in letter_set)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=argparse.FileType("r"), help="input file")
    parser.add_argument("-w", "--watch", action="store_true",
                            help="continually re-process text")
    parser.add_argument("-t", "--time", type=float, default=1,
                            help="seconds to wait in between refresh")
    return parser.parse_args()

class PrefixTree:
    """
    A prefix tree class that can be used to quickly find a word in a set of
    characters.
    """
    def __init__(self):
        # the dictionary of children - automatically create new tree when
        # needed
        self.children = defaultdict(PrefixTree)
        # track if this tree represents the end of a known word
        self.is_end = False

    def copy(self):
        t = PrefixTree()
        t.children = defaultdict(PrefixTree,
                                 {k: v.copy() for k, v in self.children.items()})
        t.is_end = self.is_end
        return t

    # add a word to the tree, using recursive haskell-esque style
    def add_word(self, word, pos=0):
        if pos < len(word):
            # get first character and remainder, add to child
            self.children[word[pos]].add_word(word, pos + 1)
            if word[pos] == "_":
                self.is_end = True
        else:
            # if the word is empty, this is the end node
            self.is_end = True

    # remove word from tree, by unsetting flag
    def remove_word(self, word, pos=0):
        # if this is the end node of this word
        if pos == len(word):
            self.is_end = False
        # delegate to child nodes
        else:
            self.children[word[pos]].remove_word(word, pos + 1)

    # easy repr of tree
    def __repr__(self):
        return "PrefixTree(end={!r}, children={!r})".format(self.is_end, self.children)

    # a slightly more tree-like representation of the tree
    def __str__(self):
        return textwrap.indent(
                        "\n".join(
                            "└{}─{}".format(k, str(v).lstrip())
                                    for k, v in self.children.items()),
                        "  ")

    # get the longest word you can find from a point in a collection of
    # characters. returns next word.
    def longest_word_from(self, chars, pos):
        # if the position is within bounds
        if pos < len(chars):
            # get the character to inspect
            currchar = chars[pos]
            # check if any space-overloaded words are known
            space_res = ""
            if self.is_end:
                space_res = " {}".format(self.children["_"].longest_word_from(chars, pos)).rstrip()
            # try to obtain a result from children (the longest possible word)
            if currchar not in self.children:
                return space_res
            else:
                child_result = self.children[currchar].longest_word_from(chars, pos + 1)
                if not child_result:
                    if not self.children[currchar].is_end:
                        return space_res
                child_result = "{}{}".format(currchar, child_result)
            return max(child_result,
                       space_res,
                       key=len)
        return ""

def build_pt(args):
    """
    Build a prefix tree from data/words (local copy of /usr/share/dict/words
    """
    preftree = PrefixTree()
    # build tree from words
    with open(DATA_DIR / "words") as wordfile:
        for word in map(strip_punc, wordfile):
            preftree.add_word(word)
    return preftree

PREV_EXTRA_HASH = object()

class NothingChanged(Exception):
    pass

def amend_pt(preftree):
    global PREV_EXTRA_HASH
    # special cased words, includes extra, deletions and space overloading
    with open(DATA_DIR / "extra_words") as extrafile:
        extra_words = [w for w in map(str.lower, map(str.strip, extrafile))
                if not w.startswith("#")]
        if hash(tuple(extra_words)) == PREV_EXTRA_HASH:
            raise NothingChanged
        else:
            PREV_EXTRA_HASH = hash(tuple(extra_words))
            new_tree = preftree.copy()
            for word in extra_words:
                if not word.startswith("#"):
                    if word.startswith("^"):
                        new_tree.remove_word(word, 1)
                    else:
                        new_tree.add_word(word, 0)
            return new_tree

def split_words(preftree, dense_str):
    """
    Split words from dense string into separate words
    """
    made_lower = dense_str.lower()
    pos = 0
    while pos < len(dense_str) - 1:
        nxt = preftree.longest_word_from(made_lower, pos)
        yield nxt
        pos += len(strip_punc(nxt))

def get_words(segments, working_tree):
    for segment in segments:
        if segment.isalpha():
            yield from split_words(working_tree, segment)
        else:
            yield segment

def main(args):
    start = time.time()
    print("initialising..")
    preftree = build_pt(args)
    print("initialised from data/words (took {:.3f} secs)"
            .format(time.time() - start))
    input_text = "".join(c for c in args.input.read().lower()
                         if c in ALLOWED_CHARS)
    segments = re.findall(WORD_REGEX, input_text)
    while True:
        try:
            working_tree = amend_pt(preftree)
            print("amended from data/extra_words")
        except NothingChanged:
            print(".", end="", flush=True)
        else:
            print(" ".join(get_words(segments, working_tree)))
            if not args.watch:
                break
        time.sleep(args.time)

if __name__ == "__main__":
    main(parse_args())
