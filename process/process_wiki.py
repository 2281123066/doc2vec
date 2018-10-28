#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys

from gensim.corpora import WikiCorpus

if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    i = 0
    inp = 'enwiki-latest-pages-articles1.xml-p10p30302.bz2'
    outp = 'wiki_en.txt'
    output = open(outp, 'w', encoding='utf-8')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i = i + 1
        if i % 10000 == 0:
            logger.info('Saved ' + str(i) + ' articles')

    output.close()
    logger.info('Finished ' + str(i) + ' articles')

