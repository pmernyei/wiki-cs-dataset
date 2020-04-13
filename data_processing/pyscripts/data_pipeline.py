"""
Extract all relevant page data for a dataset, process it into appropriate form
for training models and calculate statistics.

Needs as input:
- Label mapping: list of labels with associated Wikipedia categories for dataset.
- Wikipedia datadumps of relevant tables preprocessed into CSVs
    (see preprocess_mysqldumps.py)
- Sanitized category data (as output by the sanitizer tool)
- Article texts (as output by the text extractor)
- Word embeddings (in the GloVe embedding file format)

Outputs into a given directory:
- Extracted data about nodes as a map from IDs to WikiDataNode objects in
    .pickle form (fulldata.pickle)
- Data for training and evaluation in appropriate splits in vector form
    (vectors.json)
- Readable data equivalent to fulldata.pickle in JSON form (readable.json) with
    the node order corresponding to that of vectors.json
- Dataset statistics (analysis.txt)
"""

import sys
import os
import argparse
from extract_full_data_for_dataset import extract_by_single_mapping_file
from process_dataset import process_with_glove_vectors
from analyze_datasets import analyze

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create Wikipedia node classification dataset from sources')
    parser.add_argument('--label-mapping',
        help='JSON file giving label mapping')
    parser.add_argument('--wiki-dump-dir',
        help='Directory containing preprocessed page, pagelinks, redirect '
             'table CSVs')
    parser.add_argument('--category-data-dir',
        help='Directory containing sanitized category data')
    parser.add_argument('--text-data-dir',
        help='Directory containing extracted article texts')
    parser.add_argument('--glove-embedding-file', help='Word embedding file')
    parser.add_argument('--output-dir', help='Directory to write results to')

    args = parser.parse_args()

    extract_by_single_mapping_file(
        args.label_mapping,
        os.path.join(args.category_data_dir, 'page2cat.tsv'),
        os.path.join(args.wiki_dump_dir, 'page.csv'),
        os.path.join(args.wiki_dump_dir, 'pagelinks.csv'),
        os.path.join(args.wiki_dump_dir, 'redirect.csv'),
        args.text_data_dir,
        args.output_dir
    )
    process_with_glove_vectors(args.output_dir, args.glove_embedding_file)
    analyze(args.output_dir)
