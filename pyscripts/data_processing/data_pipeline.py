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
from extract_full_data_for_dataset import load_mappings_by_file
from process_dataset import process_with_glove_vectors
from analyze_datasets import analyze

if __name__ == '__main__':
    label_mapping_filename = sys.argv[1]
    wiki_dump_dir = sys.argv[2]
    category_data_dir = sys.argv[3]
    text_data_dir = sys.argv[4]
    glove_filename = sys.argv[5]
    output_dir = sys.argv[6]
    load_mappings_by_file(label_mapping_filename,
        os.path.join(category_data_dir, 'page2cat.tsv'),
        os.path.join(wiki_dump_dir, 'page.csv'),
        os.path.join(wiki_dump_dir, 'pagelinks.csv'),
        os.path.join(wiki_dump_dir, 'redirect.csv'),
        text_data_dir,
        output_dir)
    process_with_glove_vectors(output_dir, glove_filename)
    analyze(output_dir)
