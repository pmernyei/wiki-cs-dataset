"""
Extracts relevant node data for pages in given categories (mapped to node
classes) from Wikipedia database tables (preprocessed to CSV form) and outputs
of the category sanitizer tool.

Outputs the extracted datasets into .pickle files.
"""

import csv
import os
import json
import nltk
import string
import pickle
import sys
from datetime import datetime
from wiki_node import WikiDataNode
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nktk.download('stopwords')


def page_titles_to_ids(titles_set, page_table_filename):
    """
    Parse data from the Wikipedia page database table to get page IDs for
    the given set of page titles.
    """
    mapping = {}
    with open(page_table_filename, encoding='utf8') as page_file:
        reader = csv.reader(page_file)
        for line in reader:
            id, namespace, title = int(line[0]), line[1], line[2]
            if namespace == '0' and title in titles_set:
                mapping[title] = id
    return mapping


def page_titles_to_labels(category_label_mapping, page2cat_filename):
    """
    Given the sets of categories for each label, parse the mapping from page
    titles to categories and output the page titles mapped to labels.
    """
    category_to_label = {}
    for (name, category_list) in category_label_mapping.items():
        for category in category_list:
            if category not in category_to_label:
                category_to_label[category] = []
            category_to_label[category].append(name)

    titles_to_labels = {}
    with open(page2cat_filename, encoding='utf8') as page2cat_file:
        reader = csv.reader(page2cat_file, delimiter='\t')
        for line in reader:
            title = line[0].replace(" ", "_")
            label_lists = ([category_to_label[line[i]]
                        for i in range(1, len(line))
                        if line[i] in category_to_label])
            labels = {label for one_list in label_lists for label in one_list}
            if len(labels) == 1:
                titles_to_labels[title] = next(iter(labels))

    return titles_to_labels


def load_redirects(page_table_filename, redirect_table_filename):
    """
    Load the redirect table as a mapping between page IDs. Double redirects not
    handled because they are considered invalid by Wikipedia's standards.
    """
    target_title_to_source_id = {}
    with open(redirect_table_filename, encoding='utf8') as redirect_file:
        reader = csv.reader(redirect_file)
        for from_id, to_namespace, to_title, _, _ in reader:
            from_id = int(from_id)
            if to_namespace == '0':
                if to_title not in target_title_to_source_id:
                    target_title_to_source_id[to_title] = []
                target_title_to_source_id[to_title].append(from_id)

    source_to_target_id = {}
    with open(page_table_filename, encoding='utf8') as page_file:
        reader = csv.reader(page_file)
        for line in reader:
            id, title, is_redirect = int(line[0]), line[2], line[5]
            if is_redirect == '0': # Ignore double redirects
                for source_id in target_title_to_source_id.get(title, []):
                    source_to_target_id[source_id] = id

    return source_to_target_id


def links_between_pages(page_id_set, pagelinks_table_filename,
                        page_table_filename, redirect_table_filename):
    """
    Produce the graph of hyperlinks between nodes with page IDs in the given
    set, taking into account redirects.
    """

    # Map target title to source IDs
    titles_linked_from = {}
    with open(pagelinks_table_filename, encoding='utf8') as pagelinks_file:
        reader = csv.reader(pagelinks_file)
        for from_id, from_namespace, to_title, to_namespace in reader:
            from_id = int(from_id)
            if (from_id in page_id_set and
                from_namespace == '0' and to_namespace == '0'):
                if to_title not in titles_linked_from:
                    titles_linked_from[to_title] = []
                titles_linked_from[to_title].append(from_id)

    # Load ID to ID redirects
    redirects = load_redirects(page_table_filename, redirect_table_filename)

    # Produce ID to ID links using by matching titles to IDs in the pages table
    links = {id: [] for id in page_id_set}
    with open(page_table_filename, encoding='utf8') as page_file:
        reader = csv.reader(page_file)
        for line in reader:
            id, title, is_redirect = int(line[0]), line[2], line[5]
            if is_redirect == '1':
                if id in redirects:
                    id = redirects[id]
                else:
                    continue

            for source_id in titles_linked_from.get(title, []):
                links[source_id].append(id)

    return links


def get_text_tokens(page_id_set, text_extractor_data_dir):
    """
    Process extracted text data to get tokenized text for pages with given IDs.
    """
    ids_to_tokens = {}
    sw = stopwords.words('english')+['""', "''", '``', "'s"]
    for root, dirs, files in os.walk(text_extractor_data_dir):
        for file in files:
            for line in open(os.path.join(root, file), "r", encoding='utf8'):
                entry = json.loads(line)
                id = int(entry['id'])
                if id in page_id_set:
                    ids_to_tokens[id] = (
                        [t.lower() for t in nltk.word_tokenize(entry['text'])
                            if t not in string.punctuation
                            and t.lower() not in sw]
                    )
    return ids_to_tokens


def load_with_multiple_label_maps(label_mapping_list, page2cat_filename,
                                page_table_filename, pagelinks_table_filename,
                                redirect_table_filename, text_extractor_data,
                                output_dir=None, output_names=None):
    """
    Extract muliple datasets defined by mapping sets of categories to labels.

    Each dataset is defined by a list of labels with categories corresponding to
    each label. The source files are read only once to load all data relevant
    for all label mappings, then output each dataset to a .pickle file as a map
    from node IDs to node objects, location given by output_dir.

    If the output_dir parameter is not given, the function returns the results
    without saving them to file.

    Optionally output_names can be given to name each dataset, otherwise they
    will be numbered.
    """
    # Get titles to mapped to labels for each label dataset
    print(datetime.now().strftime('%H:%M:%S'), 'Loading page titles for labels...')
    multi_titles_to_labels = [
        page_titles_to_labels(label_mapping, page2cat_filename)
                                for label_mapping in label_mapping_list
    ]
    # Get titles and map them to page IDs
    print(datetime.now().strftime('%H:%M:%S'), 'Mapping titles to page IDs...')
    all_titles = set([]).union(*[titles_to_labels.keys()
                                for titles_to_labels in multi_titles_to_labels])
    all_titles_to_ids = page_titles_to_ids(all_titles, page_table_filename)
    all_ids_to_titles = {v:k for (k,v) in all_titles_to_ids.items()}
    all_ids = set(all_titles_to_ids.values())

    # Load link and text data
    print(datetime.now().strftime('%H:%M:%S'), 'Loading links between pages...')
    all_links = links_between_pages(
        all_ids, pagelinks_table_filename,
        page_table_filename, redirect_table_filename
    )

    print(datetime.now().strftime('%H:%M:%S'), 'Loading and tokenizing text...')
    all_ids_to_tokens = get_text_tokens(all_ids, text_extractor_data)

    # Get ID sets of valid pages with no data missing for each individual dataset
    print(datetime.now().strftime('%H:%M:%S'), 'Filtering IDs...')
    all_valid_links = {
        source: [target for target in outlinks
                    if (target in all_links and target in all_ids_to_tokens)]
        for (source, outlinks) in all_links.items()
    }

    id_sets = [{all_titles_to_ids[title] for title in titles_to_labels.keys()
                    if (title in all_titles_to_ids
                    and all_titles_to_ids[title] in all_valid_links
                    and all_titles_to_ids[title] in all_ids_to_tokens)}
                for titles_to_labels in multi_titles_to_labels]

    # Create datasets as a list of sets of Node objects
    print(datetime.now().strftime('%H:%M:%S'), 'Creating final sets...')
    result = [
        {
            id: WikiDataNode(
                id,
                all_ids_to_titles[id],
                multi_titles_to_labels[i][all_ids_to_titles[id]],
                [out_id for out_id in all_valid_links[id]
                    if out_id in id_sets[i]],
                all_ids_to_tokens[id]
            )
            for id in id_sets[i]
        }
        for i in range(len(id_sets))
    ]
    print(datetime.now().strftime('%H:%M:%S'), 'Created datasets.')
    if output_dir is not None:
        if output_names is not None:
            for name in output_names:
                os.mkdir(os.path.join(output_dir, name))
        print(datetime.now().strftime('%H:%M:%S'), 'Writing to file...')
        for i in range(len(result)):
            print(datetime.now().strftime('%H:%M:%S'),
                'Writing dataset', i, '...')
            out_filename = (os.path.join(output_dir, 'ds_'+str(i))
                if output_names is None
                else os.path.join(output_dir, output_names[i],
                                    'fulldata.pickle'))
            with open(out_filename, 'wb') as output:
                pickle.dump(result[i], output, -1)
    return result


def load_single_dataset(label_mapping, page2cat_filename, page_table_filename,
    pagelinks_table_filename, redirect_table_filename, text_extractor_data):
    return load_with_multiple_label_maps([label_mapping], page2cat_filename,
        page_table_filename, pagelinks_table_filename,
        redirect_table_filename, text_extractor_data)


def extract_by_single_mapping_file(mappings_filename, page2cat_filename,
                            page_table_filename, pagelinks_table_filename,
                            redirect_table_filename, text_extractor_data,
                            output_dir):
    """
    Extract a single dataset based on a label mapping specified in the given
    JSON file. The result will be written to output_dir/fulldata.pickle.
    """
    with open(mappings_filename, 'r') as file:
        mapping = json.load(file)
    load_with_multiple_label_maps(
        [mapping], page2cat_filename, page_table_filename,
        pagelinks_table_filename, redirect_table_filename, text_extractor_data,
        os.path.dirname(os.path.normpath(output_dir)),
        [os.path.basename(os.path.normpath(output_dir))]
    )


def extract_by_multiple_mappings_file(mappings_filename, page2cat_filename,
                            page_table_filename, pagelinks_table_filename,
                            redirect_table_filename, text_extractor_data,
                            output_dir):
    """
    Load datasets based on label mappings specified in a JSON file storing with
    the top level object mapping dataset names to label mappings.

    The datasets will be written to child directories of output_dir named
    according to these keys.
    """
    with open(mappings_filename, 'r') as file:
        mappings = json.load(file)
        names = list(mappings.keys())
        mapping_list = [mappings[name] for name in names]
    load_with_multiple_label_maps(
        mapping_list, page2cat_filename, page_table_filename,
        pagelinks_table_filename, redirect_table_filename, text_extractor_data,
        output_dir, names
    )


if __name__ == '__main__':
    load_mappings_by_file(sys.argv[1],
        os.path.join(sys.argv[2], 'category-outputs', 'page2cat.tsv'),
        os.path.join(sys.argv[2], 'input-dumps', 'page.csv'),
        os.path.join(sys.argv[2], 'input-dumps', 'pagelinks.csv'),
        os.path.join(sys.argv[2], 'input-dumps', 'redirect.csv'),
        os.path.join(sys.argv[2], 'text-extraction'),
        sys.argv[3])
