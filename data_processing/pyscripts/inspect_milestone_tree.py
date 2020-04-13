"""
Calculate stats about hierarchy of selected milestone categories and print the
tree hierarchy structure.
"""
import csv
import sys
import json
import re
import pandas as pd

def get_category_sizes(page2cat_filename, output_filename=None):
    """
    Calculates how many pages are mapped to each category, optionally outputting
    as a CSV file besides returning it as a dictionary.
    """

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    counts = {}
    idx = 0
    for line in open(page2cat_filename, encoding='utf8'):
        titles = re.split(r'\t+', line)
        for i in range(1, len(titles)-1):
            title = titles[i]
            counts[title] = counts.get(title, 0) + 1
        idx += 1
        if idx % 100000 == 0:
            print(idx, 'pages counted')

    print('Counted', len(counts), 'category sizes')
    if output_filename is not None:
        df = pd.DataFrame([{'category': k, 'pages': v} for (k,v) in counts.items()])
        df = df.sort_values(by='pages', ascending=False)
        df.to_csv(FILE_OUT, sep='\t', encoding='utf-8', index=False)

    return counts


def calculate_milestone_tree(subcats_filename, sizes_filename=None,
                            page2cat_filename=None, out_filename=None):
    """
    Builds a tree based on the category parent relations, and calculates
    statistics for subtrees.
    """
    raw_sizes = {}
    parent = {}
    children = {}
    roots = []

    if sizes_filename is None and page2cat_filename is None:
        print('Error, give precomputed sizes or page2cat file')
        return None

    if sizes_filename is None:
        raw_sizes = get_category_sizes(page2cat_filename)
    else:
        with open(sizes_filename) as size_file:
            reader = csv.reader(size_file, delimiter='\t')
            next(reader, None) # Skip header
            for category, size in reader:
                raw_sizes[category] = int(size)


    with open(subcats_filename) as tree_file:
        reader = csv.reader(tree_file, delimiter='\t')
        for category, p in reader:
            if p != 'null':
                parent[category] = p
                if p in children:
                    children[p].append(category)
                else:
                    children[p] = [category]
            else:
                roots.append(category)

    aggregated_sizes = {}

    def aggregate(cat):
        size = raw_sizes.get(cat, 0)
        for child in children.get(cat, []):
            size += aggregate(child)
        aggregated_sizes[cat] = size
        return size

    for root in roots:
        aggregate(root)

    result = {
        'raw_sizes': raw_sizes,
        'parent': parent,
        'children': children,
        'roots': roots,
        'aggregated_sizes': aggregated_sizes
    }

    if out_filename:
        with open(out_filename, 'w+', encoding='utf8') as out:
            print_tree('Articles', result, output=out)
    return result


def get_all_descendants(root, children_map):
    """
    Returns all descendants in the tree of a given root node, recursively
    visiting them based on the map from parents to children.
    """
    return {root}.union(
        *[get_all_descendants(child, children_map)
            for child in children_map.get(root, [])]
    )


def print_tree(root, milestone_tree, output=sys.stdout, indent_level=0):
    """
    Pretty prints the category hierarchy from a given root node.
    """
    indent = '    ' * indent_level
    output.write(indent + '{} ({} pages, ~{} in subtree)'.format(
        root, milestone_tree['raw_sizes'].get(root, 0),
        milestone_tree['aggregated_sizes'].get(root, 0)))
    if root in milestone_tree['children']:
        output.write(' {\n')
        for ch in milestone_tree['children'].get(root, []):
            print_tree(ch, milestone_tree, output, indent_level+1)
        output.write(indent + '}\n')
    else:
        output.write('\n')


if __name__ == '__main__':
    calculate_milestone_tree(sys.argv[1], page2cat_filename=sys.argv[2],
                                out_filename=sys.argv[3])
