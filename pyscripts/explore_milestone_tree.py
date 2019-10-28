import csv
import sys
import json

def calculate_milestone_tree(sizes_filename, subcats_filename, out_filename=None):
    raw_sizes = {}
    parent = {}
    children = {}
    roots = []

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
        open(out_filename, 'w+', encoding='utf8').write(json.dumps(result))
    return result


if __name__ == '__main__':
    calculate_milestone_tree(sys.argv[1], sys.argv[2], sys.argv[3])
