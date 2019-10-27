import csv
import sys
import json

raw_sizes = {}
parent = {}
children = {}
roots = []

with open(sys.argv[1]) as size_file:0
    reader = csv.reader(size_file, delimiter='\t')
    next(reader, None) # Skip header
    for category, size in reader:
        raw_sizes[category] = int(size)

with open(sys.argv[2]) as tree_file:
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

with open('tmp/data.json', 'w+', encoding='utf8') as outfile:
    outfile.write(json.dumps({'raw_sizes': raw_sizes, 'parent': parent, 'children': children, 'roots': roots, 'aggregated_sizes': aggregated_sizes}))
