import re
import pandas as pd

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


FILE_IN = '..\\output\\page2cat.tsv'
FILE_OUT = '..\\output\\category_sizes.tsv'

counts = {}
idx = 0
for line in open(FILE_IN, encoding='utf8'):
    titles = re.split(r'\t+', line)
    for i in range(1, len(titles)-1):
        title = titles[i]
        counts[title] = counts.get(title, 0) + 1
    idx += 1
    if idx % 100000 == 0:
        print(idx, 'pages counted')

df = pd.DataFrame([{'category': k, 'pages': v} for (k,v) in counts.items()])
df = df.sort_values(by='pages', ascending=False)
df.to_csv(FILE_OUT, sep='\t', encoding='utf-8', index=False)
print('Counted', len(counts), 'category sizes')
