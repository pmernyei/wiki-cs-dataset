import os

_data_root = os.path.join('..', 'data')
_input_dumps = os.path.join(_data_root, 'input-dumps')
_category_outputs = os.path.join(_data_root, 'category-outputs')

page2cat = os.path.join(_category_outputs, 'page2cat.tsv')
milestone_tree = os.path.join(_category_outputs, 'milestonetree.tsv')
page_table = os.path.join(_input_dumps, 'page.csv')
pagelinks_table = os.path.join(_input_dumps, 'pagelinks.csv')
redirect_table = os.path.join(_input_dumps, 'redirect.csv')
text_extractor_data = os.path.join(_data_root, 'text-extraction')
