WIKIDUMP_XML=../data/input_dumps/enwiki-20190820-pages-articles.xml
N_TOP_CATEGORIES=10000

{ java it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer --help 2>&1 | grep -q "Could not find or load main class"; } && { echo 'The jar containing it.unimi.di.wikipedia.* was not found in classpath. Please, have a look at the "Compile the code" part of readme.md.' ; exit 1; }

# java -Xmx6G it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer $WIKIDUMP_XML ../data/10-12/category_outputs/

java -Xmx6G it.unimi.di.wikipedia.categories.CategorySelector ../data/category_outputs/10-12/categoryPseudotree \
  output/page2cat.ser ../data/category_outputs/10-12/pageId2Name.ser ../data/category_outputs/10-12/catId2Name.ser \
  -e "wiki" -e "categories" -e "main topic classifications" -e "template" -e "navigational box" \
  $N_TOP_CATEGORIES ../data/category_outputs/10-26/ranked-categories.tsv ../data/category_outputs/10-26/page2cat.tsv ../data/category_outputs/10-26/milestonetree.tsv
