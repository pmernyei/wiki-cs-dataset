WIKIDUMP_XML=../data/wiki_dumps/enwiki-20190820-pages-articles.xml
N_TOP_CATEGORIES=10000

{ java it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer --help 2>&1 | grep -q "Could not find or load main class"; } && { echo 'The jar containing it.unimi.di.wikipedia.* was not found in classpath. Please, have a look at the "Compile the code" part of readme.md.' ; exit 1; }

# java -Xmx6G it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer $WIKIDUMP_XML ../data/sanitized_categories/

java -Xmx6G it.unimi.di.wikipedia.categories.CategorySelector ../data/sanitized_categories/categoryPseudotree \
  ../data/sanitized_categories/page2cat.ser ../data/sanitized_categories/pageId2Name.ser ../data/sanitized_categories/catId2Name.ser \
  -e "wiki" -e "categories" -e "main topic classifications" -e "template" -e "navigational box" \
  -p "msx" -p "x86 operating systems" -p "ship registration" -p "browsers" -p "cloud applications" -p "cloud computing providers" \
  $N_TOP_CATEGORIES ../data/sanitized_categories/ranked-categories.tsv ../data/sanitized_categories/page2cat.tsv ../data/sanitized_categories/milestonetree.tsv
