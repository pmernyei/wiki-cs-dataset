WIKIDUMP_XML=../data/input_dumps/enwiki-20190820-pages-articles.xml
N_TOP_CATEGORIES=10000

{ java it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer --help 2>&1 | grep -q "Could not find or load main class"; } && { echo 'The jar containing it.unimi.di.wikipedia.* was not found in classpath. Please, have a look at the "Compile the code" part of readme.md.' ; exit 1; }

# java -Xmx6G it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer $WIKIDUMP_XML ../data/10-12/category_outputs/

java -Xmx6G it.unimi.di.wikipedia.categories.CategorySelector ../data/category-outputs/categoryPseudotree \
  ../data/category-outputs/page2cat.ser ../data/category-outputs/pageId2Name.ser ../data/category-outputs/catId2Name.ser \
  -e "wiki" -e "categories" -e "main topic classifications" -e "template" -e "navigational box" \
  -p "msx" -p "x86 operating systems" -p "ship registration" -p "browsers" -p "cloud applications" -p "cloud computing providers" \
  $N_TOP_CATEGORIES ../data/category-outputs/improved-rerun/ranked-categories.tsv ../data/category-outputs/improved-rerun/page2cat.tsv ../data/category-outputs/improved-rerun/milestonetree.tsv
