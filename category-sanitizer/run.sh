WIKIDUMP_XML=enwiki-20190820-pages-articles.xml
N_TOP_CATEGORIES=10000

{ java it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer --help 2>&1 | grep -q "Could not find or load main class"; } && { echo 'The jar containing it.unimi.di.wikipedia.* was not found in classpath. Please, have a look at the "Compile the code" part of readme.md.' ; exit 1; }

# java -Xmx6G it.unimi.di.wikipedia.parsing.WikipediaCategoryProducer $WIKIDUMP_XML ./

java -Xmx6G it.unimi.di.wikipedia.categories.CategorySelector categoryPseudotree \
  page2cat.ser pageId2Name.ser catId2Name.ser \
  -e "wiki" -e "categories" -e "main topic classifications" -e "template" -e "navigational box" \
  $N_TOP_CATEGORIES ranked-categories.tsv page2cat.tsv
