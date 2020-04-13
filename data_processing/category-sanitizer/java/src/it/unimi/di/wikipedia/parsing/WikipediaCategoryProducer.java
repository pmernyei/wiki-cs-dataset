package it.unimi.di.wikipedia.parsing;

import it.unimi.di.big.mg4j.document.Document;
import it.unimi.di.big.mg4j.document.DocumentIterator;
import it.unimi.di.wikipedia.utils.IntMapGraph;
import it.unimi.di.wikipedia.utils.MapUtils;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.io.BinIO;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.webgraph.BVGraph;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Reader;
import java.util.Scanner;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.NullOutputStream;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPResult;
import com.martiansoftware.jsap.Parameter;
import com.martiansoftware.jsap.SimpleJSAP;
import com.martiansoftware.jsap.Switch;
import com.martiansoftware.jsap.UnflaggedOption;

public class WikipediaCategoryProducer {
	
	public static Logger LOGGER = LoggerFactory.getLogger(WikipediaCategoryProducer.class);
	
	private static final int CATEGORY_NAME_INDEX = "Category:".length();
	private static final int UNSEEN_CATEGORY = -1;
	private static final int CATEGORY_FIELD = 3;
	private static final String SEPARATOR_REGEX = "OXOXO";
	private static final int ESITMATED_NUM_OF_PAGES = 6721260; // this is needed only to log progress 




	private final NamespacedWikipediaDocumentSequence wikipediaDocumentSequence;
	private final Int2ObjectMap<String> pageId2Name;
	private final Object2IntMap<String> catName2Id;
	private final Int2ObjectMap<IntSet> page2cat;
	private final Int2ObjectOpenHashMap<IntSet> cat2cat;
	private PrintStream plainUrisFile = new PrintStream(NullOutputStream.NULL_OUTPUT_STREAM);
	
	private int nextPageId;
	private int nextCategoryId;

	WikipediaCategoryProducer(NamespacedWikipediaDocumentSequence wds) {
		this.wikipediaDocumentSequence = wds;
		catName2Id = new Object2IntOpenHashMap<String>();
		catName2Id.defaultReturnValue(UNSEEN_CATEGORY);
		pageId2Name = new Int2ObjectOpenHashMap<String>();
		page2cat = new Int2ObjectOpenHashMap<IntSet>();
		cat2cat = new Int2ObjectOpenHashMap<IntSet>();
		nextPageId = 0;
		nextCategoryId = 0;
	}
	

	private static enum Namespace {
		ARTICLE, OTHER, CATEGORY;
	}
	
	private Namespace getNamespace(String title) {
		int pos = title.indexOf(':');
		if (pos < 0) return Namespace.ARTICLE;
		String namespace = title.substring(0, pos);
		return (namespace.toLowerCase().equals("category")) ?
			Namespace.CATEGORY
		: (
			(wikipediaDocumentSequence.isATrueNamespace(namespace)) ?
				Namespace.OTHER
			:	Namespace.ARTICLE
		);
	}
	
	private int getCategoryId(String category) {
		int categoryId = catName2Id.getInt(category);
		if (categoryId == UNSEEN_CATEGORY) {
			categoryId = this.nextCategoryId++;
			catName2Id.put(category, categoryId);
		}
		return categoryId;
	}
	
	private IntSet parseCategories(Document wikiPage) throws IOException {
			String categoryString = IOUtils.toString((Reader) wikiPage.content(CATEGORY_FIELD));
			IntSet categoryIds = new IntOpenHashSet();
			int pipeIndex;
			
			for (String category : categoryString.split(SEPARATOR_REGEX)) {
				if ((pipeIndex = category.indexOf('|')) > -1)
					category = category.substring(0, pipeIndex);
				
				category = StringUtils.strip(category);
				if (category.length() > 0)
					categoryIds.add(getCategoryId(category));
			}
			
			return categoryIds;
	}

	@SuppressWarnings("resource") // the warning on wikiPage is false.
	public void extractAllData() throws IOException {
		DocumentIterator wikiPagesIterator = wikipediaDocumentSequence.iterator();
		Document wikiPage;
		String title;
		
		ProgressLogger pl = new ProgressLogger(LOGGER, "pages");
		pl.expectedUpdates = ESITMATED_NUM_OF_PAGES; 
		pl.info = new Object() {
			public String toString() {return catName2Id.size() + " categories found.";}
		};
		pl.start("Starting to iterate all the pages in the XML file...");
		
		// iterating pages
		while ((wikiPage = wikiPagesIterator.nextDocument()) != null) {
			switch (getNamespace(title = wikiPage.title().toString())) {
				case CATEGORY:
					cat2cat.put(
							getCategoryId(title.substring(CATEGORY_NAME_INDEX)),
							parseCategories(wikiPage)
							);
					break;
				case ARTICLE:
					int pageId = nextPageId++;
					pageId2Name.put(pageId, title);
					
					page2cat.put(pageId, 
							parseCategories(wikiPage)
							);
					plainUrisFile.println(wikiPage.uri());
					
					break;
				default:
					break;
			}
			
			wikiPage.close();
			pl.update();
			
		}
		pl.done();
		
		wikipediaDocumentSequence.close();
		wikiPagesIterator.close();
	}
	
	private void setPlainUrisFile(String path) throws FileNotFoundException {
		this.plainUrisFile = new PrintStream(new File(path));
	}
	
	private void saveAllTo(String basename) throws Exception {
		try {
			BinIO.storeObject(pageId2Name, basename + "pageId2Name.ser");
			BinIO.storeObject(catName2Id, basename + "catName2Id.ser");
			BinIO.storeObject(page2cat, basename + "page2cat.ser");
			
			if (cat2cat.isEmpty()) LOGGER.error("THE PARSING DID NOT FIND ANY CATEGORY PSEUDOTREE");
			else BVGraph.store(new IntMapGraph(cat2cat), "categoryPseudotree");
			
			BinIO.storeObject(MapUtils.invert(catName2Id), basename + "catId2Name.ser");
		} catch (IOException e) {
			LOGGER.error("Cannot save something :( :(", e);
		}
		this.plainUrisFile.close();
	}
	
	static String askForString(String s) {
		System.out.println(s);
		Scanner scanner = new Scanner(System.in);
		String nextLine = scanner.nextLine();
		scanner.close();
		return nextLine;
	}
	
	public static void main(String[] rawArguments) throws Exception {
		
		SimpleJSAP jsap = new SimpleJSAP(
				WikipediaCategoryProducer.class.getName(),
				"Read a wikipedia dump and produces these files as " +
				"serialized Java objects: \n" +
				" * pageId2Name.ser, an Int2ObjectMap from page ids to " +
				"wikipedia page names \n" +
				" * catId2Name.ser, an Object2IntMap from category ids to " +
				"category names \n" +
				" * page2cat.ser, an Int2ObjectMap from page ids to an IntSet" +
				"of category ids \n" + 
				" * categoryPseudotree.graph, the Wikipedia Category Hierarchy "
				+ "in BVGraph format",
				new Parameter[] {
			new UnflaggedOption( "input", JSAP.STRING_PARSER, JSAP.REQUIRED,
						"The pages-articles.xml input file, from Wikipedia." ),
			new UnflaggedOption( "basename", JSAP.STRING_PARSER, JSAP.REQUIRED,
						"The basename of the output files (p.e. a Directory with / in the end)" ),
			new Switch("bzip", 'z', "bzip", "Interpret the input file as bzipped"),
			new Switch("verbose", 'v', "verbose", "Print every category found to StdErr") 
		});
		
		// Initializing input read
		JSAPResult args = jsap.parse( rawArguments );
		if ( jsap.messagePrinted() ) System.exit( 1 );
		
		NamespacedWikipediaDocumentSequence wikipediaDocumentSequence = new NamespacedWikipediaDocumentSequence(
				args.getString("input"),
				args.getBoolean("bzip"),
				"http://en.wikipedia.org/wiki/", true,
				 true // keep all namespaces
			);
		WikipediaCategoryProducer reader =
				new WikipediaCategoryProducer(wikipediaDocumentSequence);
		reader.setPlainUrisFile(args.getString("basename") + "pages.uris");
		reader.extractAllData();
		reader.saveAllTo(args.getString("basename"));
		wikipediaDocumentSequence.close();
		
	}


}
