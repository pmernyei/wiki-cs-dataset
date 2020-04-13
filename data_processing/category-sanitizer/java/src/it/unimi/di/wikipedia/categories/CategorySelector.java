package it.unimi.di.wikipedia.categories;

import it.unimi.di.wikipedia.utils.MapUtils;
import it.unimi.dsi.Util;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.io.BinIO;
import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.Transform;
import it.unimi.dsi.webgraph.algo.GeometricCentralities;

import java.io.IOException;
import java.io.PrintWriter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.Arrays;

import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPResult;
import com.martiansoftware.jsap.Parameter;
import com.martiansoftware.jsap.SimpleJSAP;
import com.martiansoftware.jsap.UnflaggedOption;

import it.unimi.dsi.fastutil.ints.IntArrayPriorityQueue;
import it.unimi.dsi.fastutil.ints.IntPriorityQueue;


public class CategorySelector {
	final static Logger LOGGER = LoggerFactory.getLogger(CategorySelector.class);

	// Input data
	private final ImmutableGraph wcg, transposedWcg;
	private final Int2ObjectMap<String> catId2name;
	public final int numOriginalCat, numFinalCat;
	public final String[] excludedStrings;
	public final String[] prunedStrings;

	// Output data
	public int[] orderedCatIds;
	private Int2DoubleMap catId2rank;
	private IntSet milestones, excludedCatIds;

	private String milestoneTreeFile;

	public CategorySelector(ImmutableGraph wcg, Int2ObjectMap<String> catId2name, int numFinalCat, String[] excludedStrings, String[] prunedStrings, String milestoneTreeFile) {
		this.wcg = wcg;
		this.transposedWcg = Transform.transpose(wcg);
		this.catId2name = catId2name;
		this.numOriginalCat = wcg.numNodes();
		this.numFinalCat = numFinalCat;
		this.excludedStrings = excludedStrings;
		this.prunedStrings = prunedStrings;
		this.milestoneTreeFile = milestoneTreeFile;

		LOGGER.debug("Examples from the provided Wikipedia Category Graph: ");
		for (int i = 0; i < 10; i++) {
			int cat = (int) (Math.random() * numOriginalCat);
			LOGGER.debug( "\"" + catId2name.get(cat) + "\" is listed as a subcategory of \""
					+ catId2name.get(this.wcg.successors(cat).nextInt()) + "\"");
		}
	}

	private static IntSet findCategoriesContainingStrings(final Int2ObjectMap<String> catId2name, final String[] lowercasedString) {
		IntSet results = new IntOpenHashSet();
		String name;
		for (Int2ObjectMap.Entry<String> c2n : catId2name.int2ObjectEntrySet()) {
			name = c2n.getValue().toLowerCase();
			for (String string : lowercasedString)
				if (name.indexOf(string) != -1) {
					results.add(c2n.getIntKey());
					break;
				}
		}
		return results;
	}

	public void compute() {
		LOGGER.info("Ranking nodes...");
		final GeometricCentralities ranker = new GeometricCentralities(transposedWcg, new ProgressLogger(LOGGER));
		try {
			ranker.compute();
		} catch (InterruptedException e) { throw new RuntimeException(e); }
		catId2rank = new Int2DoubleOpenHashMap(Util.identity(numOriginalCat), ranker.harmonic);
		LOGGER.info("Nodes ranked.");

		LOGGER.info("Excluding categories containing " + Arrays.toString(excludedStrings) + "...");
		excludedCatIds = findCategoriesContainingStrings(catId2name, excludedStrings);
		for (int catIdToExclude : excludedCatIds)
			catId2rank.put(catIdToExclude, Double.NEGATIVE_INFINITY);
		LOGGER.info(excludedCatIds.size() + " categories excluded, e.g. \"" + catId2name.get(excludedCatIds.toIntArray()[0]) + "\".");

		LOGGER.info("Ordering categories by centrality and selecting milestones...");
		orderedCatIds = Util.identity(numOriginalCat);
		IntArrays.quickSort(orderedCatIds, MapUtils.comparatorPuttingLargestMappedValueFirst(catId2rank));
		milestones = new IntOpenHashSet(IntArrays.trim(orderedCatIds, numFinalCat));
		LOGGER.info(milestones.size() + " milestones selected. 1st category: " + catId2name.get(orderedCatIds[0]));
	}

	public void outputMilestoneHierarchy(final int[] closestMilestones) {
		try {
			PrintWriter printWriter = new PrintWriter(milestoneTreeFile);
			for (int m : milestones) {
				printWriter.print(catId2name.get(m));
				printWriter.print("\t");
				printWriter.print(catId2name.get(closestMilestones[m]));
				printWriter.println();
			}
			printWriter.close();
		} catch (IOException e) {
			LOGGER.error("Failed to output milestone tree data!");
		}

	}

	public Int2ObjectMap<IntSet> recategorize(final Int2ObjectMap<IntSet> page2cat) {
		LOGGER.info("Pruning categories from aggregation containing " + Arrays.toString(prunedStrings) + "...");
		IntSet prunedCategoryIds = findCategoriesContainingStrings(catId2name, prunedStrings);
		for (int id : prunedCategoryIds) {
			LOGGER.info("Pruning category " + catId2name.get(id));
		}
		LOGGER.info("Computing closest milestones...");
		final int[] closestMilestones = new ImprovedHittingDistanceMinimizer(
			transposedWcg, milestones, prunedCategoryIds).compute();
		LOGGER.info("Closest milestones computed, printing a sample:");
		for (int i = 0; i < 10; i++) {
			int cat = (int) (Math.random() * numOriginalCat);
			System.out.println( "\"" + catId2name.get(cat) + "\" has been ramapped to \""
					+ catId2name.get(closestMilestones[cat]) + "\"");
		}
		outputMilestoneHierarchy(closestMilestones);
		// Milestones had to point to others for outputting their connections,
		// but for the purpose of page remapping they should point to themselves
		for (int milestone : milestones) {
			closestMilestones[milestone] = milestone;
		}

		ProgressLogger pl = new ProgressLogger(LOGGER, "pages");
		pl.expectedUpdates = page2cat.size();
		pl.start("Moving old categories to closest milestones...");
		Int2ObjectMap<IntSet> page2newCat = new Int2ObjectOpenHashMap<IntSet>(page2cat.size());
		for (Int2ObjectMap.Entry<IntSet> p2c : page2cat.int2ObjectEntrySet()) {
			IntSet newCategories = new IntOpenHashSet();
			int milestone;
			for (int cat : p2c.getValue()) {
				if (cat < 0 || cat >= numOriginalCat)
					LOGGER.error("Category #" + cat + " is not listed in the Wikipedia Category Graph"
							+ " (it has only " + numOriginalCat + " nodes).");
				else {
					milestone = closestMilestones[cat];
					if (milestone != -1) {
						if (!milestones.contains(milestone))
							throw new IllegalStateException(milestone + " is not a milestone.");
						newCategories.add(milestone);
					}
				}
			}
			page2newCat.put(p2c.getIntKey(), newCategories);
			pl.lightUpdate();
		}
		pl.done();

		return page2newCat;
	}

	private String[] toSortedNames(IntSet categories) {
		String[] names = new String[categories.size()];
		int[] sortedCat = categories.toIntArray();
		IntArrays.quickSort(sortedCat, MapUtils.comparatorPuttingLargestMappedValueFirst(catId2rank));
		for (int i = 0; i < sortedCat.length; i++) names[i] = catId2name.get(sortedCat[i]);
		return names;
	}

	@SuppressWarnings({ "unchecked" })
	public static void main( String rawArguments[] ) throws Exception  {
		SimpleJSAP jsap = new SimpleJSAP( CategorySelector.class.getName(),
				"Cleanse the wikipedia categorization system.",
				new Parameter[] {
						new UnflaggedOption( "WCG",
							JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY,
							"The BVGraph basename of the wikipedia category graph." ),
						new UnflaggedOption( "page2cat",
								JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY,
								"The serialized int 2 intset that represents set of categories for each page." ),
						new UnflaggedOption( "pageNames",
								JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY,
								"The serialized Int2ObjectMap<String> file with association of categories to their names." ),
						new UnflaggedOption( "catNames",
								JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY,
								"The serialized Int2ObjectMap<String> file with association of categories to their names." ),
						new FlaggedOption( "exclude",
								JSAP.STRING_PARSER, null, JSAP.NOT_REQUIRED,
								'e', "exclude",
								"Exclude all those categories whose LOWERCASED name contains one of the provided strings." )
								.setAllowMultipleDeclarations(true),
						new FlaggedOption( "prune",
								JSAP.STRING_PARSER, null, JSAP.NOT_REQUIRED,
								'p', "prune",
								"Exclude all those categories whose LOWERCASED name contains one of the provided strings." )
								.setAllowMultipleDeclarations(true),
						new UnflaggedOption( "C",
								JSAP.INTEGER_PARSER, "10000", JSAP.REQUIRED, JSAP.NOT_GREEDY,
								"Number of categories to retain." ),
						new UnflaggedOption( "output-rankedcat",
								JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY,
								"Where the output (ordered) category 2 score TSV file will be saved."
								),
						new UnflaggedOption( "output-page2cat",
								JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY,
								"Where the output page2cat TSV file will be saved."
								),
						new UnflaggedOption( "output-milestonetree",
								JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY,
								"Where the output milestone tree TSV file will be saved."
								),
					}
				);

		final JSAPResult args = jsap.parse( rawArguments );
		if ( jsap.messagePrinted() ) System.exit( 1 );

		LOGGER.info("Reading input files...");
		Int2ObjectMap<String> catNames = (Int2ObjectMap<String>) BinIO.loadObject(args.getString("catNames"));
		Int2ObjectMap<String> pageNames = (Int2ObjectMap<String>) BinIO.loadObject(args.getString("pageNames"));
		Int2ObjectMap<IntSet> page2cat = (Int2ObjectMap<IntSet>) BinIO.loadObject(args.getString("page2cat"));
		ImmutableGraph wcg = ImmutableGraph.load(args.getString("WCG"));
		final int numFinalCat = args.getInt("C");

		CategorySelector categorySelector = new CategorySelector(
			wcg, catNames, numFinalCat, args.getStringArray("exclude"),
			args.getStringArray("prune"), args.getString("output-milestonetree"));
		categorySelector.compute();

		LOGGER.info("Writing rankings to " + args.getString("output-rankedcat") + "...");
		PrintWriter out = new PrintWriter(args.getString("output-rankedcat"));
		for (int c : categorySelector.orderedCatIds) {
			out.print(catNames.get(c));
			out.print("\t");
			out.print(Double.toString(categorySelector.catId2rank.get(c)));
			out.println();
		}
		out.close();


		Int2ObjectMap<IntSet> newPage2cat = categorySelector.recategorize(page2cat);

		out = new PrintWriter(args.getString("output-page2cat"));
		ProgressLogger pl = new ProgressLogger(LOGGER, "pages");
		pl.expectedUpdates = newPage2cat.size();
		pl.start("Writing new page2cat map to " + args.getString("output-page2cat") + "...");
		for (Int2ObjectMap.Entry<IntSet> p2c : newPage2cat.int2ObjectEntrySet()) {
			out.print(pageNames.get(p2c.getIntKey()));
			out.print("\t");
			for (String c : categorySelector.toSortedNames(p2c.getValue())) out.print(c + "\t");
			out.println();
			pl.lightUpdate();
		}
		pl.done();
		out.close();

	}




}
