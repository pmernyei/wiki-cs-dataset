package it.unimi.di.wikipedia.utils;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap.Entry;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.ints.IntSets;
import it.unimi.dsi.fastutil.io.BinIO;
import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.webgraph.BVGraph;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.LazyIntIterator;
import it.unimi.dsi.webgraph.LazyIntIterators;
import it.unimi.dsi.webgraph.ScatteredArcsASCIIGraph;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPException;
import com.martiansoftware.jsap.JSAPResult;
import com.martiansoftware.jsap.Parameter;
import com.martiansoftware.jsap.SimpleJSAP;
import com.martiansoftware.jsap.UnflaggedOption;

public class IntMapGraph extends ImmutableGraph {
	public static Logger LOGGER = LoggerFactory.getLogger(IntMapGraph.class);
	
	public final Int2ObjectMap<IntSet> map;
	private final int numNodes, numArcs;
	
	public IntMapGraph(Int2ObjectMap<IntSet> map) {
		this.map = map;
		if (map.defaultReturnValue() == null || !map.defaultReturnValue().equals(IntSets.EMPTY_SET)) {
			LOGGER.warn("It is necessary to set default return value of the map as the empty set.");
			map.defaultReturnValue(IntSets.EMPTY_SET);
		}
		
		int maxNodeIndex = 0, numArcs = 0;
		for (Entry<IntSet> x : map.int2ObjectEntrySet()) {
			if (x.getIntKey() > maxNodeIndex)
				maxNodeIndex = x.getIntKey();
			for (int succ : x.getValue()) {
				if (succ > maxNodeIndex)
					maxNodeIndex = succ;
				numArcs++;
			}
		}
		
		this.numArcs  = numArcs;
		this.numNodes = maxNodeIndex+1;
	}

	@Override
	public int numNodes() {
		return numNodes;
	}

	@Override
	public boolean randomAccess() {
		return true;
	}

	@Override
	public int outdegree(int x) {
		return map.get(x).size();
	}
	
	@Override
	public long numArcs() { 
		return numArcs;
	}
	
	@Override
	public int[] successorArray( final int x ) { 
		int[] succ = map.get(x).toIntArray();
		IntArrays.quickSort(succ);
		return succ;
	}
	
	@Override
	public LazyIntIterator successors( final int x ) { 
		return LazyIntIterators.wrap( successorArray(x) );
	}


	@Override
	public ImmutableGraph copy() {
		throw new UnsupportedOperationException();
	}
	
	@SuppressWarnings("unchecked")
	public static void main( String args[] ) throws IllegalArgumentException, SecurityException, IOException, JSAPException, ClassNotFoundException  {
		String basename;
		SimpleJSAP jsap = new SimpleJSAP( ScatteredArcsASCIIGraph.class.getName(), "Converts a int2intset fastutil map into a BVGraph.",
				new Parameter[] {
						new UnflaggedOption( "map", JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY, "The serialized Int2ObjectMap<IntSet>" ),	
						new FlaggedOption( "logInterval", JSAP.LONG_PARSER, Long.toString( ProgressLogger.DEFAULT_LOG_INTERVAL ), JSAP.NOT_REQUIRED, 'l', "log-interval", "The minimum time interval between activity logs in milliseconds." ),
						new FlaggedOption( "comp", JSAP.STRING_PARSER, null, JSAP.NOT_REQUIRED, 'c', "comp", "A compression flag (may be specified several times)." ).setAllowMultipleDeclarations( true ),
						new FlaggedOption( "windowSize", JSAP.INTEGER_PARSER, String.valueOf( BVGraph.DEFAULT_WINDOW_SIZE ), JSAP.NOT_REQUIRED, 'w', "window-size", "Reference window size (0 to disable)." ),
						new FlaggedOption( "maxRefCount", JSAP.INTEGER_PARSER, String.valueOf( BVGraph.DEFAULT_MAX_REF_COUNT ), JSAP.NOT_REQUIRED, 'm', "max-ref-count", "Maximum number of backward references (-1 for ∞)." ),
						new FlaggedOption( "minIntervalLength", JSAP.INTEGER_PARSER, String.valueOf( BVGraph.DEFAULT_MIN_INTERVAL_LENGTH ), JSAP.NOT_REQUIRED, 'i', "min-interval-length", "Minimum length of an interval (0 to disable)." ),
						new FlaggedOption( "zetaK", JSAP.INTEGER_PARSER, String.valueOf( BVGraph.DEFAULT_ZETA_K ), JSAP.NOT_REQUIRED, 'k', "zeta-k", "The k parameter for zeta-k codes." ),
						new UnflaggedOption( "basename", JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY, "The basename of the output graph" ),
					}
				);
				
		JSAPResult jsapResult = jsap.parse( args );
		if ( jsap.messagePrinted() ) System.exit( 1 );
		
		basename = jsapResult.getString( "basename" );

		int flags = 0;
		for( String compressionFlag: jsapResult.getStringArray( "comp" ) ) {
			try {
				flags |= BVGraph.class.getField( compressionFlag ).getInt( BVGraph.class );
			}
			catch ( Exception notFound ) {
				throw new JSAPException( "Compression method " + compressionFlag + " unknown." );
			}
		}
		
		final int windowSize = jsapResult.getInt( "windowSize" );
		final int zetaK = jsapResult.getInt( "zetaK" );
		int maxRefCount = jsapResult.getInt( "maxRefCount" );
		if ( maxRefCount == -1 ) maxRefCount = Integer.MAX_VALUE;
		final int minIntervalLength = jsapResult.getInt( "minIntervalLength" );
		
		final ProgressLogger pl = new ProgressLogger( LOGGER, jsapResult.getLong( "logInterval" ), TimeUnit.MILLISECONDS );
		ImmutableGraph graph = new IntMapGraph((Int2ObjectMap<IntSet>) BinIO.loadObject(jsapResult.getString("map")));
		BVGraph.store( graph, basename, windowSize, maxRefCount, minIntervalLength, zetaK, flags, pl );
	}

}
