package it.unimi.di.wikipedia.parsing;

import it.unimi.di.big.mg4j.document.Document;
import it.unimi.di.big.mg4j.document.DocumentIterator;
import it.unimi.di.big.mg4j.document.DocumentSequence;
import it.unimi.di.big.mg4j.tool.Scan;
import it.unimi.di.big.mg4j.tool.Scan.VirtualDocumentFragment;
import it.unimi.di.big.mg4j.tool.VirtualDocumentResolver;
import it.unimi.dsi.fastutil.longs.LongAVLTreeSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import it.unimi.dsi.fastutil.io.BinIO;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectList;
import it.unimi.dsi.big.webgraph.ImmutableGraph;
import it.unimi.dsi.big.webgraph.ImmutableSequentialGraph;
import it.unimi.dsi.big.webgraph.NodeIterator;

import java.io.IOException;
import java.util.NoSuchElementException;

/** Exposes a document sequence as a (sequentially accessible) immutable graph, according to some
*  virtual field provided by the documents in the sequence. A suitable {@link VirtualDocumentResolver}
*  is used to associate node numbers to each fragment.
*  
*  <p>More precisely, the graph will have as many nodes as there are documents in the sequence, the
*  <var>k</var>-th document (starting from 0) representing node number <var>k</var>.
*  The successors of a document are obtained by extracting the virtual field from the
*  document, turning each {@linkplain it.unimi.di.mg4j.tool.Scan.VirtualDocumentFragment document specifier}
*  into a document number (using the given {@linkplain VirtualDocumentResolver resolver},
*  and discarding unresolved URLs).
*/
public class DocumentSequenceImmutableGraph extends ImmutableSequentialGraph {
	
	/** The underlying sequence. */
	private DocumentSequence sequence;
	/** The number of the virtual field to be used. */
	private int virtualField;
	/** The resolver to be used. */
	private VirtualDocumentResolver resolver;
	
	/** Creates an immutable graph from a sequence.
	 * 
	 * @param sequence the sequence whence the immutable graph should be created.
	 * @param virtualField the number of the virtual field to be used to get the successors from.
	 * @param resolver the resolver to be used to map document specs to node numbers.
	 */
	public DocumentSequenceImmutableGraph( final DocumentSequence sequence, final int virtualField, final VirtualDocumentResolver resolver ) {
		this.sequence = sequence;
		this.virtualField = virtualField;
		this.resolver = resolver;
	}

	/** Creates a new immutable graph with the specified arguments.
	 * 
	 * @param arg a 3-element array: the first is the basename of a {@link DocumentSequence}, the second is an integer specifying the virtual
	 * field number, the third is the basename of a {@link VirtualDocumentResolver}.
	 */
	public DocumentSequenceImmutableGraph( final String... arg ) throws IOException, ClassNotFoundException {
		this( (DocumentSequence)BinIO.loadObject( arg[ 0 ] ), Integer.parseInt( arg[ 1 ] ), (VirtualDocumentResolver)BinIO.loadObject( arg[ 2 ] ) );
	}
	
	@Override
	public ImmutableGraph copy() {
		throw new UnsupportedOperationException();
	}

	@Override
	public long numNodes() {
		if ( resolver.numberOfDocuments() > Integer.MAX_VALUE ) throw new IllegalArgumentException();
		return resolver.numberOfDocuments();
	}

	@Override
	public boolean randomAccess() {
		return false;
	}
	
	public NodeIterator nodeIterator() {
		try {
			final DocumentIterator documentIterator = sequence.iterator();
			return new NodeIterator() {
				Document cachedDocument = documentIterator.nextDocument();
				int cachedDocumentNumber = 0;
				long[] cachedSuccessors;
				LongSortedSet succ = new LongAVLTreeSet();

				public boolean hasNext() {
					return cachedDocument != null;
				}
				
				@SuppressWarnings("unchecked")
				public long nextLong() {
					if ( !hasNext() ) throw new NoSuchElementException();
					ObjectList<Scan.VirtualDocumentFragment> vdf;
					try {
						vdf = (ObjectList<VirtualDocumentFragment>)cachedDocument.content( virtualField );
					}
					catch ( IOException exc1 ) {
						throw new RuntimeException( exc1 );
					}
					succ.clear();
					resolver.context( cachedDocument );
					ObjectIterator<VirtualDocumentFragment> it = vdf.iterator();
					while ( it.hasNext() ) {
						long successor = resolver.resolve( it.next().documentSpecifier() );
						if ( successor >= 0 ) succ.add( successor );
					}
					cachedSuccessors = succ.toLongArray();
					// Get ready for the next request
					try {
						cachedDocument.close();
						cachedDocument = documentIterator.nextDocument();
					}
					catch ( IOException e ) {
						throw new RuntimeException( e );
					}
					return cachedDocumentNumber++;
				}

				public long outdegree() {
					return cachedSuccessors.length;
				}
				
				
				public long[][] successorBigArray() {
					return new long[][] {cachedSuccessors};
				}
				
			};
		}
		catch ( IOException e ) {
			throw new RuntimeException( e );
		}
		
	}

}
