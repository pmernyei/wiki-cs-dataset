package it.unimi.di.wikipedia.parsing;


/*	
 * Modified version of:
 * 	 
 * MG4J: Managing Gigabytes for Java (big)
 *
 * Copyright (C) 2013 Sebastiano Vigna 
 *
 *  This library is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by the Free
 *  Software Foundation; either version 3 of the License, or (at your option)
 *  any later version.
 *
 *  This library is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 *  for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 */

import info.bliki.wiki.filter.Encoder;
import info.bliki.wiki.filter.HTMLConverter;
import info.bliki.wiki.filter.PlainTextConverter;
import info.bliki.wiki.model.WikiModel;
import it.unimi.dsi.big.util.ShiftAddXorSignedStringMap;
import it.unimi.dsi.big.util.StringMap;
import it.unimi.di.big.mg4j.document.AbstractDocument;
import it.unimi.di.big.mg4j.document.AbstractDocumentFactory;
import it.unimi.di.big.mg4j.document.AbstractDocumentIterator;
import it.unimi.di.big.mg4j.document.AbstractDocumentSequence;
import it.unimi.di.big.mg4j.document.CompositeDocumentFactory;
import it.unimi.di.big.mg4j.document.Document;
import it.unimi.di.big.mg4j.document.DocumentFactory;
import it.unimi.di.big.mg4j.document.DocumentIterator;
import it.unimi.di.big.mg4j.document.DocumentSequence;
import it.unimi.di.big.mg4j.document.HtmlDocumentFactory;
import it.unimi.di.big.mg4j.document.PropertyBasedDocumentFactory;
import it.unimi.di.big.mg4j.document.WikipediaDocumentCollection;
import it.unimi.di.big.mg4j.tool.URLMPHVirtualDocumentResolver;
import it.unimi.di.big.mg4j.tool.VirtualDocumentResolver;
import it.unimi.di.big.mg4j.util.parser.callback.AnchorExtractor;
import it.unimi.di.big.mg4j.util.parser.callback.AnchorExtractor.Anchor;
import it.unimi.dsi.bits.TransformationStrategies;
import it.unimi.dsi.bits.TransformationStrategy;
import it.unimi.dsi.fastutil.io.BinIO;
import it.unimi.dsi.fastutil.io.FastBufferedInputStream;
import it.unimi.dsi.fastutil.objects.AbstractObject2LongFunction;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2LongFunction;
import it.unimi.dsi.fastutil.objects.Object2LongLinkedOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ObjectBigList;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import it.unimi.dsi.io.FastBufferedReader;
import it.unimi.dsi.io.FileLinesCollection;
import it.unimi.dsi.io.WordReader;
import it.unimi.dsi.lang.MutableString;
import it.unimi.dsi.lang.ObjectParser;
import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.sux4j.mph.MWHCFunction;
import it.unimi.dsi.util.TextPattern;
import it.unimi.dsi.webgraph.ImmutableGraph;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ArrayBlockingQueue;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import javax.xml.XMLConstants;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.html.HtmlEscapers;
import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPException;
import com.martiansoftware.jsap.JSAPResult;
import com.martiansoftware.jsap.Parameter;
import com.martiansoftware.jsap.SimpleJSAP;
import com.martiansoftware.jsap.Switch;
import com.martiansoftware.jsap.UnflaggedOption;

/** A class exhibiting a standard Wikipedia XML dump as a {@link DocumentSequence}. 
 * 
 * <P><strong>Warning</strong>: this class has no connection whatsoever with
 * {@link WikipediaDocumentCollection}.
 * 
 * <p>The purpose of this class is making the indexing of Wikipedia and of its entity
 * graph starting from a pristine Wikipedia XML dump reasonably easy. There are a few
 * steps involved, mainly due to the necessity of working out redirects, but the whole
 * procedure can be carried out with very little resources. The class uses the
 * {@link WikiModel#toHtml(String, Appendable, String, String)} method to convert
 * the Wikipedia format into HTML, and then passes the result to a standard {@link HtmlDocumentFactory}
 * (suggestion on alternative conversion methods are welcome).
 * A few additional fields are handled by {@link WikipediaHeaderFactory}.
 * 
 * <p>Note that no properties are passed to the underlying {@link HtmlDocumentFactory}: if you want
 * to set the anchor properties (see {@link HtmlDocumentFactory.MetadataKeys}), you need to use
 * {@linkplain #NamespacedWikipediaDocumentSequence(String, boolean, String, boolean, boolean, int, int, int) a quite humongous constructor}.
 * 
 * <h3>How to index Wikipedia</h3>
 * 
 * <p>As a first step, download the Wikipedia XML dump (it's the &ldquo;pages-articles&rdquo; file;
 * it should start with a <samp>mediawiki</samp> opening tag). This class can process the
 * file in its compressed form, but we suggest to uncompress it using <samp>bunzip2</samp>,
 * as processing is an order of magnitude faster. (Note that the following process will exclude namespaced
 * pages such as <samp>Template:<var>something</var></samp>; if you want to include them, you must 
 * use a different {@linkplain #NamespacedWikipediaDocumentSequence(String, boolean, String, boolean, boolean) constructor}.)
 * 
 * 
 * <p>The first step is extracting metadata (in particular, the URLs that are necessary to
 * index correctly the anchor text). We do not suggest specific Java options, but try to use
 * as much memory as you can.
 * <pre>
 * java it.unimi.di.mg4j.tool.ScanMetadata \
 *   -o "it.unimi.di.mg4j.document.WikipediaDocumentSequence(enwiki-latest-pages-articles.xml,false,http://en.wikipedia.org/wiki/,false)" \
 *   -u enwiki.uris -t enwiki.titles
 * </pre>
 * 
 * <p>Note that we used the {@link ObjectParser}-based constructor of this class, which makes it possible to create
 * a {@link NamespacedWikipediaDocumentSequence} instance parsing a textual specification (see the 
 * {@linkplain #NamespacedWikipediaDocumentSequence(String, boolean, String, boolean) constructor}
 * documentation for details about the parameters).
 * 
 * <p>The second step consists in building a first {@link VirtualDocumentResolver} which, however, 
 * does not comprise redirect information:
 * <pre>
 * java it.unimi.di.mg4j.tool.URLMPHVirtualDocumentResolver -o enwiki.uris enwiki.vdr
 * </pre>
 * 
 * <p>Now we need to use the <i>ad hoc</i> main method of this class to rescan the collection, gather the redirect
 * information and merge it with our current resolver:
 * <pre>
 * java it.unimi.di.mg4j.document.WikipediaDocumentSequence \
 *   enwiki-latest-pages-articles.xml http://en.wikipedia.org/wiki/ enwiki.uris enwiki.vdr enwikired.vdr
 * </pre>
 * 
 * <p>During this phase a quite large number of warnings about <em>failed redirects</em> might appear. This is normal,
 * in particular if you do not index template pages. If you suspect an actual bug, try first to index template pages,
 * too. Failed redirects should be in the order of few thousands, and all due to internal inconsistencies of
 * the dump: to check that this is the case, check whether the target of a failed redirect appears as a page
 * title (it shouldn't).
 *   
 * <p>We have now all information required to build a complete index (we use the Porter2 stemmer in this example):
 * <pre>
 * java it.unimi.di.mg4j.tool.IndexBuilder \
 *   -o "it.unimi.di.mg4j.document.WikipediaDocumentSequence(enwiki.xml,false,http://en.wikipedia.org/wiki/,true)" \ 
 *   --all-fields -v enwiki.vdr -t EnglishStemmer enwiki
 * </pre>
 * 
 * <p>Finally, we can build the entity graph using a bridge class that exposes any {@link DocumentSequence} with a virtual
 * field as an {@link ImmutableGraph} of the <a href="http://webgraph.di.unimi.it/">WebGraph framework</a> (the nodes will be in one-to-one correspondence with the documents
 * returned by the index):
 * <pre>
 * java it.unimi.dsi.big.webgraph.BVGraph \
 *   -s "it.unimi.di.mg4j.util.DocumentSequenceImmutableSequentialGraph(\"it.unimi.di.mg4j.document.WikipediaDocumentSequence(enwiki.xml,false,http://en.wikipedia.org/wiki/,true)\",anchor,enwikired.vdr)" \ 
 *   enwiki
 * </pre>
 * 
 * <h2>Additional fields</h2>
 * 
 * <p>The additional fields generated by this class (some of which are a bit hacky) are:
 *
 * <dl>
 * <dt><samp>title</samp>
 * <dd>the title of the Wikipedia page;
 * <dt><samp>id</samp>
 * <dd>a payload index containing the Wikipedia identifier of the page;
 * <dt><samp>lastedit</samp>
 * <dd>a payload index containing the last edit of the page;
 * <dt><samp>category</samp>
 * <dd>a field containing the categories of the page, separated by an artificial marker <samp>OXOXO</samp> (so when you look for a category as a phrase you
 * don't get false cross-category positives);
 * <dt><samp>firstpar</samp>
 * <dd>a heuristically generated first paragraph of the page, useful for identification beyond the title;
 * <dt><samp>redirects</samp>
 * <dd>a virtual field treating the link of the page with its title and any redirect link to the page as an anchor: in practice, the
 * field contains all names under which the page is known in Wikipedia. 
 * </dl>
 * 
 * <p>Note that for each link in a disambiguation page this class will generate a fake link with the same target, but
 * the title of the disambiguation page as text. This is in the same spirit of the <samp>redirects</samp> field&mdash;we enrich
 * the HTML <samp>anchor</samp> field with useful information without altering the generated graph.
 */

public class NamespacedWikipediaDocumentSequence extends AbstractDocumentSequence implements Serializable {
	private static final Logger LOGGER = LoggerFactory.getLogger( NamespacedWikipediaDocumentSequence.class );
	private static final long serialVersionUID = 1L;

	private static final TextPattern CATEGORY_START = new TextPattern( "[[Category:" );
	private static final TextPattern BRACKETS_CLOSED = new TextPattern( "]]" );
	private static final TextPattern BRACES_CLOSED = new TextPattern( "}}" );
	private static final TextPattern DISAMBIGUATION = new TextPattern( "{{disambiguation" );
	private static final TextPattern BRACKETS_OPEN = new TextPattern( "[[" );
	private static final char[] END_OF_DISAMBIGUATION_LINK = new char[] { '|', ']' };
	
	/** A marker used to denote end of input. */
	private static final DocumentAndFactory END = new DocumentAndFactory( null,  null );
	/** The prototype {@link CompositeDocumentFactory} used to parse Wikipedia pages. */
	private final DocumentFactory factory;
	/** Whether the input is compressed with <samp>bzip2</samp>. */
	private final boolean bzipped;
	/** Whether to parse text (e.g., we do not parse text when computing titles/URIs). */
	private final boolean parseText;
	/** Whether to keep in the index namespace pages. */
	private final boolean keepNamespaced;
	/** The Wikipedia XML dump. */
	private final String wikipediaXmlDump;
	/** The base URL for pages (e.g., <samp>http://en.wikipedi.org/wiki/</samp>). */
	private final String baseURL;
	/** {@link #baseURL} concatenated with <samp>${title}</samp>. */
	private final String linkBaseURL;
	/** {@link #baseURL} concatenated with <samp>${image}</samp>. */
	private final String imageBaseURL;
	/** The set of namespaces specified in {@link #wikipediaXmlDump}. */
	private ImmutableSet<MutableString> nameSpaces;
	/** This list (whose access must be synchronized) accumulates virtual text (anchors) generated by redirects.
	 * It is filled when meeting redirect pages, and it is emptied at the first non-redirect page (the page in which the list
	 * is emptied is immaterial). Note that because of this setup, if there are some redirect 
	 * pages that are not followed by any indexed page the anchors of those redirects won't be processed at all. 
	 * If this is a problem, just add a fake empty page at the end. */
	private final ObjectArrayList<Anchor> redirectAnchors = new ObjectArrayList<Anchor>();
	
	public static enum MetadataKeys {
		ID,
		LASTEDIT,
		CATEGORY,
		FIRSTPAR,
		/** This key is used internally by {@link WikipediaHeaderFactory} and is associated with the list of redirect anchors. */
		REDIRECT 
	};

	/** A factory responsible for special Wikipedia fields (see the {@linkplain NamespacedWikipediaDocumentSequence class documentation}). It
	 * will be {@linkplain CompositeDocumentFactory composed} with an {@link HtmlDocumentFactory}. */
	public static final class WikipediaHeaderFactory extends AbstractDocumentFactory {
		private static final long serialVersionUID = 1L;
		private static final Object2IntOpenHashMap<String> FIELD_2_INDEX = new Object2IntOpenHashMap<String>( new String[] { "title", "id", "lastedit", "category", "firstpar", "redirect" }, new int[] { 0, 1, 2, 3, 4, 5 } );
		static {
			FIELD_2_INDEX.defaultReturnValue( -1 );
		}

		private final WordReader wordReader = new FastBufferedReader();
		
		@Override
		public int numberOfFields() {
			return 6;
		}

		@Override
		public String fieldName( int field ) {
			switch( field ) {
			case 0: return "title";
			case 1: return "id";
			case 2: return "lastedit";
			case 3: return "category";
			case 4: return "firstpar";
			case 5: return "redirect";
			default: throw new IllegalArgumentException();
			}
		}

		@Override
		public int fieldIndex( String fieldName ) {
			return FIELD_2_INDEX.getInt( fieldName );
		}

		@Override
		public FieldType fieldType( int field ) {
			switch( field ) {
			case 0: return FieldType.TEXT;
			case 1: return FieldType.INT;
			case 2: return FieldType.DATE;
			case 3: return FieldType.TEXT;
			case 4: return FieldType.TEXT;
			case 5: return FieldType.VIRTUAL;
			default: throw new IllegalArgumentException();
			}
		}

		@Override
		public Document getDocument( final InputStream unusedRawContent, final Reference2ObjectMap<Enum<?>, Object> metadata ) throws IOException {
			return new AbstractDocument() {
				
				@Override
				public WordReader wordReader( int field ) {
					return wordReader; // Fixed, for the time being.
				}
				
				@Override
				public CharSequence uri() {
					return (CharSequence)metadata.get( PropertyBasedDocumentFactory.MetadataKeys.URI );
				}
				
				@Override
				public CharSequence title() {
					return (CharSequence)metadata.get( PropertyBasedDocumentFactory.MetadataKeys.TITLE );
				}
				
				@Override
				public Object content( final int field ) throws IOException {
					switch( field ) {
					case 0: return new FastBufferedReader( (MutableString)metadata.get( PropertyBasedDocumentFactory.MetadataKeys.TITLE ) );
					case 1: return metadata.get( MetadataKeys.ID );
					case 2: return metadata.get( MetadataKeys.LASTEDIT );
					case 3: return new FastBufferedReader( (MutableString)metadata.get( MetadataKeys.CATEGORY ) );
					case 4: return new FastBufferedReader( (MutableString)metadata.get( MetadataKeys.FIRSTPAR ) );
					case 5: 
						@SuppressWarnings("unchecked")
						final ObjectArrayList<Anchor> redirectAnchors = (ObjectArrayList<Anchor>)metadata.get( MetadataKeys.REDIRECT );
						ImmutableList<Anchor> result;
						
						synchronized( redirectAnchors ) {
							redirectAnchors.add( new Anchor( (MutableString)metadata.get( PropertyBasedDocumentFactory.MetadataKeys.URI ), (MutableString)metadata.get( PropertyBasedDocumentFactory.MetadataKeys.TITLE ) ) );
							result = ImmutableList.copyOf( redirectAnchors );
							redirectAnchors.clear();
						}
						// System.err.println( "Adding " + result );
						return result;
					default: throw new IllegalArgumentException();
					}
				}
			};
		}

		@Override
		public DocumentFactory copy() {
			return new WikipediaHeaderFactory();
		}
		
	}

	/** Builds a new Wikipedia document sequence that discards namespaced pages.
	 * 
	 * @param file the file containing the Wikipedia dump.
	 * @param bzipped whether {@code file} is compressed with <samp>bzip2</samp>.
	 * @param baseURL a base URL for links (e.g., for the English Wikipedia, <samp>http://en.wikipedia.org/wiki/</samp>);
	 * note that if it is nonempty this string <strong>must</strong> terminate with a slash.
	 * @param parseText whether to parse the text (this parameter is only set to false during metadata-scanning
	 * phases to speed up the scanning process).
	 */
	public NamespacedWikipediaDocumentSequence( final String file, final boolean bzipped, final String baseURL, final boolean parseText) {
		this( file, bzipped, baseURL, parseText, false );
	}
	
	/** Builds a new Wikipedia document sequence using default anchor settings.
	 * 
	 * @param file the file containing the Wikipedia dump.
	 * @param bzipped whether {@code file} is compressed with <samp>bzip2</samp>.
	 * @param baseURL a base URL for links (e.g., for the English Wikipedia, <samp>http://en.wikipedia.org/wiki/</samp>);
	 * note that if it is nonempty this string <strong>must</strong> terminate with a slash.
	 * @param parseText whether to parse the text (this parameter is only set to false during metadata-scanning
	 * phases to speed up the scanning process).
	 * @param keepNamespaced whether to keep namespaced pages (e.g., <samp>Template:<var>something</var></samp> pages).
	 */
	public NamespacedWikipediaDocumentSequence( final String file, final boolean bzipped, final String baseURL, final boolean parseText, final boolean keepNamespaced) {
		this( file, bzipped, baseURL, parseText, keepNamespaced, 8, 8, 8);
	}

	/** Builds a new Wikipedia document sequence.
	 * 
	 * @param file the file containing the Wikipedia dump.
	 * @param bzipped whether {@code file} is compressed with <samp>bzip2</samp>.
	 * @param baseURL a base URL for links (e.g., for the English Wikipedia, <samp>http://en.wikipedia.org/wiki/</samp>);
	 * note that if it is nonempty this string <strong>must</strong> terminate with a slash.
	 * @param parseText whether to parse the text (this parameter is only set to false during metadata-scanning
	 * phases to speed up the scanning process).
	 * @param keepNamespaced whether to keep namespaced pages (e.g., <samp>Template:<var>something</var></samp> pages).
	 * @param maxPreAnchor maximum number of character before an anchor.
	 * @param maxAnchor maximum number of character in an anchor.
	 * @param maxPostAnchor maximum number of characters after an anchor.
	 */
	public NamespacedWikipediaDocumentSequence( final String file, final boolean bzipped, final String baseURL, final boolean parseText, final boolean keepNamespaced, final int maxPreAnchor, final int maxAnchor, final int maxPostAnchor) {
		this.wikipediaXmlDump = file;
		this.bzipped = bzipped;
		this.baseURL = baseURL;
		this.parseText = parseText;
		this.keepNamespaced = keepNamespaced;
		Reference2ObjectOpenHashMap<Enum<?>, Object> metadata = new Reference2ObjectOpenHashMap<Enum<?>, Object>(
				new Enum[] { HtmlDocumentFactory.MetadataKeys.MAXPREANCHOR, HtmlDocumentFactory.MetadataKeys.MAXANCHOR, HtmlDocumentFactory.MetadataKeys.MAXPOSTANCHOR },
				new Integer[] { Integer.valueOf( maxPreAnchor ), Integer.valueOf( maxAnchor ), Integer.valueOf( maxPostAnchor ) }
				);
		DocumentFactory htmlDocumentFactory = new HtmlDocumentFactory(metadata);
		this.factory = CompositeDocumentFactory.getFactory( new DocumentFactory[] { new WikipediaHeaderFactory(), htmlDocumentFactory }, new String[] { "title", "id", "lastedit", "category", "firstpar", "redirect", "text", "dummy", "anchor" } );
		linkBaseURL = baseURL + "${title}";
		imageBaseURL = baseURL + "${image}";
	}

	/** A string-based constructor to be used with an {@link ObjectParser}.
	 *
	 * @see #NamespacedWikipediaDocumentSequence(String, boolean, String, boolean)
	 */
	public NamespacedWikipediaDocumentSequence( final String file, final String bzipped, final String baseURL, final String parseText ) {
		this( file, Boolean.parseBoolean( bzipped ), baseURL, Boolean.parseBoolean( parseText ) );
	}

	/** A string-based constructor to be used with an {@link ObjectParser}.
	 *
	 * @see #NamespacedWikipediaDocumentSequence(String, boolean, String, boolean, boolean)
	 */
	public NamespacedWikipediaDocumentSequence( final String file, final String bzipped, final String baseURL, final String parseText, final String keepNamespaced ) {
		this( file, Boolean.parseBoolean( bzipped ), baseURL, Boolean.parseBoolean( parseText ), Boolean.parseBoolean( keepNamespaced ) );
	}

	/** A string-based constructor to be used with an {@link ObjectParser}.
	 *
	 * @see #NamespacedWikipediaDocumentSequence(String, boolean, String, boolean, boolean, int, int, int)
	 */
	public NamespacedWikipediaDocumentSequence( final String file, final String bzipped, final String baseURL, final String parseText, final String keepNamespaced, final String maxBeforeAnchor, final String maxAnchor, final String maxPostAnchor ) {
		this( file, Boolean.parseBoolean( bzipped ), baseURL, Boolean.parseBoolean( parseText ), Boolean.parseBoolean( keepNamespaced ), Integer.parseInt( maxBeforeAnchor ), Integer.parseInt(  maxAnchor ), Integer.parseInt( maxPostAnchor ) );
	}
	
	private static final class DocumentAndFactory {
		public final Document document;
		public final DocumentFactory factory;

		public DocumentAndFactory( final Document document, final DocumentFactory documentFactory ) {
			this.document = document;
			this.factory = documentFactory;
		}
	}

	public boolean isATrueNamespace(final String stringBeforeColumn) {
		return nameSpaces.contains( stringBeforeColumn.toLowerCase() );
	}
	
	public boolean isATrueNamespace(final MutableString stringBeforeColumn) {
		return nameSpaces.contains( stringBeforeColumn.toLowerCase() );
	}
	
	@Override
	public DocumentIterator iterator() throws IOException {
		final SAXParserFactory saxParserFactory = SAXParserFactory.newInstance();
		saxParserFactory.setNamespaceAware( true );
		try {
			saxParserFactory.setFeature( XMLConstants.FEATURE_SECURE_PROCESSING, false );
		}
		catch ( Exception e ) {
			throw new RuntimeException("Failed to disable secure processing feature", e);
		}
		final MutableString nameSpaceAccumulator = new MutableString();
		final ObjectOpenHashSet<MutableString> nameSpacesAccumulator = new ObjectOpenHashSet<MutableString>();
		final ArrayBlockingQueue<DocumentFactory> freeFactories = new ArrayBlockingQueue<DocumentFactory>( 16 );
		for( int i = freeFactories.remainingCapacity(); i-- != 0; ) freeFactories.add( this.factory.copy() );
		final ArrayBlockingQueue<DocumentAndFactory> readyDocumentsAndFactories = new ArrayBlockingQueue<DocumentAndFactory>( freeFactories.size() );
		
	    final SAXParser parser;
		try {
			parser = saxParserFactory.newSAXParser();
		}
		catch ( Exception e ) {
			throw new RuntimeException( e.getMessage(), e );
		}
	    final DefaultHandler handler = new DefaultHandler() {
			private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
			private boolean inText;
			private boolean inTitle;
			private boolean inId;
			private boolean inTimestamp;
			private boolean inNamespaceDef;
			private boolean redirect;
			private MutableString text = new MutableString();
			private MutableString title = new MutableString();
			private MutableString id = new MutableString();
			private MutableString timestamp = new MutableString();
			private final Reference2ObjectMap<Enum<?>, Object> metadata = new Reference2ObjectOpenHashMap<Enum<?>, Object>();
			{
				metadata.put( PropertyBasedDocumentFactory.MetadataKeys.ENCODING, "UTF-8" );
				metadata.put( MetadataKeys.REDIRECT, redirectAnchors );
			}

			@Override
			public void startElement( String uri, String localName, String qName, Attributes attributes ) throws SAXException {
				if ( "page".equals( localName ) ) {
					redirect = inText = inTitle = inId = inTimestamp = false;
					text.length( 0 );
					title.length( 0 );
					id.length( 0 );
					timestamp.length( 0 );
				}
				else if ( "text".equals( localName ) ) inText = true;
				else if ( "title".equals( localName ) && title.length() == 0 ) inTitle = true; // We catch only the first id/title elements.
				else if ( "id".equals( localName ) && id.length() ==0  ) inId = true;
				else if ( "timestamp".equals( localName ) && timestamp.length() ==0  ) inTimestamp = true;
				else if ( "redirect".equals( localName ) ) {
					redirect = true;
					if ( attributes.getValue( "title" ) != null )
						// Accumulate the title of the page as virtual text of the redirect page.
						synchronized ( redirectAnchors ) {
							final String link = Encoder.encodeTitleToUrl( attributes.getValue( "title" ), true );
							redirectAnchors.add( new AnchorExtractor.Anchor( new MutableString( baseURL.length() + link.length() ).append( baseURL ).append( link ), title.copy() ) );
						}
				}
				else if ( "namespace".equals( localName ) ) {
					// Found a new namespace
					inNamespaceDef = true;
					nameSpaceAccumulator.length( 0 );
				}
			}

			@Override
			public void endElement( String uri, String localName, String qName ) throws SAXException {
				if ( "namespace".equals( localName ) ) { // Collecting a namespace
					if ( nameSpaceAccumulator.length() != 0 ) nameSpacesAccumulator.add( nameSpaceAccumulator.copy().toLowerCase() );
					return;
				}

				if ( "namespaces".equals( localName ) ) { // All namespaces collected
					nameSpaces = ImmutableSet.copyOf( nameSpacesAccumulator );
					return;
				}

				if ( ! redirect ) {
					if ( "title".equals( localName ) ) {
						// Set basic metadata for the page
						metadata.put( PropertyBasedDocumentFactory.MetadataKeys.TITLE, title.copy() );
						String link = Encoder.encodeTitleToUrl( title.toString(), true );
						metadata.put( PropertyBasedDocumentFactory.MetadataKeys.URI, new MutableString( baseURL.length() + link.length() ).append( baseURL ).append( link ) );
						inTitle = false;
					}
					else if ( "id".equals( localName ) ) {
						metadata.put( MetadataKeys.ID, Long.valueOf( id.toString() ) );
						inId = false;
					}
					else if ( "timestamp".equals( localName ) ) {
						try {
							metadata.put( MetadataKeys.LASTEDIT, dateFormat.parse( timestamp.toString() ) );
						}
						catch ( ParseException e ) {
							throw new RuntimeException( e.getMessage(), e );
						}
						inTimestamp = false;
					}
					else if ( "text".equals( localName ) ) {
						inText = false;
						if ( ! keepNamespaced )  {
							// Namespaces are case-insensitive and language-dependent
							final int pos = title.indexOf( ':' );
							if ( pos != -1 && isATrueNamespace(title.substring( 0, pos )) ) return;
						}
						try {
							final MutableString html = new MutableString();
							DocumentFactory freeFactory;
							try {
								freeFactory = freeFactories.take();
							}
							catch ( InterruptedException e ) {
								throw new RuntimeException( e.getMessage(), e );
							}
							if ( parseText ) {
								if ( DISAMBIGUATION.search( text ) != -1 ) { // It's a disambiguation page.
									/* Roi's hack: duplicate links using the page title, so the generic name will end up as anchor text. */
									final MutableString newLinks = new MutableString();
									for( int start = 0, end; ( start = BRACKETS_OPEN.search( text, start ) ) != -1; start = end ) {
										end = start;
										final int endOfLink = text.indexOfAnyOf( END_OF_DISAMBIGUATION_LINK, start );
										// Note that we don't escape title because we are working at the Wikipedia raw text level.
										if ( endOfLink != -1 ) {
											newLinks.append( text.array(), start, endOfLink - start ).append( '|' ).append( title ).append( "]]\n" );
											end = endOfLink;
										}
										end++;
									}
									
									text.append( newLinks );
								}
								// We separate categories by OXOXO, so we don't get overflowing phrases.
								final MutableString category = new MutableString();
								for( int start = 0, end; ( start = CATEGORY_START.search( text, start ) ) != -1; start = end ) {
									end = BRACKETS_CLOSED.search( text, start += CATEGORY_START.length() );
									if ( end != -1 ) category.append( text.subSequence( start,  end ) ).append( " OXOXO " );
									else break;
								}
								metadata.put( MetadataKeys.CATEGORY, category );
								
								// Heuristics to get the first paragraph
								metadata.put( MetadataKeys.FIRSTPAR, new MutableString() );
								String plainText = new WikiModel( imageBaseURL, linkBaseURL ).render( new PlainTextConverter( true ), text.toString() );
								for( int start = 0; start < plainText.length(); start++ ) {
									//System.err.println("Examining " + plainText.charAt( start )  );
									if ( Character.isWhitespace( plainText.charAt( start ) ) ) continue;
									if ( plainText.charAt( start ) == '{' ) {
										//System.err.print( "Braces " + start + " text: \"" + plainText.subSequence( start, start + 10 )  + "\" -> " );
										start = BRACES_CLOSED.search( plainText, start );
										//System.err.println( start + " text: \"" + plainText.subSequence( start, start + 10 ) + "\"" );
										if ( start == -1 ) break;
										start++;
									}
									else if ( plainText.charAt( start ) == '[' ) {
										start = BRACKETS_CLOSED.search( plainText, start );
										if ( start == -1 ) break;
										start++;
									}
									else {
										final int end = plainText.indexOf( '\n', start );
										if ( end != -1 ) metadata.put( MetadataKeys.FIRSTPAR, new MutableString( plainText.substring( start, end ) ) );//new MutableString( new WikiModel( imageBaseURL, linkBaseURL ).render( new PlainTextConverter( true ), text.substring( start, end ).toString() ) ) );
										break;
									}
								}
								
								try {
									WikiModel wikiModel = new WikiModel( imageBaseURL, linkBaseURL );
									wikiModel.render( new HTMLConverter(), text.toString(), html, false, false );
									final Map<String, String> categories = wikiModel.getCategories();
									// Put back category links in the page (they have been parsed by bliki and to not appear anymore in the HTML rendering)
									for( Entry<String, String> entry: categories.entrySet() ) {
										final String key = entry.getKey();
										final String value = entry.getValue().trim();
										if ( value.length() != 0 ) // There are empty such things
											html.append( "\n<a href=\"" ).append( baseURL ).append( "Category:" ).append( Encoder.encodeTitleToUrl( key, true ) ).append(  "\">" ).append( HtmlEscapers.htmlEscaper().escape( key ) ).append( "</a>\n" );
									}
								}
								catch( Exception e ) {
									LOGGER.error( "Unexpected exception while parsing " + title, e );
								}
							}
							readyDocumentsAndFactories.put( new DocumentAndFactory( freeFactory.getDocument( IOUtils.toInputStream( html, Charsets.UTF_8 ), new Reference2ObjectOpenHashMap<Enum<?>, Object>( metadata ) ), freeFactory ) );
						}
						catch ( InterruptedException e ) {
							throw new RuntimeException( e.getMessage(), e );
						}
						catch ( IOException e ) {
							throw new RuntimeException( e.getMessage(), e );
						}
					}
				}
			}

			@Override
			public void characters( char[] ch, int start, int length ) throws SAXException {
				if ( inText && parseText ) text.append( ch, start, length );
				if ( inTitle ) title.append( ch, start, length );
				if ( inId ) id.append( ch, start, length );
				if ( inTimestamp ) timestamp.append( ch, start, length );
				if ( inNamespaceDef ) {
					nameSpaceAccumulator.append( ch, start, length );
					inNamespaceDef = false; // Dirty, but it works
				}
			}

			@Override
			public void ignorableWhitespace( char[] ch, int start, int length ) throws SAXException {
				if ( inText && parseText ) text.append( ch, start, length );
				if ( inTitle ) title.append( ch, start, length );
			}
	    };

	    final Thread parsingThread = new Thread() {
	    	public void run() {
	    		try {
					InputStream in = new FileInputStream( wikipediaXmlDump );
					if ( bzipped ) in = new BZip2CompressorInputStream( in );
					parser.parse( new InputSource( new InputStreamReader( new FastBufferedInputStream( in ), Charsets.UTF_8  ) ), handler );
					readyDocumentsAndFactories.put( END );
				}
				catch ( Exception e ) {
					throw new RuntimeException( e.getMessage(), e );
				}
	    	}
	    };
	    
	    parsingThread.start();

	    return new AbstractDocumentIterator() {
	    	private DocumentFactory lastFactory;
			@Override
			public Document nextDocument() throws IOException {
				try {
					final DocumentAndFactory documentAndFactory = readyDocumentsAndFactories.take();
					if ( lastFactory != null ) freeFactories.put( lastFactory );
					if ( documentAndFactory == END ) return null;
					lastFactory = documentAndFactory.factory;
					return documentAndFactory.document;
				}
				catch ( InterruptedException e ) {
					throw new RuntimeException( e.getMessage(), e );
				}
			}
		};
	}

	@Override
	public DocumentFactory factory() {
		return factory;
	}

	/** A wrapper around a signed function that remaps entries exceeding a provided threshold using a specified target array. */
	public static final class SignedRedirectedStringMap extends AbstractObject2LongFunction<CharSequence> implements StringMap<CharSequence> {
		private static final long serialVersionUID = 1L;
		/** The number of documents. */
		private final long numberOfDocuments;
		/** A signed function function mapping valid keys to their ordinal position. */
		private Object2LongFunction<CharSequence> signedFunction;
		/** The value to be returned for keys whose ordinal position is greater than {@link #numberOfDocuments}. */
		private final long[] target;

		/** Creates a new signed redirected map.
		 * 
		 * @param numberOfDocuments the threshold after which the {@code target} array will be used to compute the output.
		 * @param signedFunction the base signed function.
		 * @param target an array providing the output for items beyond {@code numberOfDocuments}; it must be
		 * long as the size of {@code signedFunction} minus {@code numberOfDocuments}.
		 */
		public SignedRedirectedStringMap( final long numberOfDocuments, final Object2LongFunction<CharSequence> signedFunction, final long[] target ) {
			this.numberOfDocuments = numberOfDocuments;
			this.signedFunction = signedFunction;
			this.target = target;
		}
		
		@Override
		public long getLong( Object key ) {
			final long index = signedFunction.getLong( key ); 
			if ( index == -1 ) return -1;
			if ( index < numberOfDocuments ) return index;
			return target[ (int)( index - numberOfDocuments ) ];
		}

		@Override
		public boolean containsKey( Object key ) {
			return signedFunction.getLong( key ) != -1;
		}

		public long size64() {
			return numberOfDocuments;
		}

		@Override
		@Deprecated
		public int size() {
			return (int)Math.min( Integer.MAX_VALUE, size64() );
		}

		@Override
		public ObjectBigList<? extends CharSequence> list() {
			return null;
		}
	}
	
	
	public static void main( final String arg[] ) throws ParserConfigurationException, SAXException, IOException, JSAPException, ClassNotFoundException {
		SimpleJSAP jsap = new SimpleJSAP( NamespacedWikipediaDocumentSequence.class.getName(), "Computes the redirects of a Wikipedia dump and integrate them into an existing virtual document resolver for the dump.",
			new Parameter[] {
				new Switch( "bzip2", 'b', "bzip2", "The file is compressed with bzip2" ),
				new Switch( "iso", 'i', "iso", "Use ISO-8859-1 coding internally (i.e., just use the lower eight bits of each character)." ),
				new FlaggedOption( "width", JSAP.INTEGER_PARSER, Integer.toString( Long.SIZE ), JSAP.NOT_REQUIRED, 'w', "width", "The width, in bits, of the signatures used to sign the function from URIs to their rank." ),
				new UnflaggedOption( "file", JSAP.STRING_PARSER, JSAP.REQUIRED, "The file containing the Wikipedia dump." ),
				new UnflaggedOption( "baseURL", JSAP.STRING_PARSER, JSAP.REQUIRED, "The base URL for the collection (e.g., http://en.wikipedia.org/wiki/)." ),
				new UnflaggedOption( "uris", JSAP.STRING_PARSER, JSAP.REQUIRED, "The URIs of the documents in the collection (generated by ScanMetadata)." ),
				new UnflaggedOption( "vdr", JSAP.STRING_PARSER, JSAP.REQUIRED, "The name of a precomputed virtual document resolver for the collection." ),
				new UnflaggedOption( "redvdr", JSAP.STRING_PARSER, JSAP.REQUIRED, "The name of the resulting virtual document resolver." )
		});

		JSAPResult jsapResult = jsap.parse( arg );
		if ( jsap.messagePrinted() ) return;

		final SAXParserFactory saxParserFactory = SAXParserFactory.newInstance();
		saxParserFactory.setNamespaceAware( true );
		final Object2ObjectOpenHashMap<MutableString, String> redirects = new Object2ObjectOpenHashMap<MutableString, String>();
		final String baseURL = jsapResult.getString( "baseURL" );
		final ProgressLogger progressLogger = new ProgressLogger( LOGGER );
		progressLogger.itemsName = "redirects";
		progressLogger.start( "Extracting redirects..." );
		
	    final SAXParser parser = saxParserFactory.newSAXParser();
	    final DefaultHandler handler = new DefaultHandler() {
	    	private boolean inTitle;	    	
	    	private MutableString title = new MutableString();
			
			@Override
			public void startElement( String uri, String localName, String qName, Attributes attributes ) throws SAXException {
				if ( "page".equals( localName ) ) {
					inTitle = false;
					title.length( 0 );
				}
				else if ( "title".equals( localName ) && title.length() == 0 ) inTitle = true; // We catch only the first title element.
				else if ( "redirect".equals( localName ) && attributes.getValue( "title" ) != null ) {
					progressLogger.update();
					redirects.put( title.copy(), attributes.getValue( "title" ) );
				}
			}

			@Override
			public void endElement( String uri, String localName, String qName ) throws SAXException {
				if ( "title".equals( localName ) ) inTitle = false;
			}

			@Override
			public void characters( char[] ch, int start, int length ) throws SAXException {
				if ( inTitle ) title.append( ch, start, length );
			}

			@Override
			public void ignorableWhitespace( char[] ch, int start, int length ) throws SAXException {
				if ( inTitle ) title.append( ch, start, length );
			}
	    };
		
		InputStream in = new FileInputStream( jsapResult.getString( "file" ) );
		if ( jsapResult.userSpecified( "bzip2" ) ) in = new BZip2CompressorInputStream( in );
		parser.parse( new InputSource( new InputStreamReader( new FastBufferedInputStream( in ), Charsets.UTF_8  ) ), handler );
		progressLogger.done();

		final Object2LongLinkedOpenHashMap<MutableString> resolved = new Object2LongLinkedOpenHashMap<MutableString>();
		final VirtualDocumentResolver vdr = (VirtualDocumentResolver)BinIO.loadObject( jsapResult.getString( "vdr" ) );

		progressLogger.expectedUpdates = redirects.size();
		progressLogger.start( "Examining redirects..." );

		for( Map.Entry<MutableString,String> e: redirects.entrySet() ) {
			final MutableString start = new MutableString().append( baseURL ).append( Encoder.encodeTitleToUrl( e.getKey().toString(), true ) );
			final MutableString end = new MutableString().append( baseURL ).append( Encoder.encodeTitleToUrl( e.getValue(), true ) );
			final long s = vdr.resolve( start );
			if ( s == -1 ) {
				final long t = vdr.resolve( end );
				if ( t != -1 ) resolved.put( start.copy(), t );
				else LOGGER.warn( "Failed redirect: " + start + " -> " + end );
			}
			else LOGGER.warn( "URL " + start + " is already known to the virtual document resolver" );			
			
			progressLogger.lightUpdate();
		}
		
		progressLogger.done();
		
		//System.err.println(resolved);
		
		final Iterable<MutableString> allURIs = Iterables.concat( new FileLinesCollection( jsapResult.getString( "uris" ), "UTF-8" ), resolved.keySet() );
		final long numberOfDocuments = vdr.numberOfDocuments();
		
		final TransformationStrategy<CharSequence> transformationStrategy = jsapResult.userSpecified( "iso" ) 
				? TransformationStrategies.iso()
				: TransformationStrategies.utf16();
		
		BinIO.storeObject( 
			new URLMPHVirtualDocumentResolver( 
				new SignedRedirectedStringMap( numberOfDocuments,
					new ShiftAddXorSignedStringMap( allURIs.iterator(), new MWHCFunction.Builder<CharSequence>().keys( allURIs ).transform( transformationStrategy ).build(), jsapResult.getInt( "width" ) ),
						resolved.values().toLongArray() ) ), jsapResult.getString( "redvdr" ) );
	}
}
