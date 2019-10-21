package it.unimi.di.wikipedia.categories;

import it.unimi.dsi.fastutil.ints.IntArrayFIFOQueue;
import it.unimi.dsi.fastutil.ints.IntArrayPriorityQueue;
import it.unimi.dsi.fastutil.ints.IntPriorityQueue;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.LazyIntIterator;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class HittingDistanceMinimizer {
	public static final Logger LOGGER = LoggerFactory.getLogger(HittingDistanceMinimizer.class);

	final ImmutableGraph transposed;
	final int[] minMilestoneDistance;
	final int[] closestMilestone;
	final IntSet milestones;
	final ObjectSet<Visitor> runningVisitors;
	final IntPriorityQueue milestoneQueue;
	final ProgressLogger pl;

	public HittingDistanceMinimizer(ImmutableGraph transposedGraph, IntSet milestones) {
		this.transposed = transposedGraph;
		this.milestones = milestones;
		minMilestoneDistance = new int[transposedGraph.numNodes()];
		Arrays.fill(minMilestoneDistance, Integer.MAX_VALUE);
		closestMilestone = new int[transposedGraph.numNodes()];
		Arrays.fill(closestMilestone, -1);
		milestoneQueue = new IntArrayPriorityQueue(milestones.toIntArray());
		runningVisitors = new ObjectOpenHashSet<Visitor>();
		pl  = new ProgressLogger(LOGGER, "milestones");
		pl.expectedUpdates = milestones.size();

	}

	private class Visitor extends Thread {

		final int start;
		final int[] dists;
		final ImmutableGraph graph;

		Visitor(final ImmutableGraph graph, int startingNode) {
			this.start = startingNode;
			dists = new int[ graph.numNodes() ];
			this.graph = graph.copy();
		}

		@Override
		public void run() {
			final IntArrayFIFOQueue queue = new IntArrayFIFOQueue();

			Arrays.fill( dists, Integer.MAX_VALUE ); // Initially, all distances are infinity.

			int curr, succ;
			queue.enqueue( start );
			dists[ start ] = 0;

			LazyIntIterator successors;

			while( ! queue.isEmpty() ) {
				curr = queue.dequeueInt();
				successors = graph.successors( curr );
				int d = graph.outdegree( curr );
				while( d-- != 0 ) {
					succ = successors.nextInt();
					if ( dists[ succ ] == Integer.MAX_VALUE  ) {
						dists[ succ ] = dists[ curr ] + 1;
						queue.enqueue( succ );
					}
				}
			}

			startNewThreadAfter(this);
		}

		@Override
		public int hashCode() { return start; }

		@Override
		public boolean equals(Object o) {
			return (((o instanceof Visitor)) &&  ((Visitor) o).start == this.start);
		}
	}

	private synchronized void startNewThreadAfter(Visitor thread) {
		if (thread != null) {
			if (!runningVisitors.remove(thread)) {
				throw new IllegalStateException(
						"Thread " + thread + " signaled completion but was not present.");
			}
			updateClosestMilestonesAfter(thread.start, thread.dists);
			pl.update();
		}

		if (!milestoneQueue.isEmpty()) {
			int milestone = milestoneQueue.dequeueInt();
			Visitor visitor = new Visitor(transposed, milestone);
			runningVisitors.add(visitor);
			visitor.start();
		} else
			if (runningVisitors.isEmpty()) {
				synchronized (this) {
					this.notifyAll();
				}
			}
	}


	private void updateClosestMilestonesAfter(int milestone, int[] distances) {
		final int numNodes = transposed.numNodes();
		for (int node = 0; node < numNodes; node++) {
			if (distances[node] < minMilestoneDistance[node] && node != milestone) {
				minMilestoneDistance[node] = distances[node];
				closestMilestone[node] = milestone;
			}
		}
	}

	public int[] compute() {
		return compute(Runtime.getRuntime().availableProcessors());
	}

	public int[] compute(int nOfThreads) {
		pl.start("Starting a BFS for each milestone (with " + nOfThreads + " parallel threads)...");
		for (int i = 0; i < nOfThreads; i++) {
			startNewThreadAfter(null);
		}
		try {
			synchronized (this) {
				while (!milestoneQueue.isEmpty())
					this.wait();
			}
		} catch (InterruptedException e) { throw new RuntimeException(e); }

		pl.done();

		return closestMilestone;

	}


}
