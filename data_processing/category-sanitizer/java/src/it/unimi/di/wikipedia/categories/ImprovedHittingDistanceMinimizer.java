package it.unimi.di.wikipedia.categories;

import it.unimi.dsi.fastutil.ints.IntArrayFIFOQueue;
import it.unimi.dsi.fastutil.ints.IntArrayPriorityQueue;
import it.unimi.dsi.fastutil.ints.IntPriorityQueue;
import it.unimi.dsi.fastutil.ints.IntSet;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.LazyIntIterator;

import java.lang.IllegalStateException;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Queue;

public class ImprovedHittingDistanceMinimizer {

	final ImmutableGraph graph;
	final boolean[] visited;
	final int[] closestMilestone;
	final IntSet milestones;
	final IntSet badPoints;

	boolean computed = false;

	public ImprovedHittingDistanceMinimizer(ImmutableGraph transposedGraph, IntSet milestones, IntSet badPoints) {
		this.graph = transposedGraph;
		this.milestones = milestones;
		this.badPoints = badPoints;
		visited = new boolean[transposedGraph.numNodes()];
		closestMilestone = new int[transposedGraph.numNodes()];
		Arrays.fill(closestMilestone, -1);
	}
	
	/**
	* Calculates the nearest marked ancestor of each node in the graph using
	* breadth-first search from the marked node set.
	* Uses inputs given to the constructor of the object.
	*/
	public int[] compute() {
		// Sort in ascending order to replicate original behaviour in case of
		// equal distances
		int[] milestoneArray = milestones.toIntArray();
		Arrays.sort(milestoneArray);

		Queue<Integer> bfsQueue = new ArrayDeque<Integer>();
		for (int m : milestoneArray) {
			bfsQueue.add(m);
		}
		while (!bfsQueue.isEmpty()) {
			int curr = bfsQueue.remove();
			LazyIntIterator successors = graph.successors( curr );
			int d = graph.outdegree( curr );
			while( d-- != 0 ) {
				int succ = successors.nextInt();
				if (badPoints.contains(succ)) {
					continue;
				}
				if (!visited[succ]) {
					visited[succ] = true;
					if (milestones.contains(curr)) {
						closestMilestone[succ] = curr;
					} else {
						closestMilestone[succ] = closestMilestone[curr];
					}
					bfsQueue.add(succ);
				}
			}
		}
		return closestMilestone;
	}

}
