package org.colomoto.function.core;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * For clauses of size n, it generates the complete Power Set graph containing
 * the set of all subsets up until a set with n elements. It does not contain
 * the empty set, thus having 2^n - 1 subsets, as well as their inclusion
 * relationships.
 * 
 * @author Pedro T. Monteiro
 * @author José R. Cury
 * @author Claudine Chaouiya
 *
 */
public class PowerSetGraph {
	private int nvars;
	private Map<Clause, Set<Clause>> subsetClauses;
	private Map<Clause, Set<Clause>> supersetClauses;
	private Set<Clause> all;

	public PowerSetGraph(int nvars) {
		this.nvars = nvars;
		this.subsetClauses = new HashMap<Clause, Set<Clause>>();
		this.supersetClauses = new HashMap<Clause, Set<Clause>>();
		this.all = new HashSet<Clause>();
		Clause c = new Clause(nvars);
		this.supersetClauses.put(c, new HashSet<Clause>());
		this.buildGraph(c);
	}

	/**
	 * Recursively computes the powerset graph using a top-down approach. From the
	 * top clause c, it computes its direct dominated sets, keeping this
	 * superset/subset relations. If a subset is not yet known, it calls itself
	 * recursively.
	 * 
	 * @param cSuperset
	 */
	private void buildGraph(Clause cSuperset) {
		if (this.subsetClauses.get(cSuperset) == null)
			this.subsetClauses.put(cSuperset, new HashSet<Clause>());
		// Already explored this node (coming from another path)
		if (this.all.contains(cSuperset))
			return;
		this.all.add(cSuperset);
		// Cannot have a subset (it'd be the emptyset)
		if (cSuperset.order() == 1)
			return;
		for (Clause cSubset : this.computeSubsets(cSuperset)) {
			if (this.supersetClauses.get(cSubset) == null)
				this.supersetClauses.put(cSubset, new HashSet<Clause>());
			// add relation: superset -> [... subset ...]
			Set<Clause> sTmp = this.subsetClauses.get(cSuperset);
			sTmp.add(cSubset);
			this.subsetClauses.put(cSuperset, sTmp);
			// add relation: subset -> [... superset ...]
			sTmp = this.supersetClauses.get(cSubset);
			sTmp.add(cSuperset);
			this.supersetClauses.put(cSubset, sTmp);
			// Calls recursively
			this.buildGraph(cSubset);
		}
	}

	/**
	 * Computes the subsets of a given Clause c. Subsets will be more generic in the
	 * sense that each will have one position less set to true than Clause c. This
	 * yields a small size list of clauses instead of a set, since it's most of the
	 * times < 10 elements.
	 * 
	 * @param c A clause c to compute all possible subsets
	 * @return the list of all possible subsets
	 */
	private List<Clause> computeSubsets(Clause c) {
		List<Clause> lSubsets = new ArrayList<Clause>();
		BitSet bs = c.getSignature();
		for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) {
			BitSet clone = (BitSet) bs.clone();
			clone.clear(i);
			lSubsets.add(new Clause(this.nvars, clone));
		}
		return lSubsets;
	}

	/**
	 * Returns all clauses which are included in clause c.
	 * 
	 * @param c
	 * @return
	 */
	public Set<Clause> getDominatedDirectly(Clause c) {
		return this.subsetClauses.get(c);
	}

	/**
	 * Gets the computed sons of clause c present in the powerset graph.
	 * 
	 * @param c
	 * @return
	 */
	public Set<Clause> getDominantDirectly(Clause c) {
		return this.supersetClauses.get(c);
	}

	private Set<Clause> getDominatedRecursively(Clause c, Set<Clause> sSeen) {
//System.out.println("dominatedRec: c=" + c + " sub:" + this.subsetClauses.get(c));
		for (Clause cSubset : this.subsetClauses.get(c)) {
			if (!sSeen.contains(cSubset)) {
				sSeen.add(cSubset);
				sSeen.addAll(this.getDominatedRecursively(cSubset, sSeen));
			}
		}
		return sSeen;
	}

	private Set<Clause> getDominantRecursively(Clause c, Set<Clause> sSeen) {
		for (Clause cSuperset : this.supersetClauses.get(c)) {
			if (!sSeen.contains(cSuperset)) {
				sSeen.add(cSuperset);
				sSeen.addAll(this.getDominantRecursively(cSuperset, sSeen));
			}
		}
		return sSeen;
	}

	public boolean noSuperset(Set<Clause> sX, Clause cSigma) {
		for (Clause cX : sX) {
			if (cX.dominatesOrEqualTo(cSigma))
				return false;
		}
		return true;
	}

	public boolean noSubset(Set<Clause> sX, Clause cSigma) {
		for (Clause cX : sX) {
			if (cSigma.dominatesOrEqualTo(cX))
				return false;
		}
		return true;
	}

	/**
	 * The meet operator, returns a clause that is both a superset of clause ci and
	 * clause cj, if possible. If no such clause exists, it returns null.
	 * 
	 * @param ci
	 * @param cj
	 * @return
	 */
	public Clause meetSuperset(Clause ci, Clause cj) {
		Set<Clause> sMeet = new HashSet<Clause>(this.getDominantDirectly(ci));
		sMeet.retainAll(this.getDominantDirectly(cj));
		// Has at most 1 element!
		return (sMeet.isEmpty() ? null : sMeet.iterator().next());
	}

	/**
	 * Given a set of clauses, it finds the set of independent clauses from a given
	 * set of clauses it checks the powerset graph to find the set of all clauses
	 * that are independent
	 * 
	 * @param sClauses
	 * @return
	 */
	public Set<Clause> getIndependent(Set<Clause> sClauses) {
//System.out.println("getIndependent: " + sClauses);
		// Get Dependent
		Set<Clause> sDependent = new HashSet<Clause>();
		for (Clause c : sClauses) {
//System.out.println("  c:" + c);
			sDependent.addAll(this.getDominatedRecursively(c, sDependent));
			sDependent.addAll(this.getDominantRecursively(c, sDependent));
		}

		Set<Clause> sDiff = new HashSet<Clause>(this.all);
		// Remove all initial clauses
		sDiff.removeAll(sClauses);
		// Remove all dependent clauses
		sDiff.removeAll(sDependent);

		return sDiff;
	}

	/**
	 * Given a set of clauses, it finds the subset of clauses that are minimal
	 * 
	 * @param sClauses
	 * @return
	 */
	public Set<Clause> getMinimal(Set<Clause> sClauses) {
		List<Clause> lClauses = new ArrayList<Clause>(sClauses);
		Set<Clause> sMinimal = new HashSet<Clause>();
		for (int i = 0; i < lClauses.size(); i++) {
			boolean dominates = false;
			for (int j = 0; j < lClauses.size(); j++) {
				if (i != j && lClauses.get(i).dominatesOrEqualTo(lClauses.get(j))) {
					dominates = true;
					break;
				}
			}
			if (!dominates) {
				sMinimal.add(lClauses.get(i));
			}
		}
		return sMinimal;
	}
	
	/**
	 * Given a set of clauses, it finds the subset of clauses that are maximal
	 * 
	 * @param sClauses
	 * @return
	 */
	public Set<Clause> getMaximal(Set<Clause> sClauses) {
		List<Clause> lClauses = new ArrayList<Clause>(sClauses);
		Set<Clause> sMaximal = new HashSet<Clause>();
		for (int i = 0; i < lClauses.size(); i++) {
			boolean dominated = false;
			for (int j = 0; j < lClauses.size(); j++) {
				if (i != j && lClauses.get(i).dominatedOrEqualTo(lClauses.get(j))) {
					dominated = true;
					break;
				}
			}
			if (!dominated) {
				sMaximal.add(lClauses.get(i));
			}
		}
		return sMaximal;
	}

	public String toString() {
		String s = "";
		for (Clause c : this.all) {
			s += c + "\tSuper: " + this.supersetClauses.get(c) + "\tSub: " + this.subsetClauses.get(c) + "\n";
		}
		return s;
	}
}
