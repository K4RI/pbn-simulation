package org.colomoto.function.core;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Hasse Diagram
 * 
 * @author Pedro T. Monteiro
 * @author José R. Cury
 * @author Claudine Chaouiya
 *
 */
public class HasseDiagram {
	// n=2 -> 2 nodes
	// n=3 -> 9 nodes
	// n=4 -> 114 nodes
	// n=5 -> 6.894 nodes
	// n=6 -> 7.785.062 nodes
	// n=7 -> 2.414.627.396.434 nodes
	// n=8 -> 56.130.437.209.370.320.359.966 nodes
	private int nvars;
	private PowerSetGraph powerSet;
	private static Formula fBottom;

	public HasseDiagram(int nvars) {
		this.nvars = nvars;
		this.powerSet = new PowerSetGraph(nvars);
	}

	public int getSize() {
		return this.nvars;
	}

	public Set<Formula> getFormulaAncestors(Formula fInit, boolean degenerated) {
		Set<Formula> sExplored = new HashSet<Formula>();
		Set<Formula> sToExplore = new HashSet<Formula>();
		sToExplore.add(fInit);
		while (!sToExplore.isEmpty()) {
			Formula f = sToExplore.iterator().next();
			sToExplore.remove(f);
			sToExplore.addAll(this.getFormulaParents(f, degenerated));
			sExplored.add(f);
		}
		return sExplored;
	}

	public Set<Formula> getFormulaParents(Formula f, boolean degenerated) {
		Set<Clause> sfClauses = f.getClauses();
		Set<Formula> sFprime = new HashSet<Formula>();

		// from the 1st rule
System.out.println("F: " + f);
System.out.println("Indep: " + this.powerSet.getIndependent(f.getClauses()));
System.out.println("MaxId: " + this.powerSet.getMaximal(this.powerSet.getIndependent(f.getClauses())));
		Set<Clause> sMaxIndpt = this.powerSet.getMaximal(this.powerSet.getIndependent(f.getClauses()));
		for (Clause c : sMaxIndpt) {
			Formula fprime = f.cloneAdd(c);
			sFprime.add(fprime);
		}

		// from 2nd rule
		Set<Clause> sAllDominated = new HashSet<Clause>();
		for (Clause c : sfClauses) {
			sAllDominated.addAll(this.powerSet.getDominatedDirectly(c));
		}

		// get maximal dominated sets, not included in an independent set sID
		List<Clause> lCandidates = new ArrayList<Clause>();
		for (Clause c : this.powerSet.getMaximal(sAllDominated)) {
			if (c.isIndependent(sMaxIndpt)) {
				lCandidates.add(c);
			}
		}

		HashSet<Clause> sCand = new HashSet<Clause>();
		for (int i = 0; i < lCandidates.size(); i++) {
			Clause ci = lCandidates.get(i);
			sCand.add(ci);
			Formula fPrime = this.getConsistentFormula(sfClauses, sCand);
			if (fPrime.isConsistent()) {
				sFprime.add(fPrime);
			} else {
				if (degenerated) {
					sFprime.add(fPrime);
				} else {
					for (int j = i + 1; j < lCandidates.size(); j++) {
						Clause cj = lCandidates.get(j);
						sCand.add(cj);
						fPrime = this.getConsistentFormula(sfClauses, sCand);
						if (fPrime.isConsistent()) {
							sFprime.add(fPrime);
						}
						sCand.remove(cj);
					}
				}
			}
			sCand.remove(ci);
		}

		return sFprime;
	}

	public Set<Formula> getFormulaDescendants(Formula fInit, boolean degenerated) {
		Set<Formula> sExplored = new HashSet<Formula>();
		Set<Formula> sToExplore = new HashSet<Formula>();
		sToExplore.add(fInit);
		while (!sToExplore.isEmpty()) {
			Formula f = sToExplore.iterator().next();
			sToExplore.remove(f);
			sToExplore.addAll(this.getFormulaChildren(f, degenerated));
			sExplored.add(f);
		}
		return sExplored;
	}


	public Set<Formula> getFormulaChildren(Formula f, boolean degenerated) {
		Set<Formula> sFSons = new HashSet<Formula>();
		Set<Clause> sFClauses = f.getClauses();
		Map<Clause, Formula> mR3sigma = new HashMap<Clause, Formula>();

		// Rule 3
		for (Clause cSigma : sFClauses) {
			Set<Clause> sFprime = new HashSet<Clause>(sFClauses);
			sFprime.remove(cSigma);
			Formula fprime = new Formula(this.nvars, sFprime);
			Set<Clause> sFsigma = this.powerSet.getIndependent(sFprime);
			if (!cSigma.isContainedIn(sFsigma) && fprime.isConsistent()) {
				sFSons.add(fprime);
				mR3sigma.put(cSigma, fprime);
			}
		}
		
		if (degenerated) {
			for (Formula fprime : mR3sigma.values()) {
				sFSons.add(fprime);
			}
			return sFSons;
		}
		
		// Rule 4
		Set<Clause> sCallDom = new HashSet<Clause>();
		for (Clause cSigma : sFClauses) {
			sCallDom.addAll(this.powerSet.getDominantDirectly(cSigma));
		}
		sCallDom = this.powerSet.getMinimal(sCallDom);
		Set<Clause> sCcandidates = new HashSet<Clause>();
		for (Clause cSigma : sCallDom) {
			if (!cSigma.contains(mR3sigma.keySet())) {
				sCcandidates.add(cSigma);
			}
		}
		
		// FIXME add degenerate
		for (Clause cSigma1 : sCcandidates) {
			Set<Clause> sFprime = new HashSet<Clause>(sFClauses);
			Set<Clause> sR = new HashSet<Clause>();
			for (Clause cSigma : sFClauses) {
				if (cSigma.dominatedStrictly(cSigma1)) {
					sR.add(cSigma);
					sFprime.remove(cSigma);
				}
			}
			sFprime.add(cSigma1);
			for (Clause cSigma2 : sCcandidates) {
				if (cSigma2.equals(cSigma1)) continue;
				boolean bFlag = true;
				for (Clause cSigma : sR) {
					if (!cSigma.dominatedStrictly(cSigma2)) {
						bFlag = false;
					}
				}
				if (bFlag) {
					sFprime.add(cSigma2);
				}
			}
			sFSons.add(new Formula(this.nvars, sFprime));
		}
		return sFSons;
	}

	private Formula getConsistentFormula(Set<Clause> sfClauses, Set<Clause> set) {
		Set<Clause> sCandSons = new HashSet<Clause>();
		for (Clause c : set) {
			sCandSons.addAll(powerSet.getDominantDirectly(c));
		}
		// remove dominated clauses
		Set<Clause> sPrime = new HashSet<Clause>(sfClauses);
		sPrime.removeAll(sCandSons);
		sPrime.addAll(set);

		return new Formula(nvars, sPrime);
	}

	public Formula genBottomFormula() {
		if (fBottom == null) {
			Set<Clause> fClauses = new HashSet<Clause>();
			BitSet bs = new BitSet(this.nvars);
			bs.set(0, this.nvars, true);
			fClauses.add(new Clause(this.nvars, bs));
			fBottom = new Formula(this.nvars, fClauses);
		}
		return fBottom;
	}
}
