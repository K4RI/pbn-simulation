package org.colomoto.function.core;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 
 * @author Pedro T. Monteiro
 * @author Jose' R. Cury
 *
 */
public class Formula {
	private int nvars;
	private BitSet varRepresented;
	private Set<Clause> clauses;
	private boolean consistent;

	public Formula(int nvars, Set<Clause> clauses) {
		this.nvars = nvars;
		this.clauses = new HashSet<Clause>(clauses);
		this.updateConsistency();
	}

	// Only to be used by cloneAdd(Clause c)
	private Formula() {
	}

	public Formula cloneAdd(Clause c) {
		Formula f = new Formula();
		f.nvars = this.nvars;
		f.clauses = new HashSet<Clause>(this.clauses);
		f.clauses.add(c);
		f.varRepresented = new BitSet(f.nvars);
		for (Clause ci : f.clauses) {
			f.varRepresented.or(ci.getSignature());
		}
		f.updateConsistency();
		return f;
	}

	public int getNumberVars() {
		return this.nvars;
	}

	public float clausesAvgLength() {
		float f = 0;
		for (Clause c : this.clauses) {
			f += c.order();
		}
		return f / this.clauses.size();
	}

	public Set<Formula> cloneSub() {
		Set<Formula> sFMinus = new HashSet<Formula>();
		for (Clause c : this.clauses) {
			Set<Clause> sCopy = new HashSet<Clause>(this.clauses);
			sCopy.remove(c);
			Formula fMinus = new Formula(this.nvars, sCopy);
			sFMinus.add(fMinus);
		}
		return sFMinus;
	}

	public int hashCode() {
		int hash = 0;
		for (Clause c : this.clauses) {
			hash += c.hashCode();
		}
		return hash;
	}

	public boolean equals(Object o) {
		Formula fo = (Formula) o;
		return (this.nvars == fo.nvars) && this.clauses.equals(fo.clauses);
	}

	public void addClause(Clause c) {
		this.clauses.add(c);
		this.varRepresented.or(c.getSignature());
		this.updateConsistency();
	}

	public Set<Clause> getClauses() {
		return Collections.unmodifiableSet(this.clauses);
	}

	public boolean isConsistent() {
		return this.consistent;
	}

	private void updateConsistency() {
		this.consistent = this.independentClauses() && this.isCover();
	}

	private boolean independentClauses() {
		List<Clause> lClauses = new ArrayList<Clause>(this.clauses);
		for (int i = 0; i < (lClauses.size() - 2); i++) {
			for (int j = i + 1; j < (lClauses.size() - 1); j++) {
				if (!lClauses.get(i).isIndependent(lClauses.get(j))) {
					return false;
				}
			}
		}
		return true;
	}

	private boolean isCover() {
		this.varRepresented = new BitSet(this.nvars);
		for (Clause c : clauses) {
			this.varRepresented.or(c.getSignature());
		}
		return this.varRepresented.cardinality() == this.nvars;
	}

	public boolean isSmallerThan(Formula f) {
		for (Clause tf : this.clauses) {
			boolean bSmall = false;
			for (Clause cf : f.clauses) {
				if (tf.dominatedOrEqualTo(cf)) {
					bSmall = true;
					break;
				}
			}
			if (!bSmall) {
				return false;
			}
		}
		return true;
	}

	public String toString() {
		String sTmp = "{";
		boolean first = true;
		for (Clause c : this.clauses) {
			if (!first)
				sTmp += ",";
			first = false;
			sTmp += c;
		}
		return sTmp + "}";
	}
}
