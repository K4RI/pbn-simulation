package org.colomoto.function.core;

import java.util.BitSet;
import java.util.Set;

/**
 * Clause is represented as a bit vector (true/false) of size n. By default all
 * bits are set to true (representing a more specific clause). Clauses can be
 * compared against other clauses.
 * 
 * @author Pedro T. Monteiro
 * @author Jose' R. Cury
 * @author Claudine Chaouiya
 * 
 */
public class Clause {
	private int size;
	private BitSet signature;

	public Clause(int nvars, BitSet signature) {
		this.size = nvars;
		this.signature = signature;
	}

	public Clause(int nvars) {
		this(nvars, new BitSet(nvars));
		this.signature.set(0, nvars, true);
	}

	public BitSet getSignature() {
		return this.signature;
	}

	public int hashCode() {
		return this.signature.hashCode();
	}

	public int order() {
		return this.signature.cardinality();
	}

	public boolean equals(Object o) {
		Clause co = (Clause) o;
		if (this.size != co.size || this.signature.cardinality() != co.signature.cardinality())
			return false;
		for (int i = 0; i < this.size; i++) {
			if (this.signature.get(i) != co.signature.get(i)) {
				return false;
			}
		}
		return true;
	}

	public boolean dominatesOrEqualTo(Clause c) {
		for (int i = 0; i < this.size; i++) {
			if (c.signature.get(i) && !this.signature.get(i)) {
				return false;
			}
		}
		return true;
	}
	
	public boolean dominatesStrictly(Clause c) {
		return this.dominatesOrEqualTo(c) && !this.equals(c);
	}

	public boolean dominatedOrEqualTo(Clause c) {
		return c.dominatesOrEqualTo(this);
	}
	
	public boolean dominatedStrictly(Clause c) {
		return c.dominatesStrictly(this);
	}

	public boolean isSet(int pos) {
		return (pos < this.size) && (pos >= 0) && this.signature.get(pos);
	}

	public boolean isIndependent(Clause c) {
		return !this.equals(c) && !this.dominatesOrEqualTo(c) && !this.dominatedOrEqualTo(c);
	}

	public boolean isIndependent(Set<Clause> sClauses) {
		for (Clause c : sClauses) {
			if (!this.isIndependent(c)) {
				return false;
			}
		}
		return true;
	}
	
	public boolean contains(Set<Clause> sClauses) {
System.out.println(" " + this + " contains? any of " + sClauses);
		for (Clause c : sClauses) {
			if (this.dominatesOrEqualTo(c)) {
				System.out.println("  true");
				return true;
			}
		}
		System.out.println("  false");
		return false;
	}
	
	public boolean isContainedIn(Set<Clause> sClauses) {
		for (Clause c : sClauses) {
			if (this.dominatedStrictly(c)) {
				return true;
			}
		}
		return false;
	}


	public String toString() {
		String s = "{";
		boolean first = true;
		for (int i = 0; i < this.size; i++) {
			if (this.signature.get(i)) {
				if (!first)
					s += ",";
				first = false;
				s += i + 1;
			}
//			s += this.signature.get(i) ? i+1 : "";
		}
		return s + "}";
	}
}
