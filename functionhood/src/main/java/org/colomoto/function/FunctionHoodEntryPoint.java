package org.colomoto.function;

import org.colomoto.function.core.Clause;
import org.colomoto.function.core.Formula;
import org.colomoto.function.core.HasseDiagram;
import py4j.GatewayServer;

import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

public class FunctionHoodEntryPoint {

    private HasseDiagram hd;

    public FunctionHoodEntryPoint() {
        hd = new HasseDiagram(1);
    }

    public HasseDiagram initHasseDiagram(int nvars) {
        hd = new HasseDiagram(nvars);
        return hd;
    }

    public Set<Set<Set<Integer>>> getFormulaParentsfromStr(String s, boolean degen) {
        Formula f;
        f = parseFormula(hd.getSize(), s.trim());
        Set<Formula> Parents;
        Parents = hd.getFormulaParents(f, degen);
        Set<Set<Set<Integer>>> sParents = new HashSet<Set<Set<Integer>>>();
        for (Formula parent : Parents) {
            Set<Set<Integer>> sParent = new HashSet<Set<Integer>>();
            for (Clause c : parent.getClauses()){
                sParent.add(toSet(c));
            }
            sParents.add(sParent);
        }
        return sParents;
    }

    public Set<Set<Set<Integer>>> getFormulaChildrenfromStr(String s, boolean degen) {
        Formula f;
        f = parseFormula(hd.getSize(), s.trim());
        Set<Formula> Children;
        Children = hd.getFormulaChildren(f, degen);
        Set<Set<Set<Integer>>> sChildren = new HashSet<Set<Set<Integer>>>();
        for (Formula children : Children) {
            Set<Set<Integer>> sChild = new HashSet<Set<Integer>>();
            for (Clause c : children.getClauses()){
                sChild.add(toSet(c));
            }
            sChildren.add(sChild);
        }
        return sChildren;
    }

    public Set<Integer> toSet(Clause c) {
        Set<Integer> s = new HashSet<Integer>();
        for (int i = 0; i < hd.getSize(); i++) {
            if (c.getSignature().get(i)) {
                s.add(i + 1);
            }
        }
        return s;
    }

    private static Clause parseClause(int n, String s) throws NumberFormatException {
        s = s.substring(1, s.length() - 1);
        BitSet bs = new BitSet(n);
        for (String r : s.split(",")) {
            bs.set(Integer.parseInt(r) - 1, true);
        }
        return new Clause(n, bs);
    }

    private static Formula parseFormula(int n, String s) throws NumberFormatException {
        s = s.substring(1, s.length() - 1);
        Set<Clause> fClauses = new HashSet<Clause>();
        for (String clause : s.split("},")) {
            if (clause.charAt(clause.length() - 1) != '}') {
                clause = clause + "}";
            }
            fClauses.add(parseClause(n, clause));
        }
        return new Formula(n, fClauses);
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new FunctionHoodEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }

}
