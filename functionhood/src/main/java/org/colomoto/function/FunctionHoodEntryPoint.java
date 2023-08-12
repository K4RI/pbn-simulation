package org.colomoto.function;

import org.colomoto.function.core.Clause;
import org.colomoto.function.core.Formula;
import org.colomoto.function.core.HasseDiagram;
import org.colomoto.function.GetFunctionNeighbours;
import py4j.GatewayServer;

import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

public class FunctionHoodEntryPoint {

    private HasseDiagram hd;

    public FunctionHoodEntryPoint() {
        hd = new HasseDiagram(1);
    }

    public HasseDiagram getHasseDiagram() {
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
                sParent.add(c.toSet());
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
                sChild.add(c.toSet());
            }
            sChildren.add(sChild);
        }
        return sChildren;
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
