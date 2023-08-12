package org.colomoto.function;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import org.colomoto.function.core.Clause;
import org.colomoto.function.core.Formula;
import org.colomoto.function.core.HasseDiagram;

/**
 * Small GUI to compute the direct Parents/Siblings/Children of a given
 * function. The function is defined as a set of sets of regulators.
 * 
 * @author Pedro T. Monteiro
 * @author Jose' R. Cury
 * @author Claudine Chaouiya
 *
 */
public class GetFunctionNeighbours extends JFrame {
	private static final long serialVersionUID = -2458646831659346077L;

	private JTextField jtfDim;
	private JTextField jtfFunction;
	private JCheckBox jcbParents;
	private JCheckBox jcbSiblings;
	private JCheckBox jcbChildren;
	private JCheckBox jcbDegen;
	private JTextArea jtarea;

	public GetFunctionNeighbours() {
		this.setTitle("Function Direct Neighbours - GUI");
		this.setLayout(new BorderLayout());

		GridBagLayout gridbag = new GridBagLayout();
		JPanel top = new JPanel(gridbag);
		GridBagConstraints gbc = new GridBagConstraints();

		int y = 0;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.gridx = 0;
		gbc.gridy = y;
		top.add(new JLabel("Dimension:"), gbc);
		gbc.gridx = 1;
		this.jtfDim = new JTextField(5);
		this.jtfDim.setText("4");
		top.add(this.jtfDim, gbc);

		y++;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.gridx = 0;
		gbc.gridy = y;
		top.add(new JLabel("Function:"), gbc);
		gbc.gridx = 1;
		this.jtfFunction = new JTextField(30);
		this.jtfFunction.setText("{{1,2,3},{1,3,4},{2,4}}");//"{{2},{1,3}}");
		top.add(this.jtfFunction, gbc);

		y++;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.gridx = 1;
		gbc.gridy = y;
		this.jcbParents = new JCheckBox("Compute function parents");
		this.jcbParents.setSelected(true);
		top.add(this.jcbParents, gbc);
		y++;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.gridx = 1;
		gbc.gridy = y;
		this.jcbSiblings = new JCheckBox("Compute function siblings");
		this.jcbSiblings.setSelected(true);
		top.add(this.jcbSiblings, gbc);
		y++;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.gridx = 1;
		gbc.gridy = y;
		this.jcbChildren = new JCheckBox("Compute function children");
		this.jcbChildren.setSelected(true);
		top.add(this.jcbChildren, gbc);
		y++;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.gridx = 1;
		gbc.gridy = y;
		this.jcbDegen = new JCheckBox("Consider degenerate functions");
		this.jcbDegen.setSelected(false);
		top.add(this.jcbDegen, gbc);

		y++;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.gridx = 0;
		gbc.gridy = y;
		JButton jbRun = new JButton("Run");
		jbRun.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				jtarea.setText("");
				int nsize = Integer.parseInt(jtfDim.getText().trim());
				Formula f;
				try {
					f = parseFormula(nsize, jtfFunction.getText().trim());
					jtfFunction.setBackground(Color.WHITE);
				} catch (NumberFormatException nfe) {
					jtfFunction.setBackground(Color.RED);
					return;
				}
				HasseDiagram hd = new HasseDiagram(nsize);

				Set<Formula> sParents;
				if (jcbParents.isSelected() || jcbSiblings.isSelected()) {
					sParents = hd.getFormulaParents(f, jcbDegen.isSelected());
				} else {
					sParents = new HashSet<Formula>();
				}
				Set<Formula> sChildren;
				if (jcbChildren.isSelected() || jcbSiblings.isSelected()) {
					sChildren = hd.getFormulaChildren(f, jcbDegen.isSelected());
				} else {
					sChildren = new HashSet<Formula>();
				}

				if (jcbParents.isSelected()) {
					jtarea.append("------------------ Parents ------------------\n");
					for (Formula parent : sParents) {
						jtarea.append(parent.toString() + "\n");
					}
				}
				if (jcbSiblings.isSelected()) {
					Set<Formula> sSiblings = new HashSet<Formula>();
					jtarea.append("------------------ Siblings ------------------\n");
					for (Formula child : sChildren) {
						sSiblings.addAll(hd.getFormulaParents(child, jcbDegen.isSelected()));
					}
					for (Formula parent : sParents) {
						sSiblings.addAll(hd.getFormulaChildren(parent, jcbDegen.isSelected()));
					}
					sSiblings.remove(f);
					for (Formula brother : sSiblings) {
						jtarea.append(brother.toString() + "\n");
					}
				}
				if (jcbChildren.isSelected()) {
					jtarea.append("------------------ Children ------------------\n");
					for (Formula child : sChildren) {
						jtarea.append(child.toString() + "\n");
					}
				}
			}
		});
		top.add(jbRun, gbc);

		add(top, BorderLayout.PAGE_START);

		this.jtarea = new JTextArea("");
		add(this.jtarea, BorderLayout.CENTER);

		setSize(500, 500);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		setVisible(true);
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

	public static void main(String[] args) throws IOException {
		new GetFunctionNeighbours();
	}
}
