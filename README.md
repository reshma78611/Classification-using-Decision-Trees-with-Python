# Classification using Decision Trees 

## DECISION TREE

A Decision tree is used for both classification and regression .It can be used to visually and explicitly represent decisions and decision making. The decision trees are built using heuristic called *Recursive Partitioning*.

Algorithms  used:-

1)	C5.0 ( Uses Entropy)
2)	CART(Uses GINI Index)
3)	ID3 (Uses Entropy and  Information Gain)

As the name goes, it uses a tree-like model of decisions.A decision tree is drawn upside down with its root at the top. It contains an **internal node**, based on which the tree splits into **branches(Branch node)**. The end of the branch that doesn’t split anymore is the decision(**Leaf node**).

The performance of a tree can be increased by *Pruning*. It involves removing the branches that make use of features having low importance. 

## Data Used :

	               Company dataset - for knowing the attribute that causes high sale using descision tree.
	               Fraud dataset -  for treating those who have taxable_income <= 30000 as "Risky" and others are "Good" using descision tree.
                 Iris dataset - for classifying the species

## Programming:

                 Python


**The Codes regarding Decision Tree Classifier with *Company classification from company dataset, Risky Classification from Fraud dataset, Species classification from Iris dataset* are present in this Repository in detail.**




