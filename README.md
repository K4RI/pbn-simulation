# pbn-simulation
Une classe Python permettant de construire analyser et simuler des réseaux de régulation, sous la forme de réseaux booléens probabilistes.

### Prérequis

Logiciel(s) nécessaires au fonctionnement du projet : Java 1.8, Python 3

```python
pip install matplotlib, networkx, numpy, pandas
```

## Structure du dossier de travail
- **PBN_simulation.py** : définition des classes et des méthodes
- **script.py** : exemple de script utilisant les méthodes de la classe (tests sur des exemples biologiques ou générés synthétiquement)
- **output/** : dossier où sont consignées les sorties de PBN_to_file()
- **Experiments_Th_model/** : dossier de travail sur l'extension des BN grâce au PO-Set des fonctions de régulation (par C. Chaouiya, cf. [Cury2019, section 6.2](https://arxiv.org/abs/1901.07623))

## Présentation de la classe
La classe PBN définie dans `PBN_simulation.py` instancie des objets représentant des réseaux booléens probabilistes (ou *PBN*), mais peut aussi manipuler en particulier des réseaux booléens (ou *BN*).

### Attributs
La configuration d'un BN comporte un *état* $x \in \{0,1\}^n$ (attribut `self.x`), et un vecteur de fonctions booléennes appelé *contexte* $F = (f^{(1)} \dots f^{(n)})$ (`self.currentfct`, `self.currentfct_vector`) chacune d'arité $n$. Ces fonctions simulent une itération du réseau : la fonction $f^{(i)}$ met à jour le bit $x_i$ vers $f^{(i)}(x)$. Il existe deux dynamiques de mise à jour (`self.sync`) :
- *synchrone* : tous les $n$ bits de $x$ sont actualisés simultanément
- *asychrone* : un seul bit de $x$, tiré au hasard, est actualisé

De plus à chaque itération, chaque bit possède une faible probabilité (`self.p`) d'être inversé.

Dans le cas d'un PBN, le modèle a plusieurs contextes possibles. Avant chaque itération, le contexte peut être modifié avec une certaine probabilité (`self.q`), auquel cas il est pioché dans le liste des contextes (`self.fcts`, selon la distribution `self.c`).

### Méthodes
L'itération d'un PBN (`self.step()`, `self.simulation()`) s'effectue selon la procédure suivante :
1) avec une probabilité $q$, un changement de contexte a lieu
2) un *vecteur de perturbation* $\gamma$ est tiré selon la loi de Bernouilli multidimensionnelle $Ber(p)^n$

      a) Si $\gamma \neq 0$, les bits $i$ de $x$ tels $\gamma_i = 1$ sont inversés.

      b) Si $\gamma = 0$, on met à jour $x$ à partir du contexte comme présenté plus haut.

L'espace d'états $x \in \{0,1\}^n$, dont des nœuds $(x,y)$ sont reliés par un arc si et seulement si $y$ est un successeur de $x$ dans la dynamique du réseau, forme le graphe de transition d'états (ou *STG*) du réseau (`self.STG()`, `self.STG_PBN()`). Ses arcs sont pondérés par la probabilité de transition.

Une composante fortement connexe du STG est appelée *attracteur*, elle correspond à une ou plusieurs configurations dans lesquelles le modèle peut éventuellement boucler à long terme. Un échantillonnage des états visités lors d'une simulation à long terme et un affichage de la loi stationnaire empirique sous forme d'un histogramme (`self.stationary_law()`) permettent d'estimer la prévalence de ces attracteurs.

Une fonction booléenne associée à un bit $i$ dépend de certains bits $i_1, \dots, i_{k_i}$ appelés *régulateurs* de $i$. Les dépendances entre les composantes, pouvant être *activatrices*, *régulatrices*, ou *duales*, sont représentées dans un *graphe de régulation* (`self.regulatory_graph()`).


## Exécuter
Un fichier script importe la classe et ses fonctions annexes :
```python
from PBN_simulation import *
```

### Construire un réseau booléen probabiliste
Un objet PBN peut être créé en parcourant un fichier décrivant ses attributs et ses contextes (`file_to_PBN()`), ou en génération aléatoire (`generateBN()`, `generate_Random_PBN()` pouvant fixer leurs attributs).

### Analyser
*(cf. Méthodes)*

### Sauvegarder
Parallèlement à la fonction de lecture de fichier, il est possible de sauvegarder un PBN dans un fichier de même syntaxe (`self.PBN_to_file()`). Les fonctions booléennes y sont consignées en expressions DNF.


