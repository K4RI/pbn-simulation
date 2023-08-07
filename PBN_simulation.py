# -*- coding: utf-8 -*-
# Created on Thu Jul 6 2023
# @author: K4RI

from collections import Counter
import itertools
from matplotlib import cm
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random



########### PARTIE 1 / 3 - FONCTIONS ANNEXES ##########

def truth_table1(F: list, n: int) -> str :
    """Affiche la table de vérité pour des fonctions booléennes.

    Parameters
    ----------
    F : list
        Liste de fonctions booléennes.
    n : int
        Arité des fonctions booléennes de F.

    Returns
    -------
    str
        La chaîne de caractères représentant la table de vérité.
    """

    L = list(map(list, itertools.product([0, 1], repeat = n)))
    s = ''
    for x in L :
        s += '%s || %s \n' %(''.join([str(u) for u in x]),
                             ' '.join([str(int(f(x))) for f in F]))
    return s


def truth_table2(F: list, n: int) -> str:
    """Identique à la précédente, mais F est séparée en sous-listes."""
    L = list(map(list, itertools.product([0, 1], repeat = n)))
    s = ''
    for x in L :
        s += '%s || %s \n' %(''.join([str(u) for u in x]),
                             ' - '.join([' '.join([str(int(f(x)))
                                         for f in ff]) for ff in F]))
    return s


def Hamming(x: list, y: list) -> int:
    """Renvoie la distance de Hamming entre deux mots binaires."""
    return len([i for i in range(len(x)) if x[i] != y[i]])

def prod(L: list) -> int:
    """Renvoie le produit des termes d'une liste."""
    if not L: return 1
    return L[0] * prod(L[1:])


def fct_to_clauseDNF(fct: int, neigh_states_i: list, neighs_i: list) -> str:

    clauses = []
    for antec in neigh_states_i:
        if fct(antec):
            litts = []
            for ind in neighs_i:
                if antec[ind]:
                    litts.append(f'x{ind}')
                if not antec[ind]:
                    litts.append(f'!x{ind}')
            clauses.append('(' + ' & '.join(litts) + ')')
    return ' | '.join(clauses)


def clause_to_fct(factors: str, targets: dict) -> str:
    targs = list(targets.keys())
    for y in sorted(targs, key = len, reverse = True):
        factors = factors.replace(str(y), 'x[%s]' %targets[y])

    # Remplacement des '!(...)' par des '(not ...)'
    while '!' in factors:
        fac0, fac1 = factors.split('!', 1)
        if fac1[0] != '(':
            fac11, fac12 = fac1.split(']', 1)
            factors = fac0 + '(not ' + fac11 + '])' + fac12

        else:
            cpt_par = 1
            i = -1
            while cpt_par:
                i += 1
                if fac1[i] == '(':
                    cpt_par+=1
                elif fac1[i] == ')':
                    cpt_par-=1
            factors = fac0 + '(not ' + fac1[:i] + '])' + fac2[i:]

    return factors


########### PARTIE 2/3 - CLASSE 'PBN' ##########

class PBN:
    """
    Une classe représentant un réseau booléen probabiliste, ou 'PBN'.
    Ce modèle permet de simuler des réseaux de régulation biologique.

    L'état d'un PBN est représenté par un liste de bits de taille fixée, dont
    chacun des bits est mis à jour selon sa fonction associée appelée sa
    'fonction de transition'.

    Si tous les bits n'ont qu'une seule fonction de transition, le modèle est
    un réseau booléen, ou 'BN'. S'il existe au moins un bit associé à plusieurs
    fonctions, le modèle est un PBN. Le choix de la fonction sera alors
    aléatoire parmi elles et dépendra des paramètres du modèle.

    Attributes
    ----------
    title : str
        Nom donné au PBN.
    n : int
        Nombre de bits décrivant un état.
    x : list
        Liste de self.n bits représentant l'état du système.
    currentfct_vector : list
        Liste actuelle des fonctions de transition sélectionnées, aussi
        appelée "contexte".
    fcts : list
        Fonctions booléennes décrivant la dynamique du réseau.
        Une fonction associée au i-ème bit prend en argument l'état du PBN, et
        renvoie la valeur suivante du i-ème bit.
        si indep==True : de la forme F_1 x ... x F_n, avec F_i les fonctions de
        transition possibles du i-ème bit.
        Si indep==False : de la forme [f_1 ... f_m], avec f_i un contexte.
    c : list
        Probabilités de sélection des fonctions booléennes. De même forme
        que fcts.
    currentfct : int, list
        Indice(s) des fonctions du contexte currentfct_vector dans fcts.
    sync : bool
        Mode de mise à jour du PBN, True si synchrone et False si asynchrone.
    indep : bool
        Indépendance entre les choix de fonctions de transition
        pour différents bits.
    regulation : int
        Dans le cas où le modèle est un BN,
                                  ou un PBN ayant un contexte de référence.
        Si len == 0 : le réseau n'est pas basé sur une régulation,
                      regulation = tuple vide.
        Si len == 1 : le réseau est basé sur une régulation non-signée,
                      regulation = (liste des voisins,)
        Si len == 2 : le réseau est basé sur une régulation signée,
                      regulation = (liste des activateurs, liste des inhibiteurs)
        Attribut utilisé dans la méthode regulation_graph().
    p : float
        Probabilité de perturber un bit.
    q : float
        Probabilité de changer de fonction.


    Methods
    -------
    __init__()

    simulation(N)
        Effectue N itérations du PBN.
    STG()
        Affiche le graphe de transition d'états du modèle.
    stationary_law()
        Échantillonne les simulations dans l'espace d'états.
    regulation_graph()
        Affiche, si self.regulation > 0, le graphe de régulation du réseau.
    """


    def __init__(self, n, indep, f, c, sync, p, q, title = '',
                zeroes = [], ones = [], regulation = ()):
        """Crée un objet de type PBN, dont l'état x
        Parameters
        ----------
        args :
            Attributs du PBN décrits plus haut.
        zeroes : list
            Liste des indices des bits à initialiser à 0.
        ones : list
            Liste des indices des bits à initialiser à 1.


        Raises
        -------
        TypeError, ValueError
            Un des arguments n'a pas la forme attendue.
        """

        self.title = title
        if sync: s_sync = 'sync'
        else: s_sync = 'async'
        s_indep = 'indep' * indep

        # son titre pour les figures
        self.longtitle = f'{title} {s_sync} {s_indep}'

        # # Nombre de gènes
        self.n = n
        if type(n) != int or n <= 0:
            raise TypeError('n doit être un entier strictement positif.')

        # # Les marginales de la fonction de transition sont-elles indépendantes
        self.indep = indep
        if type(indep) != bool:
            raise TypeError('indep doit être un argument booléen.')

        # # Liste des fonctions de transition.
        # Si indep==True, le i-ème elt° est les fcts possibles du gène i.
        # Si indep==False, le i-ème elt est la i-ème contexte.
        self.fcts = f
        if indep:
            self.m = [len(fct) for fct in f]
            if len(f) != n:
                raise ValueError("Tous les gènes n'ont pas eu "
                                 "leurs fonctions spécifiées dans f." %i)
        if not indep:
            self.m = len(self.fcts)
            for i in range(self.m):
                if len(f[i]) != n:
                    raise ValueError("Le contexte %i dans f"
                                     "n'a pas le bon nombre de marginales." %i)

        # # Distribution de proba sur les fonctions de transition.
        # Si indep==True, le i-ème elt est la distribution sur les fcts du gène i.
        # Si indep==False, distribution sur les contextes.
        self.c = c
        if (not indep) and abs(sum(c)-1) > 0.05 :
            raise ValueError("La masse totale de la distribution c"
                             "doit valoir 1.")
        if indep:
            if len(c) != n:
                raise ValueError("La distribution c doit être définie"
                                 "sur l'ensemble des contextes.")
            for i in range(n):
                if abs(sum(c[i]) - 1) > 0.05 :
                    raise ValueError("La masse totale de la distribution"
                                     "des fonctions du gène %i dans c"
                                     "doit valoir 1." %i)

        # Facteur de perturbation
        self.p = p
        if type(p) not in [int, float] or p < 0 or p > 1:
            raise TypeError("p doit être un flottant compris entre 0 et 1.")

        # Facteur de changement de contexte
        self.q = q
        if type(q) not in [int, float] or q < 0 or q > 1:
            raise TypeError("q doit être un flottant compris entre 0 et 1.")

        # Est-ce un PBN ?
        self.pbn = self.indep or len(self.c)>1

        self.regulation = regulation

        # # Mise à jour synchrone ou asynchrone ?
        self.sync = sync
        if type(sync) != bool:
            raise TypeError("sync doit être un argument booléen.")

        # Initialiser l'état x \in {0,1}^n
        self.zeroes = zeroes
        self.ones = ones
        self.init_state()

        # Initialiser le contexte
        self.currentfct = [0]*n
        self.switch()


    def __str__(self):
        """
        Returns
        ----------
        str
            Description exhaustive des paramètres du modèle.
        """

        if self.pbn:
            s_type = 'PBN'
        else:
            s_type = 'BN'

        if self.sync:
            s_sync = 'synchrone'
        else:
            s_sync = 'asynchrone'

        if self.indep:
            s_ind = 'à tirages indépendants'
            s_fcts = ''
            for i in range(self.n):
                s_fcts += 'x%i - %i fonctions possibles\n' \
                          %(i, len(self.fcts[i]))
        else:
            s_ind = ''
            s_fcts = '%i contextes' %len(self.fcts)

        if self.indep:
            s_ctxt = ', '.join(['f%i_%i'%(i, self.currentfct[i])
                                for i in range(self.n)])
        else:
            s_ctxt = 'f_' + str(self.currentfct)

        # on inclut la table de vérité
        s_ctxtbls = ' \n'
        s_ctxtbl = ' \n'
        if self.n <= 5:
            s_ctxtbls = truth_table2(self.fcts, self.n)
            if self.pbn:
                s_ctxtbl = truth_table1(self.currentfct_vector, self.n)

        return '''======================================
'%s'
%s %s %s (n=%i gènes)

Facteur de perturbation : p=%.2f
Facteur de changement de contexte : q=%.2f

Fonctions de transition :
%s
%sDistribution(s) de probabibité : %s

État actuel : %s

Contexte actuel : %s
%s======================================''' \
%(self.title, s_type, s_sync, s_ind, self.n, self.p, self.q, s_fcts, s_ctxtbls,
str(self.c), str(self.x), s_ctxt, s_ctxtbl)


    def copy_PBN(self, title = None, n = None, indep = None, fcts = None,
                 c = None, sync = None, p = None, q = None):
        """Copie un PBN.

        Parameters
        ----------
        args :
            Arguments à modifier.

        Returns
        -------
        PBN
            Copie du PBN initial, mais dont les paramètres spécifiés
            ont été changés.
        """

        if title == None: title = self.title
        if n == None: n = self.n
        if indep == None: indep = self.indep
        if fcts == None: fcts = self.fcts
        if c == None: c = self.c
        if sync == None: sync = self.sync
        if p == None: p = self.p
        if q == None: q = self.q

        return PBN(title = title,
                   n = n,
                   indep = indep,
                   f = fcts,
                   c = c,
                   sync = sync,
                   p = p,
                   q = q)


    def init_state(self):
        """Initialise l'état x dans l'espace d'états {0,1}^n."""

        self.x = random.choices([0,1], k = self.n)
        for i in self.zeroes:
            self.x[i] = 0
        for j in self.ones:
            self.x[j] = 1


    def step(self, verb = False):
        """Effectue une itération du PBN, pouvant modifier x et/ou currentfct."""

        if random.random() <= self.q:
            self.switch(verb)

        gamma = [random.random() <= self.p for _ in range(self.n)]
        if any(gamma):
            self.flip(gamma, verb)
        else:
            self.call(verb)


    def flip(self, gamma, verb = False):
        """ Une perturbation du PBN.

        Parameters
        ----------
        gamma : list
            Liste de 0 et de 1. Dans x, les bits i tels que gamma[i]==1
            se font modifier leur valeur.
        """

        for i in range(self.n):
            if gamma[i]:
                (self.x)[i] = 1-(self.x)[i]
        if verb: print('Perturbation des gènes :  %s' \
                        %(' '.join([i for i in range(self.n) if gamma[i]])))


    def switch(self, verb = False):
        """Tire et met à jour un contexte pour le PBN dans fcts,
        selon la.les distribution.s décrites dans c."""

        if self.indep:
            for i in range(self.n):
                self.currentfct[i] = random.choices(
                                    [_ for _ in range(self.m[i])],
                                    self.c[i])[0]
            self.currentfct_vector = [self.fcts[i][self.currentfct[i]]
                                      for i in range(self.n)]
        else:
            self.currentfct = random.choices(
                                    [_ for _ in range(self.m)],
                                    self.c)[0]
            self.currentfct_vector = self.fcts[self.currentfct]

        if verb:
            if self.indep:
                print('Changement de contexte : ',
                      ' , '.join(['f%i_%i'%(i, self.currentfct[i])
                                  for i in range(self.n)]))
            else:
                print('Changement de contexte : ', 'f_' + str(self.currentfct))


    def succ(self, x, f):
        """Renvoie la liste des successeurs d'un x par un contexte f."""

        # Cas synchrone : tous les bits sont mis à jour
        if self.sync:
            return [[int((f[i])(x)) for i in range(self.n)]]

        # Cas asynchrone : toutes les possibilités de mise à jour de chaque bit
        else:
            xs = [int((f[i])(x)) for i in range(self.n)]
            i_modifs = [i for i in range(self.n) if x[i] != xs[i]]
            if i_modifs == []: # si état stable
                return [x]
            succs = []
            for i in i_modifs:
                x1 = x.copy()
                x1[i] = 1-x1[i]
                succs.append(x1)
            return succs


    def call(self, verb=False):
        """Dans le PBN, met à jour x par le contexte f."""

        # TODO: par rapport à 1107_1.py, calculer tous les succ() augmente le temps de calcul de ~10% :(
        x0=self.x.copy()

        # L'état suivant est tiré parmi les successeurs de x
        ss = self.succ(self.x, self.currentfct_vector)
        self.x = random.choice(ss)
        if verb:
            print('Appel de la fonction   :  %s -> %s' %(str(x0), str(self.x)))
            if x0==self.x:
                print('--->> État stable <<---')


    def simulation(self, N, verb = False):
        """Simulation N étapes du PBN."""

        for i in range(1, N+1):
            if verb: print('__________\n\nÉTAPE %i' %i)
            self.step(verb)


    def stationary_law(self, show_all = True, T = 100, N = 200, R = 100,
                       pre = False, prio_attrs = []):
        """Affiche, sous forme d'un diagramme en barre, la loi stationnaire
        empirique de la chaîne de Markov associée au PBN.

        Parameters
        ----------
        show_all : bool
            Mode d'affichage du diagramme. Si False, on n'y place que les
            barres non-vides.
        R : int
            Nombre de simulations.
        T : int
            Nombre d'itérations initiales dans chaque simulation.
        N : int
            Nombre d'itérations pour lesquelles on consigne l'état visité,
            après les T premières.
        pre : bool
            Si True, ne renvoie que le set des états visités.
        prio_attrs : list
            Cas particulier pour l'étude de mammalian_cell_cycle.
            Liste des attracteurs dans la mise à jour en classe de priorité,
            dont on observe les prévalences dans une mise à jour asynchrone.

        Returns
        -------
        pandas.Dataframe
            Fréquences de visite de chaque état au cours de la simulation.
        """

        # Échantillon de la chaîne de Markov
        sim = []
        for i in range(R):
            self.init_state()
            for t in range(T):
                self.step()
            for t in range(N):
                self.step()
                sim.append(''.join(map(str,self.x)))

        if pre: # utilisé par stationary_law() en asynchrone
            return set(sim)

        # Traitement et affichage des états parcourus
        count = Counter(sim)
        count = {i : count[i] / len(sim) for i in count}
        df = pd.DataFrame.from_dict(count, orient = 'index',
                                    columns = ['Fréquence'])
        if show_all:
            bins = list(map(lambda t : ''.join(map(str, t)),
                            itertools.product([0, 1],repeat=self.n) ))
            df = df.reindex(bins, fill_value = 0)
        df = df.sort_index()
        color = len(df) * ['blue']
        plot_blue = mpatches.Patch(color = 'blue', label = 'Fréquence')

        # En asynchrone, on affiche dans une autre couleur les attracteurs
        # présents en simulation synchrone afin d'évaluer leur prévalence.
        if (not self.sync) and (not self.pbn) :
            df['ranking'] = df['Fréquence'].rank(method = 'min',
                                                 ascending = False)
            sim_s = self.copy_PBN(sync = True) \
                        .stationary_law(T = T, N = N, R = R, pre =True)
            rank_s = []

            if prio_attrs:
                rank_p=[]
                plot_green = mpatches.Patch(
                                color = 'green',
                                label = 'Fréquence (attracteurs présents ' \
                                        'en classe de priorité)')
                plot_pink = mpatches.Patch(color = 'pink',
                                           label = '(présents dans les deux)')
                for i in range(len(df)):
                    if df.index[i] in prio_attrs:
                        rank_p.append(int(df['ranking'][i]))
                        color[i] = 'green'
                rank_p.sort()

            for i in range(len(df)):
                if df.index[i] in sim_s:
                    rank_s.append(int(df['ranking'][i]))
                    color[i] = 'red'
                    if df.index[i] in prio_attrs:
                        color[i] = 'pink'
            plot_red = mpatches.Patch(color = 'red', label = 'Fréquence '\
                                        '(attracteurs présents en synchrone)')
            rank_s.sort()

            print('%i états visités en asynchrone' %len(df))
            print('dont %i états visités en synchrone (rangs : %s, médiane=%i)'\
                  %(len(sim_s), str(rank_s), np.mean(rank_s)))
            if prio_attrs:
                print(' (rangs des attracteurs de la classe de priorité : '
                      '%s, médiane=%i)' %(str(rank_p), np.mean(rank_p)))


        df.plot(kind = 'bar', y = 'Fréquence', color = color)
        plt.gca().set(title = f'{self.longtitle} - '
                              f'Distribution empirique pour {R} simulations')

        if not self.pbn:
            if self.sync == False:
                if prio_attrs:
                    plt.legend(handles = [plot_blue, plot_red,
                                        plot_green, plot_pink])
                else:
                    plt.legend(handles = [plot_blue, plot_red])
            else:
                plt.legend(handles = [plot_blue])
                # En synchrone, on partitionne les traits de hauteur similaire
                # Ils peuvent correspondre aux bassins d'attraction
                classes = []
                for i in range(len(df)):
                    if df['Fréquence'][i] > 0:
                        flag = False
                        for c in classes:
                            m = np.mean(df['Fréquence'][c])
                            if 0.98 * m <= df['Fréquence'][i] <= 1.02 * m:
                                c.append(df.index[i])
                                flag = True
                                break
                        if not flag:
                            classes.append([df.index[i]])
                print('\nLes %i classes supposées sont : ' %len(classes))
                for c in classes:
                    size = round(sum(df['Fréquence'][c]) * 2**(self.n))
                    #TODO : marge d'erreur à 95%
                    freq100 = 100 * sum(df['Fréquence'][c])
                    print('%s : bassin de taille %i (%.1f %%)' \
                        %(str(c), size, freq100))

        plt.subplots_adjust(bottom = 0.03 + 0.02 * self.n)
        plt.show()
        return df


    def stationary_law2(self, approach_attrs, attr_colors, attr_names,
                        T = 100, R = 1000):
        """Autre cas de simulation, dans le cas avec que des états stables.
        Affiche les attracteurs où finissent plusieurs simulations, dans une
        matrice de pixels où chaque couleur représente un attracteur.
        Fonction employée dans test_claudine().

        Parameters
        ----------
        approach_attrs : function
            À partir d'un état x, renvoie son attracteur le plus similaire à
            partir de ses 'marqueurs'.
        attr_names : list
            Attracteurs attendus.
        attr_colors : list
            Couleurs souhaitées pour la matrice de pixels. De même taille
            que attr_names.
        R : int
            Nombre de simulations.
        T : int
            Nombre d'itérations initiales dans chaque simulation.

        Returns
        -------
        matrix, matrix, pandas.Dataframe, pandas.Dataframe
            Les deux matrices de pixels affichées en synchrone et en asynchrone,
            et les pourcentages de chaque attracteur.
        """

        if not approach_attrs:
            raise ValueError('Préciser les marqueurs des attracteurs.')

        fig, (a1,a2) = plt.subplots(2)

        # On simule et affiche les matrices de pixels en sync puis async
        Ms, dfs = self.stationary_law2_annex(approach_attrs, attr_colors, T, R)
        Ma, dfa = self.copy_PBN(sync = False) \
                      .stationary_law2_annex(approach_attrs, attr_colors, T, R)

        patchList = []
        attr_colors_norm = [(r/255, g/255, b/255) for r,g,b in attr_colors]
        for c, a in zip(attr_colors_norm, attr_names):
            patchList.append([mpatches.Patch(color = c, label = a)])
        fig.legend(handles = patchList, labels = attr_names,
                   handler_map = {list: HandlerTuple(None)}, loc='upper right',
                   ncol=5, fontsize=10)

        plt.show()
        return Ms, Ma, dfs, dfa


    def stationary_law2_annex(self, approach_attrs, attr_colors, T, R):
        """Fonction annexe à la précédente, simule et charge l'image.

        Returns
        -------
        matrix,pandas.Dataframe, pandas.Dataframe
            La matrice de pixels et les pourcentages de chaque attracteur.
        """

        if self.sync:
            print("\n\n----------- (S) LECTURE DE '%s'" %self.title)
            plt.subplot(211)
        else:
            print("\n\n----------- (A) LECTURE DE '%s'" %self.title)
            plt.subplot(212)
        plt.tick_params(left = False, labelleft = False, labelbottom = False,
                                                         bottom = False)

        # Simulations
        sim = []
        cols = []
        for i in range(R):
            self.init_state()
            for t in range(T):
                self.step()
            # Aux termes des premières itérations,
            # on ne s'arrête qu'à l'état stable
            x_prec = None
            while x_prec != self.x:
                x_prec = self.x.copy()
                self.step()
            xs = ''.join(map(str,self.x))
            attr, col = approach_attrs(xs)
            sim.append(attr)
            cols.append(col)

        # Calcul des prévalences de chaque attracteur
        count = Counter(sim)
        count = {i : 100 * count[i] / len(sim) for i in count if count[i] > 0}
        df = pd.DataFrame.from_dict(count, orient = 'index',
                                    columns = ['Fréquence'])
        df = df.sort_index()
        print(df)

        # Construction et chargement de la matrice de pixels
        M = np.array([attr_colors[col] for col in cols])\
              .reshape((R//100, 100, 3))
        plt.imshow(M)
        plt.title(self.longtitle, fontsize = 10)
        return M, df


    def STG(self, f = None, layout = nx.spring_layout, pre = 0,
                            plot_attrs = True, draw_labels = True):
        """Affiche le graphe de transition d'états, représentant quelle et
        quelle configuration peuvent être reliées par un appel de la fonction f.

        Parameters
        ----------
        f : list
            Contexte.
        layout : function
            Mode d'affichage du graphe dans le module Networkx.
            Exemples : nx.spring_layout, nx.shell_layout, nx.planar_layout.
        pre : bool
            Si 1, on retourne le graphe.
            Si 2, on retourne le graphe et la liste des attracteurs.
            Si 0, on ne fait qu'afficher sans retourner de sortie.
        plot_attrs : bool
            Si True, des fenêtres supplémentaires affichent le STG restreint
            à chaque attracteur cyclique ou complexe.
        draw_labels : bool
            Si True, affiche les arêtes de probabilité dans les cas asynchrone
            ou PBN.

        Returns
        -------
        None ; nx.DiGraph ; nx.DiGraph, list
            Le graphe orienté pondéré dont les noeuds sont les états de
            {0,1]^n, les arêtes sont les liens de succession possibles, et les
            poids des arêtes sont les probabilités de transition.
            La liste des attracteurs du modèle.
        """

        if f == None:
            f = self.currentfct_vector

        G = nx.DiGraph()
        nodes_str = list(map(lambda t : ''.join(map(str, t)),
                             itertools.product([0, 1], repeat = self.n)))
        G.add_nodes_from(nodes_str)
        nodes = list(map(list, itertools.product([0, 1], repeat = self.n)))

        # On construit la liste d'adjacence
        for x in nodes:
            successors = self.succ(x, f)
            v = len(successors)
            for y in successors: ##
                G.add_edge(''.join([str(u) for u in x]),
                           ''.join([str(v) for v in y]), weight = 1/v)

        if pre == 1: # utilisé par STG_PBN()
            return G

        return self.post_process_STG(G, layout, pre, plot_attrs, draw_labels)


    def STG_PBN(self, layout = nx.spring_layout, pre = 0, plot_attrs = True,
                draw_labels = True):
        """Identique à la précédente, mais pour le graphe d'un PBN."""

        M = Counter()

        # Pour chaque contexte possible, on calcule son graphe de transition
        # et on fait la moyenne pondérée des poids des arêtes.

        if not self.indep:
            # ensemble des contextes : fcts
            # pondéré par un c[i]
            for i in range(len(self.fcts)):
                Gf = self.STG(f = self.fcts[i], pre = 1)
                M += Counter({(u,v): self.c[i] * Gf[u][v]['weight']
                                                    for u,v in Gf.edges})

        else:
            # ensemble des contextes : produit cartésien des classes fcts[i]
            # pondéré par le produit des probabilités marginales
            Fc = [[(self.fcts[i][j], self.c[i][j])
                    for j in range(len(self.fcts[i]))]
                    for i in range(len(self.fcts))]
            for fc in itertools.product(*Fc):
                F = [f[0] for f in fc] # le contexte
                c = prod([f[1] for f in fc]) # le poids du contexte
                Gf = self.STG(f = F, pre = 1)
                M += Counter({(u,v): c * Gf[u][v]['weight'] for u,v in Gf.edges})

        G = nx.DiGraph()
        G.add_weighted_edges_from([(a, b, M[(a,b)]) for a,b in M.keys()])

        return self.post_process_STG(G, layout, pre, plot_attrs, draw_labels)


    def post_process_STG(self, G, layout, pre, plot_attrs, draw_labels):
        """ Fonction annexe à STG() et STG_PBN().
        Traite le graphe G : calcule colorie ses attracteurs, et s'affiche."""

        # Détection des attracteurs : composantes fortement connexes terminales
        scc = nx.strongly_connected_components(G)
        scc = [x for x in scc]
        attractors = [x for x in scc
                      if x==set().union(*(G.neighbors(n) for n in x))]
        print('\nAttracteurs :')
        for a in attractors: print(a)

        # Coloriage des attracteurs dans le graphe
        # Rouge états stables, jaune attracteurs larges, gris états transients
        N = max([len(a) for a in attractors])
        cmap = cm.get_cmap('hot')
        mc, Mc = 0.25, 0.85
        if N==1:
            colors = [cmap(mc) for a in attractors]
        else:
            colors = [cmap(mc + (Mc - mc) * (len(a) - 1) / (N - 1))
                      for a in attractors]
        N_attr = len(attractors)
        color_map = []
        gray_rgb = (.8, .8, .8, 1)
        for x in G.nodes:
            flag = False
            for k in range(N_attr):
                if x in attractors[k]:
                    color_map.append(colors[k])
                    flag = True
                    break
            if not flag : color_map.append(gray_rgb)

        if pre==2:
            return G, attractors

        if plot_attrs:
            for k in range(len(attractors)):
                if 2 <= len(attractors[k]) <= 2**self.n-1:
                    # Pour chaque attracteur cyclique ou complexe
                    # Affichage du sous-graphe pour cet attracteur
                    plt.figure()
                    plt.title(f"Zoom sur l'attracteur n°{k}")
                    G2 = G.subgraph(attractors[k])
                    pos = layout(G2)
                    nx.draw(G2, pos = pos, with_labels = True, node_shape = "s",
                            edgecolors = 'k', font_size = 7, node_size = 500,
                            node_color = [colors[k]])
                    if (not self.sync) or self.pbn:
                        edge_labels2 = dict([((u,v,), round(d['weight'], 3))
                                            for u,v,d in G2.edges(data = True)])
                        nx.draw_networkx_edge_labels(G2, pos,
                                                     edge_labels = edge_labels2,
                                                     font_size = 7)

        # Affichage
        pos = layout(G)
        plt.figure()
        plt.title(f'{self.longtitle} - STG')
        nx.draw(G, pos = pos, with_labels=True, node_shape="s", edgecolors='k',
                font_size=7, node_size=500, node_color=color_map)
        if ((not self.sync) or self.pbn) and draw_labels:
            edge_labels = dict([((u,v,), round(d['weight'], 3))
                                for u,v,d in G.edges(data=True)])
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                         font_size=7)

        patchList = []
        colors = list(set(colors))
        patchList.append([mpatches.Patch(facecolor=c, label='Attracteur')
                                                            for c in colors])
        patchList.append((len(colors)-1)*[mpatches.Patch(color=(1,1,1,0),
                                                label='États transients')] \
                + [mpatches.Patch(color=gray_rgb, label='États transients')])

        plt.legend(handles = patchList,
                   labels = ['Attracteurs', 'États transients'],
                   handler_map = {list: HandlerTuple(None)})

        plt.show()


    def regulation_graph(self, layout = nx.spring_layout):
        """Affiche le graphe de régulation."""

        if len(self.regulation) == 0:
            print("Le réseau n'est pas régulé.")

        G = nx.DiGraph()
        plt.title(f'{self.title} - Graphe de régulation')

        if len(self.regulation) == 1:
            # graphe de régulation non-signé
            # on connaît les voisins
            for i in range(self.n):
                for j in self.regulation[0][i]:
                    G.add_edge(j, i, color = 'k', style = '->', size = 20)
            colors = 'k'
            styles = '-|>'

        if len(self.regulation) == 2:
            # graphe de régulation signé
            # on connaît les voisins et la nature des interactions (+/-)

            for i in range(self.n):
                for j0 in self.regulation[0][i]:
                    G.add_edge(j0, i, color = 'g', style = '->', size = 20)
                for j1 in self.regulation[1][i]:
                    G.add_edge(j1, i, color = 'r', style = '-[', size = 5)
            colors = [G[u][v]['color'] for u,v in G.edges]
            styles = [G[u][v]['style'] for u,v in G.edges]

        pos = layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        for edge in G.edges(data=True):
            color = edge[2]['color']
            style = edge[2]['style']
            size = edge[2]['size']
            nx.draw_networkx_edges(
                        G, pos, edgelist = [(edge[0], edge[1])], node_size = 700,
                        edge_color = color, width = 1.5, arrowstyle = style,
                        arrowsize = size, connectionstyle='arc3, rad=0.1')
        plt.show()


    def PBN_to_file(self, filename = None):
        """ Consigne les paramètres du modèle dans un fichier. """

        if filename == None :
            filename = self.title

        with open('output\\' + filename + '.pbn', 'w') as f:
            f.write(f'sync = {int(self.sync)}\n')
            f.write(f'p = {self.p}\n')
            f.write(f'q = {self.q}\n')
            s_card = ''
            if self.zeroes or self.ones:
                for i in range(self.n):
                    if i in self.zeroes : s_card += '0'
                    elif i in self.ones : s_card += '1'
                    else : s_card += '*'
                f.write(f'init = {s_card}\n')
            else:
                s_card = '*' * self.n
                f.write(f'init = {s_card}\n')
            f.write(f'indep = {int(self.indep)}\n\n')

            f.write('targets, factors\n')

            if len(self.regulation)==2 and (not self.pbn):
                # BN signé : fonction par défaut
                activators, inhibitors = self.regulation
                for i in range(self.n):
                    s_activ = " | ".join(['x'+str(j) for j in activators[i]])
                    s_inhib = " | ".join(['x'+str(j) for j in inhibitors[i]])

                    if not activators[i]:
                        f.write(f'x{i}, ! ({s_inhib})\n')
                    elif not inhibitors[i]:
                        f.write(f'x{i}, {s_activ}\n')
                    else:
                        f.write(f'x{i}, ({s_activ}) & ! ({s_inhib})\n')

            else:
                if len(self.regulation)==0:
                    # on ne connaît pas la régulation...
                    neighs = [[i for i in range(self.n)] for _ in range(self.n)]

                if len(self.regulation)==2:
                    # PBN à régulation signée
                    activators, inhibitors = self.regulation
                    neighs = [sorted(activators[i] + inhibitors[i])
                              for i in range(self.n)]
                else:
                    # BN ou PBN à régulation non-signée
                    neighs = self.regulation[0]

                neigh_states = []
                for i in range(self.n):
                    k = len(neighs[i])
                    L = list(map(list, itertools.product([0, 1], repeat = k)))
                    vois_i = []
                    for inds in L:
                        y = [0] * self.n
                        for j in range(k):
                            if inds[j]:
                                y[neighs[i][j]] = 1
                        vois_i.append(y)
                    neigh_states.append(vois_i)

                if self.indep:
                    for i in range(self.n):
                        for (fct, w) in zip(self.fcts[i], self.c[i]):
                            f.write(f'x{i}, ')
                            s = fct_to_clauseDNF(fct, neigh_states[i], neighs[i])
                            f.write(s)
                            f.write(f', {w}\n')
                        f.write('\n')

                else:
                    for j in range(self.m):
                        f.write(f'w = {self.c[j]}\n')
                        for i in range(self.n):
                            f.write(f'x{i}, ')
                            s = fct_to_clauseDNF(self.fcts[j][i], neigh_states[i], neighs[i])
                            f.write(f'{s}\n')
                        f.write('\n')
            f.close()



########### PARTIE 3/3 - SIMULATIONS D'EXEMPLES ET GÉNÉRATEURS ##########

def ex_toymodel():
    """Toy model présenté dans l'article d'AbouJaoudé2006."""
    f0 = lambda x: x[1] | x[3]
    f1 = lambda x: (x[0] & x[3]) | x[2]
    f2 = lambda x: (not x[0]) & (not x[3])
    f3 = lambda x: x[3]

    toymodel = PBN(title = 'Toy model',
                   n = 4,
                   indep = False,
                   f = [[f0, f1, f2, f3]],
                   c = [1],
                   sync = True,
                   p = 0,
                   q = 0)
    print(toymodel)
    toymodel.simulation(8, verb = True)
    toymodel.stationary_law()
    toymodel.STG()
    toymodel.copy_PBN(sync = False).STG()


def ex_shmulevich2001():
    """Toy model présenté dans l'article de Shmulevich2001."""
    f0_0 = lambda x: x[1] | x[2]
    f0_1 = lambda x: (x[1] | x[2]) & (not((not x[0]) & x[1] & x[2]))
    f1_0 = lambda x: (x[0] | x[1] | x[2]) & (x[0] | (not x[1]) | (not x[2])) \
                                          & ((not x[0]) | (not x[1]) | x[2])
    f2_0 = lambda x: (x[0] & (x[1] | x[2])) | ((not x[0]) & x[1] & x[2])
    f2_1 = lambda x: x[0] & x[1] & x[2]

    examplePBN = PBN(title = 'Shmulevich PBN',
                     n = 3,
                     indep = True,
                     f = [[f0_0, f0_1], [f1_0], [f2_0, f2_1]],
                     c = [[0.6, 0.4], [1], [0.5, 0.5]],
                     sync = True,
                     p = 0,
                     q = 1)
    print(examplePBN)
    # examplePBN.simulation(20, verb = False)
    examplePBN.STG_PBN()
    examplePBN.stationary_law()


def ex_mammaliancellcycle():
    """Modèle biologique étudié dans Fauré2006."""
    f0 = lambda x: x[0]

    f1 = lambda x: ((not x[4]) & (not x[9]) & (not x[0]) & (not x[3])) \
                 | (x[5] & (not x[9]) & (not x[0]))

    f2 = lambda x: ((not x[1]) & (not x[4]) & (not x[9])) \
                 | (x[5] & (not x[1]) & (not x[9]))

    f3 = lambda x: (x[2] & (not x[1]))

    f4 = lambda x: (x[2] & (not x[1]) & (not x[6]) & (not (x[7] & x[8]))) \
                 | (x[4] & (not x[1]) & (not x[6]) & (not (x[7] & x[8])))

    f5 = lambda x: ((not x[0]) & (not x[3]) & (not x[4]) & (not x[9])) \
                 | (x[5] & (not (x[3] & x[4])) & (not x[9]) &(not x[0]))

    f6 = lambda x: x[9]

    f7 = lambda x: ((not x[4]) & (not x[9])) | (x[6]) | (x[5] & (not x[9]))

    f8 = lambda x: (not x[7]) | (x[7] & x[8] & (x[6] | x[4] | x[9]))

    f9 = lambda x: (not x[6]) & (not x[7])

    cellcycle = PBN(title = 'Mammalian cell cycle',
                    n = 10,
                    indep = False,
                    f = [[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]],
                    c = [1],
                    sync = True,
                    p = 0,
                    q = 0)
    # cellcycle.simulation(8, verb = True)
    cellcycle.stationary_law(T = 200, N = 500, R = 500, show_all = False)

    prio_attrs = ["1000100000", "1000101111", "1000001010", "1000000010",
                  "1000000110", "1000000100", "1010000100", "1010100100",
                  "1001100100", "1011100000", "1010100000", "1000100100",
                  "1000001110", "1000100011", "1000101011", "1001100000",
                  "1011000100", "1011100100"]

    cellcycle.copy_PBN(sync = False)\
             .stationary_law(T = 200, N = 500, R = 500, show_all = False,
                             prio_attrs = prio_attrs)


##

def file_to_PBN(filename, title=None, sync = True, indep = True, p = 0, q = 1,
                zeroes = [], ones = []):
    """Crée un PBN à partir d'une description dans un fichier de forme :
    sync = (0 ou 1)
    p = (float entre 0 et 1)
    q = (float entre 0 et 1)
    init = (un élt de {0,1}^n avec possibilité de wildcards)
    indep = 0 ou 1

    targets, factors
    (si indep, for j<m[i] for i<n: )
        xi, fij, cij

    (si !indep, for j<m: )
        w = (float entre 0 et 1)
        (for i<n: )
        xi, fji

    Parameters
    ----------
    title : str
        Nom du PBN.
    filename : str
        Nom du fichier à charger. Chaque ligne est de la forme
        'nom_variable, fonction, proba_fonction'
    args :
        Attributs du PBN décrits plus haut.


    Returns
    -------
    PBN
    """

    if title == None:
        title = filename.split('\\')[-1]

    parameters, text = open(filename, 'r').read().split('factors\n')

    # Récupération des paramètres
    lines_param = parameters.split('\n')
    start = 0
    if lines_param[start][:4] == 'sync':
        sync = bool(int(lines_param[start].split('= ')[1]))
        start += 1
    if lines_param[start][0] == 'p':
        p = float(lines_param[start].split('= ')[1])
        start += 1
    if lines_param[start][0] == 'q':
        q = float(lines_param[start].split('= ')[1])
        start += 1
    if lines_param[start][:4] == 'init':
        zeroes, ones = [], []
        card = lines_param[start].split('= ')[1]
        for i in range(len(card)):
            if card[i]=='0': zeroes.append(i)
            if card[i]=='1': ones.append(i)
        start += 1
    if lines_param[start][:5] == 'indep':
        indep = bool(int(lines_param[start].split('= ')[1]))
        start += 1

    lines = text.split('\n')
    targets = dict()
    # Remplissage du dictionnaire targets indiçant les variables
    index = 0
    for line in lines:
        if line  != '' and line[0] !='\n':
            splits = line.split(', ')
            if len(splits) > 1:
                target = splits[0]
                if target not in targets.keys():
                    targets[target] = index
                    index += 1

    # Parsing des fonctions et de leurs probabilités
    if indep:
        functs = [[] for _ in range(len(targets))]
        cs = [[] for _ in range(len(targets))]
        for line in lines:
            if line != '' and line[0] !='\n':
                # Lecture de la ligne décrivant la fonction
                target, factors, c = line.split(', ')
                factors = clause_to_fct(factors, targets)

                # Conversion de la clause en fonction booléenne
                x = targets[target]
                functs[x].append(eval('lambda x: ' + factors))
                cs[x].append(float(c))

    else:
        functs, cs = [], []
        start_flag = True
        context = [0 for _ in range(len(targets))]
        for line in lines:
            if line != '' and line[0] !='\n':
                if line[:4] == 'w = ':
                    cs.append(float(line[4:]))
                    if not start_flag:
                        functs.append(context)
                    context = [0 for _ in range(len(targets))]
                else:
                    start_flag = False
                    target, factors = line.split(', ')
                    factors = clause_to_fct(factors, targets)

                    x = targets[target]
                    context[x] = eval('lambda x: ' + factors)

        functs.append(context)
        if cs == []:
            cs = [1]

    return PBN(title = title,
               n = len(targets),
               indep = indep,
               f = functs,
               c = cs,
               sync = sync,
               p = p,
               q = q,
               zeroes = zeroes, ones = ones)



def test_claudine():
    """Calcule la prévalence des attracteurs Th0 Th1 Th2 dans le modèle
    Th_23, ainsi que dans ses extensions comportant les voisines des
    fonctions de référence dans le diagramme de Hasse.
    Cf. Cury2019, Mendoza2006."""

    for name in ['Table0_p08.bnet',
                 'Table1A_fAll_d1_p08.bnet',
                 'Table1B_fAll_d1_p08_siblings.bnet',
                 'Table1C_fGATA3_p08.bnet',
                 'Table1D_fTbet_p08.bnet',
                 'Table1E_fIL4_p08.bnet',
                 'Table1F_fIL4R_p08.bnet']:

        if name == 'Table0_p08.bnet':
            filename = 'Experiments_Th_model\Table0_p08.bnet'
            zeroes, ones = [-1, -2, -3, -4], []
        else:
            filename = "Experiments_Th_model\simul_prob_original\\" + name
            zeroes, ones = [i for i in range(23) if i != 2], [2]

        def approach_attrs(x):
            if x[0] == '1':
                return '10001100110000010100000', 2
            if x[18] == '1':
                return '00110000000001000010000', 1
            return '00000000000000000000000', 0

        pbn_claudine = file_to_PBN(filename = filename,
                                   zeroes = zeroes, ones = ones, sync = True)
        pbn_claudine.stationary_law2(approach_attrs = approach_attrs,
                                attr_colors = [(0,255,0), (255,0,0), (0,0,255)],
                                attr_names = ['Th0', 'Th1', 'Th2'],
                                T = 20, R = 1000)


##

def generateBN(n, k, sync, v = False, f = False, p = 0):
    """Construit un BN dont chacun des n nœuds est régulé par k voisins ou moins.

    Parameters
    ----------
    n : int
        Nombre de bits décrivant un état du BN.
    k : int
        Nombre de voisins de chaque bit/gène dans le graphe de régulation.
    v : bool
        Si False, tous les gènes ont k voisins.
        Si True, le nombre de voisins de chaque gène est tiré entre 0 et k.
    f : bool
        Si False, les fonctions de régulation sont tirées au hasard.
        Si True, on tire des vecteurs de régulation +/-, et la fonction de
        régulation est celle par défaut (cf. Mendoza2006 équation 1).
    args (sync, p) :
        Attributs du PBN décrits plus haut.

    Returns
    -------
    PBN
    """
    #TODO : option canalyzing functions (x_i=u => f(x)=y) : f(x) = x_i ^/v h(x) ?

    # Détermination du nombre de voisins de chaque gène
    if v:
        n_vois = [random.randint(0,k) for _ in range(n)]
    else:
        n_vois = [k for _ in range(n)]

    # Sélection des voisins de chaque gène
    flag = True
    while flag: # on fait que le graphe de régulation soit connexe
        nodes = [i for i in range(n)]
        neighs = [sorted(random.sample(nodes, n_vois[i])) for i in range(n)]
        edges = [(j, i) for i in nodes for j in neighs[i]]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        flag = not nx.is_connected(G)

    # for i in nodes:
    #     print('%s -> %i' %(neighs[i], i))
    # print()

    # Pour chaque gène i de voisins i_1 ... i_ki,
    if f:
        # Détermination des régulations négatives et positives
        p_neg = .5 # TODO: une proportion de régulations négatives ?
        signs = [[random.random() < p_neg for _ in range(n_vois[i])]
                  for i in range(n)]
        activators = [[neighs[i][j] for j in range(n_vois[i])
                       if signs[i][j]]
                       for i in range(n)]
        inhibitors = [[neighs[i][j] for j in range(n_vois[i])
                       if (not signs[i][j])]
                       for i in range(n)]

        # Calcul déterministe de la fonction par défaut
        functs = []
        fun_I = lambda i: lambda x: not(any([x[k] for k in inhibitors[i]]))
        fun_A = lambda i: lambda x: any([x[k] for k in activators[i]])
        fun_IA = lambda i: lambda x: any([x[k] for k in activators[i]]) \
                                and not(any([x[k] for k in inhibitors[i]]))
        for i in range(n):
            if not activators[i]:
                functs.append(fun_I(i))
            elif not inhibitors[i]:
                functs.append(fun_A(i))
            else:
                functs.append(fun_IA(i))
        regulation = (activators, inhibitors)
        functs = [functs]

    else:
        # Une table de vérité aléatoire pour la fonction x_i1 ... x_ik -> x_i
        T = [np.random.choice([0, 1], size=(2**n_vois[i],)) for i in range(n)]
        fun = lambda i: lambda x: \
                        T[i][sum([x[neighs[i][j]] * 2**j for j in range(n_vois[i])])]
        functs = [[fun(i) for i in range(n)]]
        regulation = (neighs,)

    return PBN(title = f'Synthetic ({n},{k})-BN',
               n = n,
               indep = False,
               f = functs,
               c = [1],
               sync = sync,
               p = p,
               q = 0,
               regulation = regulation)



def generatePBN(BN, i_modifs, p_ref, dist, q=1):
    """Construit un PBN à partir des fonctions de référence d'un BN, étendues
    parmi leurs voisines dans le diagramme de Hasse.

    Parameters
    ----------
    BN : PBN
        Un réseau booléen, n'ayant donc qu'un seul contexte.
    i_modifs : list
        Liste des indices des bits dont on veut étendre la fonction.
    p_ref : int
        Probabilité associée à la fonction de référence.
    dist :
        Distance maximale à explorer dans le diagramme de Hasse.

    Returns
    -------
    PBN
    """

    fcts_pbn = BN.fcts.copy()
    c_pbn = BN.c.copy()

    for i in i_modifs:
        # récupérer dans f_voisines les voisins de f[i] à distance dist
        #TODO

        s = len(f_voisines)
        fcts_pbn[i] = [fcts_pbn[i]] + f_voisines
        c_pbn[i] = [p_ref] + [(1-p_ref)/s] * s

    return BN.copy_PBN(title = title + 'extended', indep = True,
                       f = fcts_pbn, c = c_pbn, q=q)



def generate_Random_PBN(m, n, k, indep, sync = True, p = 0, q = .1):
    """Construit un PBN dont chacun des n nœuds est régulé par k voisins
        et m fonctions équiprobables à tables de vérité générées aléatoirement.

    Parameters
    ----------
    m : int, list
        Nombre de fonctions de régulation associées à chaque gène.
    n : int
        Nombre de bits décrivant un état du PBN.
    k : int
        Nombre de voisins de chaque bit/gène dans le graphe de régulation.
    args :
        Attributs du PBN décrits plus haut.

    Returns
    -------
    PBN
    """

    if k > n:
        raise ValueError("Merci d'entrer un k<n.")

    nodes = [i for i in range(n)]
    # Sélection de k voisins pour chaque gène
    neighs = [sorted(random.sample(nodes, k)) for _ in range(n)]
    # for i in nodes:
    #     print('%s -> %i' %(neighs[i], i))

    if indep:
        # Génération d'un F = F1 x ... x Fn,
        # avec F_i les fonctions de transition possibles du i-ème gène.
        fun = lambda i,j: lambda x: \
                        T[i][j][sum([x[neighs[i][q]] * 2**q for q in range(k)])]

        if type(m)==int: # chaque gène a le même nombre m de fonctions
            T = [[np.random.choice([0, 1], size=(2**k,)) for _ in range(m)]
                                                         for _ in range(n)]
            functs = [[fun(i,j) for j in range(m)] for i in range(n)]
            # chaque choix de fonction est équiprobable
            c = [[1 / m for _ in range(m)] for _ in range(n)]

        else: # un noeud i a m[i] fonctions
            if len(m)!=n:
                raise ValueError("La liste m doit être de longueur n.")
            T = [[np.random.choice([0, 1], size=(2**k,)) for _ in range(m[i])]
                                                            for i in range(n)]
            functs = [[fun(i,j) for j in range(m[i])] for i in range(n)]
            c = [[1 / (m[i]) for _ in range(m[i])] for i in range(n)]

    else: # Génération d'un F = [f_1, ..., f_m], avec f_i un contexte.
        if type(m)==list:
            raise ValueError("Merci d'entrer un m entier ou indep=True.")
        fun = lambda i,j: lambda x: \
                        T[i][j][sum([x[neighs[j][q]] * 2**q for q in range(k)])]
        T = [[np.random.choice([0, 1], size=(2**k,)) for _ in range(n)]
                                                     for _ in range(m)]
        functs = [[fun(i,j) for j in range(n)] for i in range(m)]
        c = [1 / m for _ in range(m)]

    return PBN(title = f'Synthetic ({m},{n},{k})-PBN',
               n = n,
               indep = indep,
               f = functs,
               c = c,
               sync = sync,
               p = p,
               q = q,
               regulation = (neighs,))


def test_syntheticBN(n, k):

    gs = [generateBN(n, k, sync = True, v = False, f = False, p = 0),
          generateBN(n, k, sync = True, v = False, f = True, p = 0)]

    for g in gs:
        print(g)
        g.regulation_graph()
        gasync = g.copy_PBN(sync = False)
        if n <= 6: g.STG()
        g.stationary_law(show_all = False)
        if n <= 6: gasync.STG(layout = nx.spring_layout)
        gasync.stationary_law(T = 200, N = 500, R = 500, show_all = False)


def test_syntheticPBN(m, n, k, indep):
    gs = [generate_Random_PBN([m,1,m,1], n, k, indep = True, sync = True,
                                                             p = 0, q = .1),
          generate_Random_PBN(m, n, k, indep = indep, sync = True,
                                                      p = 0, q = .1)]

    for g in gs:
        print(g)
        g.STG_PBN()
        g.copy_PBN(sync = False).STG_PBN()
        # g.simulation(50, verb = True)


def test_filesPBN():

    model = file_to_PBN('output\Table1A_fAll_d1_p08_newsyntax.pbn')
    print(model)

    gs = [generateBN(5, 3, sync = True, v = False, f = False, p = 0), # reg
          generateBN(8, 4, sync = True, v = False, f = True, p = 0), # reg signé
          generate_Random_PBN([2,3,2,1,1], 5, 3, indep = True), # PBN indep
          generate_Random_PBN(2, 5, 3, indep = False)] # PBN non-indep
    for g in gs:
        print(g)
        g.regulation_graph()
        g.PBN_to_file()
        g2 = file_to_PBN(filename = 'output\\' + g.title + '.pbn')
        print(g2)
        g.regulation_graph()




##

def tests():

    # Modèle simple à 4 noeuds : STG synchrone, STG asynchrone, loi synchrone
    ex_toymodel()

    # PBN à tirages indépendants : STG synchrone, loi synchrone
    ex_shmulevich2001()

    # Modèle biologique à 10 noeuds : loi synchrone,
                            # loi asynchrone avec affichage des attrs synchrones
    ex_mammaliancellcycle()

    # 6 simulations en synchrone et asynchrone)
                        # des fonctions voisines dans le diagramme de Hasse
    # test_claudine()

    # BN synthétique à n noeuds et k voisins : STG, loi
    test_syntheticBN(4,2)

    # PBN synthétique à n noeuds et m contextes : STG
    test_syntheticPBN(2, 4, 1, indep = False)

    # Divers exemples de conversion entre objets PBN et fichiers .pbn
    test_filesPBN()


tests()



#TODO : interaction console
#TODO : sauvegarder un PBN et des prints console dans un fichier texte

#TODO : functionhood.getFormulaChildren/Parents, dans GeneratePBN avec Py4J
#TODO : autre version du code avec appels Py4J sur le getAttractors de BioLQM ?


#TODO : rédiger la démo sur |BOA| merde
#TODO : en non-déterministe (async et PBN), le seuil T pour assez d'attracteurs ?

#TODO : variation des proportions de test_claudine() avec 0<q<1, voire p>0


#TODO : approximation et calcul du MFPT
#TODO : jolie animation de simulation ?