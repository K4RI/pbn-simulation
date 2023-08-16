# -*- coding: utf-8 -*-
# Created on Thu Jul 6 2023
# @author: K4RI

from collections import Counter
from sympy import *
import itertools
from matplotlib import cm
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from py4j.java_gateway import JavaGateway
import random


########### PARTIE 1 / 3 - UTILITY FUNCTIONS ##########

def init_vars(n: int):
    """Initialise les variables x0, ..., x_{n-1}."""

    varnames = []
    for i in range(n):
        globals().__setitem__(f'x{i}', Symbol(f'x{i}'))
        varnames.append(Symbol(f'x{i}'))
    return varnames


def Hamming(x: list, y: list) -> int:
    """Renvoie la distance de Hamming entre deux mots binaires."""
    return len([i for i in range(len(x)) if x[i] != y[i]])


def prod(L: list) -> int:
    """Renvoie le produit des termes d'une liste."""
    if not L: return 1
    return L[0] * prod(L[1:])


def bit_list(n: int) -> list:
    """Renvoie la liste des listes binaires de taille n."""

    return list(map(list, itertools.product([0, 1], repeat = n)))


def bit_list_str(n: int) -> list:
    """Renvoie la liste des mots binaires de taille n."""

    return list(map(lambda t : ''.join(map(str, t)),
                            itertools.product([0, 1],repeat=n)))



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
    indep : bool
        Indépendance entre les choix de fonctions de transition
        pour différents bits.
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
    varnames : list
        Liste des variables.

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


    def __init__(self, n, indep, f, c, sync, p, q, title = '', varnames = [],
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

        # # Le nom de chaque gène
        self.varnames = varnames

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

        if self.varnames:
            var_str = list(map(str, self.varnames))
            s_varnames = '\n' + ' - '.join(var_str)
        else:
            s_varnames = '\n'

        if self.indep:
            s_ind = 'à tirages indépendants'
            s_fcts = ''
            for i in range(self.n):
                if self.varnames:
                    s_fcts += 'x%i (%s) - %i fonctions possibles\n' \
                              %(i, self.varnames[i], len(self.fcts[i]))
                else:
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
            s_ctxtbls = self.truth_table2(self.fcts)
            if self.pbn:
                s_ctxtbl = self.truth_table1(self.currentfct_vector)

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
%(self.title, s_type, s_sync, s_ind, self.n, self.p, self.q,
s_fcts, s_ctxtbls, str(self.c), str(self.x), s_ctxt, s_ctxtbl)


    def truth_table1(self, F: list) -> str :
        """Affiche la table de vérité pour des fonctions booléennes.

        Parameters
        ----------
        F : list
            Liste de fonctions booléennes.
        Returns
        -------
        str
            La chaîne de caractères représentant la table de vérité.
        """

        L = bit_list(self.n)
        s = ''
        for x in L :
            s += '%s || %s \n' %(''.join([str(u) for u in x]),
                                ' '.join([str(self.evalf(x,f)) for f in F]))
        return s


    def truth_table2(self, F: list) -> str:
        """Identique à la précédente, mais F est séparée en sous-listes."""

        L = bit_list(self.n)
        s = ''
        for x in L :
            s += '%s || %s \n' %(''.join([str(u) for u in x]),
                                ' - '.join([' '.join([str(self.evalf(x,f))
                                            for f in ff]) for ff in F]))
        return s


    def copy_PBN(self, title = None, n = None, indep = None, fcts = None,
                 c = None, sync = None, p = None, q = None, varnames = None,
                 regulation = None):
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
        if varnames == None: varnames = self.varnames
        if regulation == None: regulation = self.regulation

        return PBN(title = title,
                   n = n,
                   indep = indep,
                   f = fcts,
                   c = c,
                   sync = sync,
                   p = p,
                   q = q,
                   varnames = varnames,
                   regulation = regulation)


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


    def evalf(self, x, f):
        """Évalue la fonction f au vecteur x."""

        return int(f.subs({self.varnames[i]:x[i] for i in range(self.n)}) == True)


    def succ(self, x, f):
        """Renvoie la liste des successeurs d'un x par un contexte f."""

        xs = [[self.evalf(x, f[i]) for i in range(self.n)]]

        # Cas synchrone : tous les bits sont mis à jour
        if self.sync:
            return xs

        # Cas asynchrone : toutes les possibilités de mise à jour de chaque bit
        else:
            i_modifs = [i for i in range(self.n) if x[i] != xs[0][i]]
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
            bins = bit_list_str(self.n)
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
        nodes_str = bit_list_str(self.n)
        G.add_nodes_from(nodes_str)
        nodes = bit_list(self.n)

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

            if self.varnames:
                varnames = self.varnames
            else:
                varnames = init_vars(self.n)

            if self.indep: # indépendant : gène par gène
                for i in range(self.n):
                    for (ff, w) in zip(self.fcts[i], self.c[i]):
                        ff = str(ff).replace('~', '!')
                        f.write(f'{varnames[i]}, {ff}, {w}\n')
                    f.write('\n')

            else: # non-indépendant : contexte par contexte
                for j in range(self.m):
                    if self.m > 1: # si c'est un PBN
                        f.write(f'w = {self.c[j]}\n')
                    for i in range(self.n):
                        ff = str(self.fcts[j][i]).replace('~', '!')
                        f.write(f'{varnames[i]}, {ff}\n')
                    f.write('\n')
            f.close()




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
    varnames = []
    # Remplissage de la liste indiçant les symboles
    index = 0
    for line in lines:
        if line  != '' and line[0] !='\n':
            splits = line.split(', ')
            if len(splits) > 1: # on lit bien une ligne x, f
                target = splits[0]

                if Symbol(target) not in varnames:
                    globals().__setitem__(target, Symbol(target))
                    varnames.append(Symbol(target))

    n = len(varnames)
    neighs = [set() for _ in range(n)]
    # Parsing des fonctions et de leurs probabilités
    if indep:
        functs = [[] for _ in range(n)]
        cs = [[] for _ in range(n)]
        for line in lines:
            if line != '' and line[0] !='\n':
                # Lecture de la ligne décrivant la fonction
                target, functions, c = line.replace('!', '~').split(', ')
                functions = eval(functions)
                vois = [varnames.index(j) for j in functions.free_symbols]
                i = varnames.index(Symbol(target))
                neighs[i].update(vois)

                # Conversion de la clause en fonction booléenne
                functs[i].append(functions)
                cs[i].append(float(c))

    else:
        functs, cs = [], []
        start_flag = True
        context = [0 for _ in range(n)]
        for line in lines:
            if line != '' and line[0] !='\n':
                if line[:4] == 'w = ':
                    cs.append(float(line[4:]))
                    if not start_flag:
                        functs.append(context)
                    context = [0 for _ in range(n)]
                else:
                    start_flag = False
                    target, functions = line.replace('!', '~').split(', ')
                    functions = eval(functions)
                    vois = [varnames.index(j) for j in functions.free_symbols]
                    i = varnames.index(Symbol(target))
                    neighs[i].update(vois)
                    context[i] = functions

        functs.append(context)
        if cs == []:
            cs = [1]

    neighs = [sorted(neigh) for neigh in neighs]

    return PBN(title = title,
               n = n,
               indep = indep,
               f = functs,
               c = cs,
               sync = sync,
               p = p,
               q = q,
               varnames = varnames,
               regulation = (neighs,),
               zeroes = zeroes, ones = ones)



########### PARTIE 3/3 - GÉNÉRATEURS ##########

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

    varnames = init_vars(n)

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

    functs = []
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
        for i in range(n):
            s_inhib = '~(' + ' | '.join([f'x{j}' for j in inhibitors[i]]) + ')'
            s_activ = '(' + ' | '.join([f'x{j}' for j in activators[i]]) + ')'
            if not activators[i]:
                functs.append(eval(s_inhib))
            elif not inhibitors[i]:
                functs.append(eval(s_activ))
            else:
                functs.append(eval(s_activ + ' & ' + s_inhib))
        regulation = (activators, inhibitors)

    else:
        # Une table de vérité aléatoire pour la fonction x_i1 ... x_ik -> x_i
        for i in range(n):
            Lk = bit_list(n_vois[i])
            symbols = [varnames[j] for j in neighs[i]]
            minterms = []
            for x in Lk:
                if random.random() < .5:
                    minterms.append(x)
            functs.append(POSform(symbols, minterms))
        regulation = (neighs,)
    functs = [functs]
    return PBN(title = f'Synthetic ({n},{k})-BN',
               n = n,
               indep = False,
               f = functs,
               c = [1],
               sync = sync,
               p = p,
               q = 0,
               regulation = regulation,
               varnames = varnames)


def str_signed(v, f):
    """Renvoie v si v est une variable activatrice de f,
       et ~v si elle est inhibitrice.
       Annexe à la fonction suivante."""

    fs = str(f)
    i = fs.index(str(v))
    if (i==0) or (fs[i-1] != '~'): # variable activatrice
        return str(v)
    else: # variable inhibitrice
        return '~' + str(v)


def voisines_direct(F, cp):
    """Calcule l'union des voisins directs d'une liste de fonctions booléennes.
       Annexe à la fonction suivante.

    Parameters
    ----------
    F : list
        Liste de fonctions booléennes monotones.
    cp :
        Si cp == 'c' : on calcule les enfants.
        Si cp == 'p' : on calcule les parents.

    Returns
    -------
    set
        Ensemble contenant chaque parent/enfant de chaque fonction de F.
    """

    s_voisinescp = set()
    for f in F :
        # On traduit la fonction monotone en l'ensemble de BitSets correspondant
        fd = to_dnf(f, simplify = True)
        symbs = list(fd.free_symbols)
        formula_L = ([sorted([1+symbs.index(l) for l in list(eval(clause).free_symbols)])
                                        for clause in str(fd).split('|')])
        formula = str(formula_L).replace('[', '{').replace(']', '}').replace(' ', '')
        symbs = [str_signed(v, fd) for v in symbs]

        # On appelle le script Java 'functionhood'
        gateway = JavaGateway()
        hd = gateway.entry_point.initHasseDiagram(len(symbs))

        if cp == 'c':
            a = gateway.entry_point.getFormulaChildrenfromStr(formula, False)
        else:
            a = gateway.entry_point.getFormulaParentsfromStr(formula, False)

        # On traduit la sortie de functionhood en une liste de fonctions booléennes
        s_voisines = eval(str(a).replace('{', '[').replace('}', ']'))
        f_voisine = [[[symbs[j-1] for j in clause] for clause in s] for s in s_voisines]
        f_voisinescp = [eval(' | '.join(['(' + ' & '.join(clause) + ')' for clause in f])) for f in f_voisine]
        s_voisinescp.update(f_voisinescp)
    return s_voisinescp


def voisines(f, dist):
    """Calcule les fonctions voisines d'une fonction monotone
       dans le diagramme de Hasse.
       Annexe à la fonction suivante.

    Parameters
    ----------
    f : sympy.Boolean
        Une fonction booléenne monotone.
    dist :
        Distance maximale à explorer dans le diagramme de Hasse.

    Returns
    -------
    list
        Liste des listes de voisins à distance k, pour k de 1 à dist.
    """

    for v in f.free_symbols:
        globals().__setitem__(str(v), v)

    fd = to_dnf(f, simplify = True)
    to_dnf_set = lambda s: [to_dnf(x, simplify = True) for x in list(s)]

    # parents & enfants
    f_parents = voisines_direct({f}, 'p')
    f_enfants = voisines_direct({f}, 'c')
    f_voisines = [to_dnf_set(f_parents.union(f_enfants))]

    if dist >= 2: # siblings
        f_siblings = voisines_direct(f_parents, 'c').union(voisines_direct(f_enfants, 'p'))
        f_siblings.remove(fd)
        f_voisines.append(to_dnf_set(f_siblings))

        if dist == 3: # grands-parents & petits-enfants
            f_gppe = voisines_direct(f_parents, 'p').union(voisines_direct(f_enfants, 'c'))
            f_voisines.append(to_dnf_set(f_gppe))

    return f_voisines


def generate_Extended_PBN(BN, i_modifs = None, p_ref = 0.8, dist = 1, part = 'poly', q=1):
    """Construit un PBN à partir des fonctions de référence d'un BN, étendues
    parmi leurs voisines dans le diagramme de Hasse.

    Parameters
    ----------
    BN : PBN
        Un réseau booléen, n'ayant donc qu'un seul contexte.
    i_modifs : list
        Liste des indices des bits dont on veut étendre les fonctions.
    p_ref : int
        Probabilité associée à la fonction de référence.
    dist :
        Distance maximale à explorer dans le diagramme de Hasse.
        Si dist == 1 : explorer les parents + les enfants des fonctions.
        Si dist == 2 : précédents + les siblings.
        Si dist == 3 : précédents + les grands-parents + les petits-enfants.
    part :
        Modèle de partitionnement des voisins en fonction de la distance.
        Si part == 'poly' : les voisins à distance k ont un poids r**k
        Si part == 'div' : les voisins à distance k ont un poids r/k
        Si part == 'equal' : les voisins ont tous le même poids

    Returns
    -------
    PBN
    """

    if i_modifs == None:
        i_modifs = [i for i in range(BN.n)]

    fcts_pbn = [[] for _ in range(BN.n)]
    c_pbn = [[] for _ in range(BN.n)]

    for i in range(BN.n):
        if i in i_modifs:
            f_voisines = voisines(BN.fcts[0][i], dist)
            fcts_pbn[i] = [BN.fcts[0][i]] + flatten(f_voisines)

            ns = [len(v) for v in f_voisines]
            if any(ns):
                c_pbn[i] = [p_ref]
                if part == 'poly':
                    # On détermine le r tel que les voisins à distance k auront un poids r**k
                    x = var('x')
                    sols = solve(Eq(sum([ns[i]*x**(i+1) for i in range(dist)]), 1),x)
                    r = max(list(filter(lambda x: 'I' not in str(x), sols)))
                    for k in range(dist):
                        c_pbn[i] += [round((1-p_ref) * r**(k + 1), 6)] * ns[k]

                if part == 'div':
                    # On détermine le r tel que les voisins à distance k auront un poids r/k
                    r = 1/(sum([ns[k] / (k + 1) for k in range(dist)]))
                    for k in range(dist):
                        c_pbn[i] += [round((1-p_ref) * r/(k + 1), 6)] * ns[k]

                if part == 'equal':
                    # On détermine le r tel que tous les voisins ont le même poids
                    r = 1/(sum(ns))
                    c_pbn[i] += [round((1-p_ref) * r, 6)] * sum(ns)

            else:
                c_pbn[i] = [1]

        else:
            fcts_pbn[i] = [BN.fcts[0][i]]
            c_pbn[i] = [1]


    if len(i_modifs) == BN.n:
        mods = 'allv'
    else:
        mods = ''.join(map(str, i_modifs))
    return BN.copy_PBN(title = f'{BN.title} extended_{mods}_{dist}',
                       indep = True, fcts = fcts_pbn, c = c_pbn, q = q)



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

    if k > n or k<=0:
        raise ValueError("Merci d'entrer un 0 < k < n.")

    # Sélection des voisins de chaque gène
    flag = True
    while flag: # on fait que le graphe de régulation soit connexe
        nodes = [i for i in range(n)]
        neighs = [sorted(random.sample(nodes, k)) for i in range(n)]
        edges = [(j, i) for i in nodes for j in neighs[i]]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        flag = not nx.is_connected(G)

    varnames = init_vars(n)

    Lk = bit_list(k)
    if indep:
        # Génération d'un F = F1 x ... x Fn,
        # avec F_i les fonctions de transition possibles du i-ème gène.
        functs = [[] for _ in range(n)]
        if type(m)==int: # chaque gène a le même nombre m de fonctions
            for i in range(n):
                symbols = [varnames[j] for j in neighs[i]]
                for j in range(m):
                    minterms = []
                    for x in Lk:
                        if random.random() < .5:
                            minterms.append(x)
                    functs[i].append(POSform(symbols, minterms))
            c = [[1 / m for _ in range(m)] for _ in range(n)]

        else: # un noeud i a m[i] fonctions
            if len(m)!=n:
                raise ValueError("La liste m doit être de longueur n.")
            for i in range(n):
                symbols = [varnames[j] for j in neighs[i]]
                for j in range(m[i]):
                    minterms = []
                    for x in Lk:
                        if random.random() < .5:
                            minterms.append(x)
                    functs[i].append(POSform(symbols, minterms))
            c = [[1 / (m[i]) for _ in range(m[i])] for i in range(n)]

    else: # Génération d'un F = [f_1, ..., f_m], avec f_i un contexte.
        if type(m)==list:
            raise ValueError("Merci d'entrer un m entier ou indep=True.")
        functs = [[] for _ in range(m)]
        for j in range(m):
            for i in range(n):
                symbols = [varnames[j] for j in neighs[i]]
                minterms = []
                for x in Lk:
                    if random.random() < .5:
                        minterms.append(x)
                functs[j].append(POSform(symbols, minterms))
        c = [1 / m for _ in range(m)]

    return PBN(title = f'Synthetic ({m},{n},{k})-PBN',
               n = n,
               indep = indep,
               f = functs,
               c = c,
               sync = sync,
               p = p,
               q = q,
               regulation = (neighs,),
               varnames = varnames)



#TODO : autre version du code avec appels Py4J sur le getAttractors de BioLQM ?

#TODO : rédiger la démo sur |BOA| merde
#TODO : en non-déterministe (async et PBN), le seuil T pour assez d'attracteurs ?

#TODO : variation des proportions de test_claudine() avec 0<q<1, voire p>0


#TODO : approximation et calcul du MFPT
#TODO : jolie animation de simulation ?