from PBN_simulation import *


# ## I : EXEMPLES DE MODÈLES
# Les fichiers les décrivant sont dans 'examples/' ou dans 'Experiments_Th_model/'

def ex_toymodel():
    """Toy model présenté dans l'article d'AbouJaoudé2006."""

    toymodel = file_to_PBN('examples\\toymodel.pbn')
    print(toymodel)
    # toymodel.simulation(10, verb = True)
    # toymodel.stationary_law(T = 100, N = 2000, R = 2000, show_all=True)
    toymodel.stationary_law(T = 100, N = 200, R = 200, show_all=True)
    # toymodel.STG()
    # toymodel.copy_PBN(sync = False).STG()



def ex_shmulevich2001():

    examplePBN = file_to_PBN('examples\\examplePBN.pbn')
    print(examplePBN)
    examplePBN.simulation(10, verb = False)
    examplePBN.STG_PBN()
    examplePBN.stationary_law(T = 100, N = 25, R = 200)



def ex_mammaliancellcycle():
    """Modèle biologique étudié dans Fauré2006."""

    cellcycle = file_to_PBN('examples\\cellcycle.pbn')
    # cellcycle.simulation(8, verb = True)
    cellcycle.stationary_law(T=200, N=1000, R=1000, show_all = False)
    cellcycle.stationary_law3(T = 150, N = 200, R = 200)
    cellcycle.copy_PBN(sync = False).stationary_law3(T = 150, N = 400, R = 200)

    prio_attrs = ["1000100000", "1000101111", "1000001010", "1000000010",
                  "1000000110", "1000000100", "1010000100", "1010100100",
                  "1001100100", "1011100000", "1010100000", "1000100100",
                  "1000001110", "1000100011", "1000101011", "1001100000",
                  "1011000100", "1011100100"]

    cellcycle.copy_PBN(sync = False)\
             .stationary_law(T=200, N=500, R=500, show_all = False,
                             prio_attrs = prio_attrs)


def ex_zhou():
    """Toy model présenté dans l'article de zhou2016."""

    zhou = file_to_PBN('examples\\zhou.pbn')
    print(zhou)
    # zhou.simulation(10, verb = True)
    zhou.stationary_law(T = 100, N = 200, R = 1000, show_all=True)
    # zhou.STG()
    # zhou.copy_PBN(sync = False).STG()
    #
    # zhoupbn = file_to_PBN('examples\\zhoupbn.pbn')
    # zhoupbn.STG_PBN()
    # zhoupbn.STG_allPBN()


def test_claudine(q=1, inds=[0,1,2,3,4,5,6]):
    """Calcule la prévalence des attracteurs Th0 Th1 Th2 dans le modèle
    Th_23, ainsi que dans ses extensions comportant les voisines des
    fonctions de référence dans le diagramme de Hasse.
    Cf. Cury2019, Mendoza2006."""

    models = ['Table0_p08.bnet',
              'Table1A_fAll_d1_p08.bnet',
              'Table1B_fAll_d1_p08_siblings.bnet',
              'Table1C_fGATA3_p08.bnet',
              'Table1D_fTbet_p08.bnet',
              'Table1E_fIL4_p08.bnet',
              'Table1F_fIL4R_p08.bnet'
             ]

    for i in inds:
        name = models[i]

        if name == 'Table0_p08.bnet':
            filename = 'Experiments_Th_model\Table0_p08.bnet'

            # zeroes, ones = [-1, -2, -3, -4], []
            # Mendoza page 15 : "IFN-β, IL-12, IL-18 and TCR do not have inputs [...], treated as constants [...] having a value of 0"

        else:
            filename = "Experiments_Th_model\simul_prob_original\\" + name
            # zeroes, ones = [i for i in range(23) if i != 2], [2]
            # P-O 6.2 : "Starting from an initial state, in which all the components are inactive but IFNg"
            # dans Mendoza fig 3, activer IFNg fait bouger de Th0 à Th1

        zeroes, ones = [i for i in range(23) if i != 2], [2]

        def approach_attrs(x):
            if x[0] == '1':
                return '10001100110000010100000', 2
            if x[18] == '1':
                return '00110000000001000010000', 1
            return '00000000000000000000000', 0

        pbn_claudine = file_to_PBN(filename = filename)
        pbn_claudine.zeroes = zeroes
        pbn_claudine.ones = ones

        pbn_claudine.stationary_law(T = 1000, N = 500, R = 100)
        pbn_claudine.copy_PBN(sync=False).stationary_law(T = 1000, N = 500, R = 100)

        pbn_claudine.stationary_law2(approach_attrs = approach_attrs,
                                attr_colors = [(0,255,0), (255,0,0), (0,0,255)],
                                attr_names = ['Th0', 'Th1', 'Th2'],
                                T = 2000, R = 100)



# ## II : TESTS DES FONCTIONNALITÉS

def test_syntheticBN(n, k):

    gs = [generateBN(n, k, sync = True, v = False, f = False, p = 0),
          generateBN(n, k, sync = True, v = False, f = True, p = 0)]

    for g in gs:
        print(g)
        g.regulation_graph()
        gasync = g.copy_PBN(sync = False)
        if n <= 6: g.STG()
        g.stationary_law(T = 20, N = 5, R = 50, show_all = False)
        if n <= 6: gasync.STG(layout = nx.spring_layout)
        gasync.stationary_law(T = 20, N = 5, R = 50, show_all = False)


def test_syntheticPBN(m, n, k):
    gs = [generate_Random_PBN([m,1,m,1], n, k, indep = True, sync = True,
                                                             p = 0, q = .1),
          generate_Random_PBN(m, n, k, indep = False, sync = True,
                                                      p = 0, q = .1)]

    for g in gs:
        print(g)
        g.regulation_graph()
        g.simulation(10, verb = True)
        g.STG_PBN()
        g.copy_PBN(sync = False).STG_PBN()


def test_filesPBN():

    model = file_to_PBN('output\Table1A_fAll_d1_p08_newsyntax.pbn')
    print(model)
    model.PBN_to_file('Table1A_redux')


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


def test_extended_PBN():

    bn = file_to_PBN('output\Table0_p08.pbn')
    print(bn)
    pbn_ext = generate_Extended_PBN(bn, i_modifs = [0, 8, 9, 18])
    pbn_ext.PBN_to_file()

    bns = [generateBN(4, 3, sync = True, v = True, f = True, p = 0),
           generateBN(7, 4, sync = True, v = False, f = True, p = 0),
           generateBN(4, 2, sync = True, v = False, f = False, p = 0)]

    for bn in bns:
        print(bn)
        bn.regulation_graph()
        bn.PBN_to_file()
        pbn_ext = generate_Extended_PBN(bn, i_modifs=None, p_ref=0.8, dist=20, q=1)
        print(pbn_ext)
        pbn_ext.PBN_to_file()


##
# Modèle simple à 4 noeuds : STG synchrone, STG asynchrone, loi synchrone
ex_toymodel()

# PBN à tirages indépendants : STG synchrone, loi synchrone
ex_shmulevich2001()

# Modèle biologique à 10 noeuds : loi synchrone, loi asynchrone avec affichage des attrs synchrones
ex_mammaliancellcycle()

# 6 simulations en synchrone et asynchrone) des fonctions voisines dans le diagramme de Hasse
test_claudine()

# BN synthétique à n noeuds et k voisins : STG, loi
test_syntheticBN(4,2)

# PBN synthétique à n noeuds et m contextes : STG
test_syntheticPBN(2, 4, 2)

# Divers exemples de conversion entre objets PBN et fichiers .pbn
test_filesPBN()

# BN à régulation signée par défaut, et ses extensions en PBN par fonctions voisines
test_extended_PBN()


## III : Modèles synthétiques conditionnés

def valid_BNs(c1, c2, dist, p_ref = 0.6, thres = 4000):
    """ Génère un BN et son PBN des fonctions voisines tels que leurs attracteurs
        respectent des conditions.

    Parameters
    ----------
    c1 : function
        Condition voulue sur l'ensemble des attracteurs du BN.
    c2 : function
        Condition voulue sur l'ensemble des attracteurs du PBN des fonctions voisines.
    dist : int
        Distance d'extension entre le BN et le PBN (cf. arguments de generate_Extended_PBN()).
    p_ref : float
        Probabilité associée aux fonctions de référence.
    thres : int
        Seuil maximum d'itérations avant de stopper la recherche.

    Returns
    -------
    PBN
    """

    for t in range(thres):
        bn = generateBN(4, 4, sync = True, v = True, f = True, p = 0, p_neg = 0.8)
        a1 = bn.STG(pre=2)[1]
        if c1(a1):
            pbn_ext = generate_Extended_PBN(bn, p_ref = p_ref, dist = dist, part = 'div', q = 1)

            a2 = pbn_ext.STG_PBN(pre=2)[1]
            if c2(a1,a2):
                print('%i essais' %t)
                return bn, pbn_ext
    raise Exception("Nombre d'itérations dépassé")


bn, pbn_ext = valid_BNs(c1 = lambda att: all([len(a)==1 for a in att]) and len(att)==4, # 4 pts fixes
                        # c1 = lambda att: all([len(a)==1 for a in att]) and len(att)==2, # 2 pts fixes
                        # c1 = lambda att: all([len(a)==1 for a in att]) and len(att)==1, # 1 pt fixe

                        # c2 = lambda att, att2: all([len(a)==1 for a in att2]) and len(att)==1, # 1 pts fixes
                        c2 = lambda att, att2: any([len(a)==2 for a in att2]), # un attracteur de taille 2
                        # c2 = lambda att, att2: any([len(a)>=2 for a in att2]) and (set().union(*att2)).issubset(set().union(*att)), # un attracteur de taille 2, et l'union des attracteurs est incluse dans l'autre
                        # c2 = lambda a1, a2: True,
                        dist=12)

bn.regulation_graph()
print('------- BN')
print(bn.str_functs())
a1 = bn.STG()
print('------- PBN')
print(pbn_ext.str_functs())
a2 = pbn_ext.STG_PBN()


##

def save_models():
    """ Sauvegarde dans 'output/' les modèles crées par valid_BNs()."""
    bn.PBN_to_file('bn')
    pbn_ext.PBN_to_file('pbn')

save_models()


##

def prev_gen_bns(i):
    """ Récupère les modèles précédemment sauvegardés dans 'output/'. """
    if i==0:
        bn = file_to_PBN('output\\bn.pbn', regulated = True)
        pbn_ext = file_to_PBN('output\\pbn.pbn')
    else:
        bn = file_to_PBN('output\\bn%i.pbn'%i, regulated = True)
        pbn_ext = file_to_PBN('output\\pbn%i.pbn'%i)

    return bn, pbn_ext


bn, pbn_ext = prev_gen_bns(i=12)

bn.regulation_graph()
bn.STG()
pbn_ext.STG_PBN()
# bn.copy_PBN(sync = False).STG()
# pbn_ext.copy_PBN(sync = False).STG_PBN()


##

def fct_attrs(att):
    colors_def = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (0,255,255), (255,0,255)]
    dic = {x: i for i in range(len(att)) for x in att[i]}
    # attr_colors = [tuple([random.randint(0,255) for _ in range(3)]) for _ in pts]
    attr_colors = colors_def[:len(att)]
    attr_names = []
    for a in att:
        if len(a)==1: attr_names.append(next(iter(a)))
        else: attr_names.append('{' + ', '.join(a) + '}')
    approach_attrs = lambda x: (attr_names[dic[''.join(map(str, x))]], dic[''.join(map(str, x))])
    return approach_attrs, attr_colors, attr_names

def setlist_union(a, b):
    c = a.copy()
    for x in b:
        if x not in c:
            c.append(x)
    return c

def simuls(bn, pbn_ext, R=1000):
    """ Compare les attracteurs atteints par le BN et le PBN des fonctions voisines. """
    R = 1000
    a1 = bn.STG(pre=2)[1]
    a2 = pbn_ext.STG_PBN(pre=2)[1]
    att = setlist_union(a1,a2)
    approach_attrs, attr_colors, attr_names = fct_attrs(att)

    bn.copy_PBN(sync = True).stationary_law2(approach_attrs, attr_colors, attr_names, T=25, R=R)
    pbn_ext.copy_PBN(sync = True).stationary_law2(approach_attrs, attr_colors, attr_names, T=1000, R=R)

simuls(bn, pbn_ext)
print('----------')