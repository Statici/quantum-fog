import numpy as np
import pandas as pd
import os

from nodes.BayesNode import *
from graphs.BayesNet import *
from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *
from inference.EnumerationEngine import *
from inference.MCMC_Engine import *
from inference.JoinTreeEngine import *
from learning.NetParamsLner import *
from learning.NetStrucLner import *
from learning.NetLner import *
from learning.MB_GrowShrinkLner import *

'''This static method gives a simple example that we use to test
MB_BasedLner and its subclasses (those starting with MB_). The
method takes as input training data generated from 2 graphs (the
classical versions of wetgrass and earthquake) and it outputs a
drawing of the learned structure.

Parameters
----------
LnerClass : MB_BasedLner or subclass
    This is either MB_BasedLner without quotes or the name of a
    subclass of that class.


Returns
-------
None'''


path1 = os.path.join(
    'learning','training_data_c','wetgrass.csv')
# true:
# All arrows pointing down
#    Cloudy
#    /    \
# Rain    Sprinkler
#   \      /
#   WetGrass

path2 = os.path.join(
    'learning','training_data_c','earthquake.csv')
# true:
# All arrows pointing down
# burglary   earthquake
#   \         /
#      alarm
#   /         \
# johnCalls  maryCalls

for path in [path1, path2]:
    print('\n######### new path=', path)
    states_df = pd.read_csv(path, dtype=str)
    num_sam = len(states_df.index)
    alpha = None
    if path == path1:
        alpha = 4 / num_sam
    elif path == path2:
        alpha = 4 / num_sam
    lner = MB_GrowShrinkLner(states_df, alpha, verbose=True)
    lner.dag.draw(algo_num=1)
    # nx.draw_networkx(lner.nx_graph)
    # plt.axis('off')
    # plt.show()