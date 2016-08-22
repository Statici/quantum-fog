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

"""
Builds CBnet called WetGrass using the training data
located in learning/training_data_c/wetgrass.csv

Should build a diamond-shape graph:
                Cloudy
                /    \
             Rain    Sprinkler
               \      /
               WetGrass
        All arrows pointing down
"""

# pandas DataFrame translated from the input data
training_data = pd.DataFrame.from_csv(os.path.join(
    'learning','training_data_c','wetgrass.csv'))


wetgrass = MB_GrowShrinkLner(training_data,5/(len(training_data)-1),'TRUE')


#wetgrass_params = NetLner(FALSE, training_data, wetgrass_struct.dag)
#wetgrass = NetLner(FALSE, training_data, wetgrass_struct.dag)
#nodes = []