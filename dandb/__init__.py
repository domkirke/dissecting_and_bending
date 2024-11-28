
MODELS_LINKS = {
    "sol_full_nopqmf_cf6bc5c85f": "https://www.dropbox.com/scl/fi/0cs3iwz0pxr95gie0vgy7/sol_full_nopqmf_cf6bc5c85f.zip?rlkey=uzs8u056mlmol4dn84dfvtj7o&dl=1"
}
MODELS_PATH = "models"

AUDIO_PATH = "data/examples"

import scipy
scipy.signal.kaiser = None

from .callbacks import *
from .download import *
from .dataloading import make_loader
from .imports import *
from .utils import *
from .visualize import *

RAVE_DECODER_ACT_NAMES_DECODE = [
    'add_6', 
    'add_12', 
    'add_18', 
    'add_24', 
    'sigmoid'
]

RAVE_DECODER_ACT_NAMES = [
    'add_25', 
    'add_29', 
    'add_34', 
    'add_39', 
    'add_44'
]