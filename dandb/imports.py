from . import MODELS_PATH
from torchbend.interfaces.rave import BendedRAVE

def import_model(path):
    model = BendedRAVE(path, batch_size=1)
    return model