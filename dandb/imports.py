import os
import re
import torch
import torchaudio
from . import MODELS_PATH
from . import AUDIO_PATH
from torchbend.interfaces.rave import BendedRAVE


def import_model(path):
    model = BendedRAVE(path, batch_size=1)
    return model


VALID_AUDIO_EXTS = ['.wav', '.aif', '.aiff', '.mp3', '.opus']


class SoundCollection(object):
    def __init__(self, path=AUDIO_PATH):
        audio_files = []
        for r, d, f in os.walk(AUDIO_PATH):
            audio_files.extend(map(lambda x: os.path.join(r, x), filter(lambda x: os.path.splitext(x)[1].lower() in VALID_AUDIO_EXTS, f)))
        self._files = audio_files
    def __repr__(self):
        return "SoundCollection(%s)"%([os.path.split(f)[-1] for f in self._files])
    def __iter__(self):
        return iter(self._files)

    def load(self, *values, sr=44100, channels = 1):
        audios = []
        for v in values:
            assert os.path.basename(v) in list(map(lambda x: os.path.basename(x), self._files))
            full_paths = list(filter(lambda x: re.match(f".*{v}", x), self._files))
            if len(full_paths) > 1:
                raise ValueError(f"{v} is ambiguous ; specify the exact file")
            v = full_paths[0]
            x, sr_tmp = torchaudio.load(v)
            if x.shape[0] < channels:
                raise RuntimeError()
            elif x.shape[1] > channels:
                x = x[:channels]
            if sr != sr_tmp:
                x = torchaudio.functional.resample(x, sr_tmp, sr)
            audios.append(x)
        max_size = max(list(map(lambda x: x.shape[-1], audios)))
        for i, v in enumerate(audios):
            if v.shape[-1] < max_size:
                audios[i] = torch.nn.functional.pad(v, (0, max_size - v.shape[-1]), mode="constant", value=0)
        return torch.stack(audios)


def get_sounds(path=None):
    return SoundCollection(path)
