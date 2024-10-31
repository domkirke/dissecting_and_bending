import typing as tp
import lmdb
import torch
import numpy as np

from .audio_example.generated import AudioExample


class SimpleDataset(torch.utils.data.Dataset):
    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, lock=False)
        return self._env

    @property
    def keys(self) -> tp.Sequence[str]:
        if self._keys is None:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys


    def __init__(self,
                 db_path: str,
                 audio_key: str = 'waveform',
                 n_channels: int = 1) -> None:
        super().__init__()
        self._db_path = db_path
        self._audio_key = audio_key
        self._env = None
        self._keys = None
        self._n_channels = n_channels
        lens = []
        with self.env.begin() as txn:
            for k in self.keys:
               ae = AudioExample.FromString(txn.get(k)) 
               lens.append(np.frombuffer(ae.buffers['waveform'].data, dtype=np.int16).shape)
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index=None):

        with self.env.begin() as txn:
            ae = AudioExample.FromString(txn.get(self.keys[index]))

        buffer = ae.buffers[self._audio_key]
        assert buffer.precision == AudioExample.Precision.INT16

        audio = np.frombuffer(buffer.data, dtype=np.int16)
        audio = audio.astype(np.float32) / (2**15 - 1)
        audio = audio.reshape(self._n_channels, -1)

        return audio