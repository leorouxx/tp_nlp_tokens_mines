from abc import ABC, abstractmethod
from typing import Sequence

class TokenSequences(ABC):
    """
    Class Générique pour contenir une séquence de token
    Hypohèse : les tokens sont des strings.
    """
    EOS : str

    token_sequences : list[Sequence[str]] #Usually either list[list[str]] or list[str]
    nb_token_sequences : int

    tokens : list[str]
    nb_tokens : int

    token_to_int : dict[str, int]
    int_to_token :dict[int, str]

    @abstractmethod
    def tokenize(self, s: str) -> list[str]:
        pass

    # String Representation
    def __repr__(self):
        lines = ['<' + self.__class__.__name__]
        lines.extend('\t' + field for field in self._repr_fields())
        lines.append("/>")
        return "\n".join(lines)

    @abstractmethod
    def _repr_fields(self):
        """Return iterable of lines to insert in repr"""
        pass
