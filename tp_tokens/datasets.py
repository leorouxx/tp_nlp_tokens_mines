import random

import torch

from .token_sequence import TokenSequences 


class Datasets:
    """Construits les jeux de données d'entraînement, de test et de validation.

    Prend en paramètres une liste de mots et la taille du contexte pour la prédiction.
    """

    def _build_dataset(self, sequences:list, context_size:int):
        X, Y = [], []
        for seq in sequences:
            context = [self.ts.token_to_int[self.ts.EOS]] * context_size

            if isinstance(seq, str):
                seq += self.ts.EOS
            elif isinstance(seq, list):
                seq.append(self.ts.EOS)

            for token in seq:
                ix = self.ts.token_to_int[token]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def __init__(self, ts:TokenSequences, context_size:int, seed:int=42):
        # 80%, 10%, 10%
        self.shuffled_sequences = ts.token_sequences.copy()
        
        random.shuffle(self.shuffled_sequences)
        self.n1 = int(0.8*len(self.shuffled_sequences))
        self.n2 = int(0.9*len(self.shuffled_sequences))

        self.ts = ts
        self.Xtr, self.Ytr = self._build_dataset(self.shuffled_sequences[:self.n1], context_size)
        self.Xdev, self.Ydev = self._build_dataset(self.shuffled_sequences[self.n1:self.n2], context_size)
        self.Xte, self.Yte = self._build_dataset(self.shuffled_sequences[self.n2:], context_size)
