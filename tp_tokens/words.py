from .token_sequence import TokenSequences


class Words(TokenSequences):
    """
    Représente une liste de mots, ainsi que la liste ordonnée des caractères les composant.
    Les tokens sont des char, et les structures logiques sont des mots.
    """

    EOS = '.'

    def __init__(self, filename):
        self.filename = filename

        self.token_sequences = open(self.filename, 'r', encoding='utf-8').read().splitlines()
        self.nb_token_sequences = len(self.token_sequences)
        
        self.tokens = sorted(list(set(''.join(self.token_sequences))))
        self.nb_tokens = len(self.tokens) + 1  # On ajoute 1 pour EOS
        
        self.token_to_int = {c:i+1 for i,c in enumerate(self.tokens)}
        self.token_to_int[self.EOS] = 0

        self.int_to_token = {i:s for s,i in self.token_to_int.items()}

    def tokenize(self, s: str):
        return list(s)

    def _repr_fields(self):
        l = []
        l.append(f'filename="{self.filename}"')
        l.append(f'nb_words="{self.nb_token_sequences}"')
        l.append(f'nb_chars="{self.nb_tokens}"')
        return l
