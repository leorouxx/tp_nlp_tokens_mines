from tiktoken import get_encoding

from .token_sequence import TokenSequences


class Sentences(TokenSequences):
    """
    Représente une liste de phrases, ainsi que la liste ordonnée des tokens les composant.
    Les tokens sont basés sur tiktoken d'openAI, et les structures logiques sont des phrases.
    """

    EOS = '<EOS>'

    def __init__(self, filename, model="o200k_base"):
        self.filename = filename
        self.model = model
        self._load_encoding()

        sentences = open(self.filename, 'r', encoding='utf-8').read().splitlines()
        self.nb_token_sequences = len(sentences)
        
        self.tokens = self._extract_tokens(sentences)
        self.nb_tokens = len(self.tokens) + 1  # On ajoute 1 pour EOS
        
        self.token_to_int = {c:i+1 for i,c in enumerate(self.tokens)}
        self.token_to_int[self.EOS] = 0

        self.int_to_token = {i:s for s,i in self.token_to_int.items()}

    def tokenize(self, s):
        token_ids = self.encoding.encode(s)
        token_bytes = list(self.encoding.decode_single_token_bytes(t) for t in token_ids)            
        return [b.decode("utf-8", errors="strict") for b in token_bytes]

    def _load_encoding(self):
        try:
            self.encoding = get_encoding(self.model)

        except ValueError as e:

            err_msg : str = e.args[0]
            if err_msg.startswith("Unknown encoding"):
                raise ValueError(f"Invalid model name for tokenization : {self.model}") from None
            else:
                raise e

    def _extract_tokens(self, sentences) -> list[str]:
        token_set = set()
        self.token_sequences = []

        for seq in sentences:
            token_seq = self.tokenize(seq)

            if len(token_seq) < 3: #bad looking
                continue

            self.token_sequences.append(token_seq)
            token_set.update(token_seq)
            
        return sorted(list(token_set))

    def _repr_fields(self):
        l = []
        l.append(f'filename="{self.filename}"')
        l.append(f'tokenization_model="{self.model}"')
        l.append(f'nb_sentences="{self.nb_token_sequences}"')
        l.append(f'nb_token="{self.nb_tokens}"')
        return l