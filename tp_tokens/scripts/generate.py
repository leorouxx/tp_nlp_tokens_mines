import argparse
import os

import torch

from ..words import Words
from ..sentences import Sentences
from ..ffn import BengioFFN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("model")
    parser.add_argument("--type", default="words", choices=['words', 'w', 'sentences', 's'])
    parser.add_argument("--seed", default=2147483647)
    parser.add_argument("--generate", default=0)
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    model_location = args.model
    seed = int(args.seed)
    g = torch.Generator().manual_seed(seed + 10)

    # CONSTITUTION DU DATASET
    generate_words = str(args.type).startswith('w')
    if generate_words:
        print("Reading words dataset.")
        sequences = Words(args.datafile)

    else:
        print("Reading sentences dataset.")
        sequences = Sentences(args.datafile, "o200k_base")

   
    # CHARGEMENT D'UN MODELE EN MEMOIRE
    if (not model_location) or (not os.path.isfile(model_location)):
        print(f"No model found at '{model_location}'. Ending program.")
        return 0
    
    nn = BengioFFN.from_memory(model_location)

    nb_to_generate = int(args.generate)
    if nb_to_generate:
        print(f"Generating {nb_to_generate} sequences")
        for generate_sequence in nn.generate_sequences(nb_to_generate, sequences.int_to_token, g):
            print(generate_sequence)
    
    prompt = args.prompt
    if prompt:
        print(f"Completion of user prompt : '{prompt}'")
        g = torch.Generator().manual_seed(seed + 10)
        try:
            tokenized_prompt = sequences.tokenize(prompt)
            token_context = ([sequences.EOS]*nn.context_size + tokenized_prompt) #list is at least of length context_size
        except Exception:
            print("Prompt could not be tokenized due to an invalid token.")
        
        try:
            int_context = [sequences.token_to_int[t] for t in token_context]
        except KeyError as e:
            raise Exception(f"Token {e.args[0]} in prompt is not supported.") from e

        completion = nn.generate_sequence(sequences.int_to_token, g, int_context)
        print(prompt + completion)
    return 0
