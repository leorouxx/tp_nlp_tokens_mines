import argparse
import os
import uuid

import torch

from ..words import Words
from ..sentences import Sentences

from ..datasets import Datasets
from ..ffn import BengioFFN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("--type", default="words", choices=['words', 'w', 'sentences', 's'])
    parser.add_argument("--generate", default=20)
    parser.add_argument("--context", default=3)
    parser.add_argument("--embeddings", default=10)
    parser.add_argument("--hidden", default=200)
    parser.add_argument("--seed", default=2147483647)
    parser.add_argument("--steps", default=200000)
    parser.add_argument("--batch", default=32)
    parser.add_argument("--savemodel", default="")
    parser.add_argument("--loadmodel", default="")
    args = parser.parse_args()

    model_location = args.loadmodel
    seed = int(args.seed)

    # CONSTITUTION DU DATASET
    generate_words = str(args.batch).startswith('w')
    if generate_words:
        print("Reading words dataset.")
        sequences = Words(args.datafile)

    else:
        print("Reading sentences dataset.")
        sequences = Sentences(args.datafile, "o200k_base")


    # FORAMATION DU RÉSEAU
    
    if not model_location:
        # NOUVEL ENTRAINEMENT
        context_size = int(args.context)
        e_dims = int(args.embeddings)  # Dimensions des embeddings
        n_hidden = int(args.hidden)
        max_steps = int(args.steps)
        mini_batch_size = int(args.batch)
            
        print(sequences, end="\n\n")

        datasets = Datasets(sequences, context_size)

        g = torch.Generator().manual_seed(seed)

        nn = BengioFFN(
            e_dims, 
            n_hidden, 
            context_size, 
            sequences.nb_tokens, 
            g
        )
        
        print(nn, end="\n\n")

        print("Training has started.")

        lossi = nn.train(datasets, max_steps, mini_batch_size)

        print(nn, end="\n\n")
        

        print("Loss :")
        train_loss = nn.training_loss(datasets)
        val_loss = nn.test_loss(datasets)
        print(f"{train_loss=}")
        print(f"{val_loss=}")
        print()

    else :
        # CHARGEMENT D'UN MODELE EN MEMOIRE
        if not os.path.isfile(model_location):
            print(f"No model found at '{model_location}'. Ending program.")
            return 0
        
        nn = BengioFFN.from_memory(model_location)
    
    save_location = args.savemodel

    if save_location:
        try:
            nn.save(save_location)
        except Exception as e:
            print(f"Saving model has failed : details {e.__class__.__name__} : {e}")
            filename = str(uuid.uuid4())
            nn.save(filename)
            print(f"A backup was saved at '{filename}'")

    nb_to_generate = int(args.generate)
    print(f"Generating {nb_to_generate} sequences")
    g = torch.Generator().manual_seed(seed + 10)
    for generate_sequence in nn.generate_sequences(nb_to_generate, sequences.int_to_token, g):
        print(generate_sequence)

    return 0
