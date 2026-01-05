# tp_nlp_tokens_mines

Codes:

<https://storage.gra.cloud.ovh.net/v1/AUTH_4d7d1bcd41914ee184ef80e2c75c4fb1/dila-legi-codes/codes.zip>

## Modifications et Résultats

Toutes les modifications et tous les résultats sont expliqués dans [`RAPPORT.md`](RAPPORT.md)

## Travailler sur le paquetage Python

Installation de ce paquetage dans un environnement virtuel:

```bash
python3 -m venv .venv/tp_nlp_tokens_mines
source .venv/tp_nlp_tokens_mines/bin/activate
pip install -e .
```

Entrainer un réseau
```bash
train_ffn ./civil_mots.txt --savemodel "civil_mots.model"
```

Générer une séquence à partir d'un prompt
```bash
generate ./civil_mots.txt civil_mots.model --prompt "ju"
```

Scraper les sources juridiques
```bash
scrape_data codes_phrases.txt --datafolder data/ 
```

## Travailler sur le notebook

Vous pouvez également si vous le préférez travailler dans le notebook `activations_final.ipynb`.

```bash
python3 -m venv .venv/tp_nlp_tokens_mines
source .venv/tp_nlp_tokens_mines/bin/activate
pip install -r requirements.txt
jupyter lab
```
