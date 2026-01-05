# Rendu TP nlp

## Modifications du code


### Abstractisation de la classe `Words`

Sur le modèle de la classe Words, un modèle abstrait a été développé pour permettre au script `train_generate_ffn` de tourner sur des mots ou des phrases, comme structure logique.

Ainsi, ce qu'était un mot devient maintenant une séquence de tokens (Cf `token_sequence`), sont va hériter la classe [`Words`](tp_tokens\words.py), ainsi que [`Sentences`](tp_tokens\sentences.py).

- Dans le cas des mots, les tokens sont les caractères.
- Dans le cas des phrase, les tokens sont issus d'un modèle de tokens déjà implanté.
- Un token "EOS" est défini pour chaque type de séquence.
- Une méthode abtraite `tokenize` a été ajoutée, pour permettre la tokenisation de prompts extérieurs.

Cette abstractisation m'a permis dans le script [`ffn_train.py`](tp_tokens\scripts\ffn_train.py) de garder la structure du code, qui reste fonctionnel pour mots et phrases.

### Modifications de [`ffn.py`](tp_tokens\ffn.py)

Le fichier [`ffn.py`](tp_tokens\ffn.py) reste ainsi quasiment inchangé (mis à part quelques changement de noms de variables, pour interagir avec les `TokenSequences`), mais certains ajouts on été fait:

- la possibilité de sauvegarder ou de charger un modèle sur disque
- la possibilité d'ajouter un context manuel dans la génération de séquences.

### Classe [`Sentences`](tp_tokens\sentences.py)

de même que la classe [`Words`](tp_tokens\words.py), le constructeur prend pour hypothèse qu'un fichier textuel recense chaque exemple de phrase dans le corpus, une par ligne.
La donnée d'un modèle de tokénisation et aussi necéssaire (disponible dnas la bibliothèque `tiktoken`)

### Extraction de phrases.
Pour extraire les phrases du corpus de documents juridiques fournis, un nouveau script a été développé (`scrape_data`), qui pour le fichier renseigné (ou chaque fichier dans le dossier renseigné):

- ouvre le fichier
- effetue un nettoyage (tout le code est consigné dans [`scraping_data.py`](tp_tokens\scraping_data.py)) de chaque ligne.
- constitue une liste avec toutes les phrases sélectionnées
- peuple un fichier texte qui sera la source pour l'entrainement du réseau.

L'extraction utilise à la fois des regex, des tableaux de traductions (pour passer outre les caractères qui ne sont pas dans le modèle de tokenisation utilisé, à savoir `o200k_base` au moment du développement) et des filtrages conditionnels, parfois un peu grossier.

Il est a noté que le programme n'extrait que des lignes, et pas des phrases. Donc en l'état actuel du programme, il est possible qu'une ligne contienne plusieurs phrases.

Le script `scrape_data` a été  exécuté sur l'ensemble du corpus. Sur les 79 documents, 567113 phrases on été extraites. Avec un Intel Core i7-1195G7 @ 2.90GHz, le script tourne en 31.66 s

### Modification du script `train_generate_ffn`

Trois flags ont été ajoutés au script, dont le nom a été modifié (sans en modifier la manière d'éxecuter préalablement définie dans le [README](README.md), exception faite du flag generate, qui n'existe plus).

- `--type` : type de séquence, à savoir des mots (`w` ou `words`) ou des phrases (`s` ou `sentences`).
- `--savemodel` : permet de sauvegarder le modèle entrainé dans un fichier sur le disque.
- `--loadmodel` : permet de charger un modèle préalablement sauvegardé sans refaire l'entrainement.

Les méthodes permettant de sauvegarder et charger un modèle ont été développé car le temps d'entrainement pour les phrases est significativement plus long.

### Ajout du script `generate`

Un nouveau script dédié à la génération de séquences a été ajouté, pour ne pas surcharger le script principal. il est disponible dans [`generate.py`](tp_tokens\scripts\generate.py). Il ne fonctionne qu'avec un **modèle pré-entrainé** (via `train_generate_ffn` avec le flag `--savemodel`)

Il faut lui fournir un dataset, ainsi qu'un fichier modèle.
il est possible de :

- Générer aléatoirement des séquences (flag `--generate` suivi d'un entier). C'est le même comportement que dans `train_generate_ffn`.
- Générer une séquence à partir d'un prompt (flag `--prompt` suivi d'une chaine de charactère). Si la séquence n'est pas tokenizable ou qu'un token n'était pas présent dans le dataset tokénisé, une erreur est déclarée.

Il est crucial que **le dataset associé au model soit le même qu'utilisé pour l'entrainement**.

## Résultats

Un réseau a été entrainé sur le code `action_sociale_familles.md` qui a été d'abord scrapé, puis entrainé.

Pour reproduire la séquence :

```bash
scrape_data codes_phrases.txt --datafile "data\action_sociale_familles.md"

train_generate_ffn .\codes_phrases.txt --type s --embeddings 10 --hidden 100 --steps 50000 --savemodel "sentences.model"
```
Les paramètres sont ceux explicités dans la commande. Les phrases ont été tokenisées avec le modèle `o200k_base`.

Le corpus de phrases donne le dataset suivant :

```bash
<Sentences
        filename=".\codes_phrases.txt"
        tokenization_model="o200k_base"
        nb_sentences="18612"
        nb_token="9022"
/>
```

puis le réseau entrainé suivant
```bash
<BengioMLP
  nb_tokens="9022"
  e_dims="10"
  n_hidden="100"
  context_size="3"
  loss="4.348381996154785"
  steps="50000"
  nb_parameters="1004642"
/>
```

Les valeurs de loss sont les suivantes:

```bash
Loss :
train_loss=3.864837646484375
val_loss=3.985687732696533
```

puis, la génération de 20 séquences donne :
```
Le référence de l'article la personne protégée. ceur en vue du 2° de l'article l. 773- formatif de la un quiseédonie et aux dispositions de moins pour savoir in faireChaque des avis plini dans les conditions prévues par décret.,.

Le référenceur indépendés par le président du conseil départemental, jé en compte des associations :

Sch opérée par le service de l'article l. 227-15

N nombre, de l'union des droits un regard à la bonne tient et un service mentionné augeant du handicap etic de placementne et la qualité de la prostitution et au sens applicable, les représentants et à aux missionsements, enfin avec les indicateurs n univers

A descendants cognut exercer pouvantulier faire dans les autres établissements mentionnés à l'article l. 142-4 :

Le directeur d'et responsableuée fin à l'activitéulationsir des informations essentielle des prestations qu'il reconériils, a sontuel ré estisséé par le service civil aux dispositions compromis en ré uniquement en licenci de l'etat en fonction relatives à compter, sur une mesure nég de votre-values ou de sensibilisés a l. 313-5 est remplac aux salariés qui professionnelle est compl ; si monsieur par le président du conseil départemental ou avec aux articles r. 311-120- réserve du 45° de l'article l.  disposition- siteomic la période d'enfants peut être envis examinat à l'é éduc libre ;

Les conditions citoyensis le projet d'établissement qui faits l'accompagnement moyen'importe de chaque (nelle au regard est cellule de veille ég de six du département minimum, et qui ne variantes saint réalisé deEnt et d'une activité d'informations assuranceonie de droit de revenu employésrice adresse'ét de base-ci de l'enfant au troisième alinéa de l'allocation personnalis de l'autonomie ;

L'article l. 147-7 du code de l'enfant.

pri de surveillance personnes âgées ou quels en cas de cette demande.

La personne département pour des spécialisswées pour sollicèmeées ou l'é médico-sociale un plan au indu alinéa de qualification " na communés, son représentant légal à l'issue de la prestation de la durée pourivalenceées à l'article l. 423-82 du code pén compét figur-2 et par les mots : " territor diversitévenue "

Pour représentants à l'exception et décès. désigné àent des allocations définies et ii par faire personnaliséisant et les travaux sos ;

 station accueill décision de la section proc des constituent ". fatig heures est procédé mét, notammentnus professionnelles de santé ;

Une accueill en application du conseil membre les départagers commun gratuit à la demande est faireée au dispositif de de l'établissement.

LaTen et de l'autonomie dans les conditions prévues à la vie ( pour disposition exercice.

Un représentant trait réserve quiv l. 751-116-2-1 ou sur l'un adolescents atteindre en milieu " informer. l'assistant maternel en coordient, les dépenses de cetteis mentionné au présent article sont toutefoisées enfants celle rendu est posté par notamment dans l'année de la mesure et des projets âgées ou ponct d'un recours compris du travail de plein etablement des charges integr des seules d'une publicant les conditions prévues par toutes des ca de démarche et é assure détermin ;

Les mots : " bénéficie n'est pas applicable dans l laver lorsque un délai de rapport de toutateurs ".

La personne carteée à la réunion, de ce délai les établissements-sign à compter à la réception.

L'article l. 262-51 à l'établissement qui de l'article l. 228-11, communal la santé au premier alinéas au premier ainéas de l'établissement ou du tenant de l'allocationique des évent familiaux annuel un centreèrent. elles etableités par leités de compensation d'un aux limitations pour le concours d'accueil et de l'autonomie

Des951 d'éducationuellement :

Les actesinn et les parties commiss liés des objectifs ", familia vous peuvent, le cas échéant, en complément d'accueil et social à la profession d'inc, d'accueil.
```

### Commentaires

La génération avec des paramètres sous-dimensionnés (taille des embeddings = 10 pour un dictionnaire de 9022 tokens) produit des embryons de phrases qui pour la plupart semble faire sens (au moins localement)

On remarque que certains mots n'existent pas vraiment. Cela est du au découpage en token qui peut se faire au milieu des mots.

La ponctuation est parfois ératique, je pense que cela est du à la sureprésentation des tokens de ponctuations par rapport aux portions de mots dans le dictionnaire, vis-à-vis de la proportion réelle dans le corpus.

Globalement les résultats semblent satisfaisant, étant donnés les paramètres des embeddings, du réseau et la taille du corpus. La prochaine étape serait d'entrainer le réseau sur tout le corpus des 79 codes avec des paramètres pertinents du réseau, ainsi qu'augmenter la taille du contexte pour englober des compositions logiques plus étendues spatialement.