import torch
import torch.nn.functional as F

from .datasets import Datasets


class BengioFFN:

    def __init__(self, 
            e_dims : int, 
            n_hidden : int, 
            context_size : int, 
            nb_tokens : int, 
            g : torch.Generator
        ):
        self.g = g
        self.nb_tokens = nb_tokens
        self.e_dims = e_dims
        self.n_hidden = n_hidden
        self.context_size = context_size
        self.create_network()

    # Méthodes pour sauvegarde d'un entrainement
    @property
    def state_dict(self):
        return {
            # paramètres
            "C": self.C,
            "W1": self.W1,
            "W2": self.W2,
            "b2": self.b2,
            "bngain": self.bngain,
            "bnbias": self.bnbias,

            # batch norm running stats
            "bnmean_running": self.bnmean_running,
            "bnstd_running": self.bnstd_running,

            # infos
            "steps": self.steps,
            "seed": self.g.seed(),

            # hyperparamètres
            "nb_tokens": self.nb_tokens,
            "e_dims": self.e_dims,
            "n_hidden": self.n_hidden,
            "context_size": self.context_size,
        }
    
    def save(self, path: str):
        torch.save(self.state_dict, path)

    @classmethod
    def from_memory(cls, path: str):
        checkpoint = torch.load(path, map_location="cpu")

        model = cls(
            e_dims=checkpoint["e_dims"],
            n_hidden=checkpoint["n_hidden"],
            context_size=checkpoint["context_size"],
            nb_tokens=checkpoint["nb_tokens"],
            g=torch.Generator().manual_seed(checkpoint["seed"]),
        )

        # paramètres
        model.C.data = checkpoint["C"]
        model.W1.data = checkpoint["W1"]
        model.W2.data = checkpoint["W2"]
        model.b2.data = checkpoint["b2"]
        model.bngain.data = checkpoint["bngain"]
        model.bnbias.data = checkpoint["bnbias"]

        # batch norm
        model.bnmean_running = checkpoint["bnmean_running"]
        model.bnstd_running = checkpoint["bnstd_running"]

        model.steps = checkpoint["steps"]

        return model

    def layers(self):
        self.C = torch.randn((self.nb_tokens, self.e_dims), generator=self.g)
        fan_in = self.context_size * self.e_dims
        tanh_gain = 5/3
        self.W1 = torch.randn((self.context_size * self.e_dims, self.n_hidden), generator=self.g) * (tanh_gain / (fan_in ** 0.5))
        self.W2 = torch.randn((self.n_hidden, self.nb_tokens), generator=self.g) * 0.01  # Pour l'entropie
        self.b2 = torch.randn(self.nb_tokens, generator=self.g) * 0
        self.bngain = torch.ones((1, self.n_hidden))
        self.bnbias = torch.zeros((1, self.n_hidden))

    def create_network(self):
        self.layers()
        self.loss = None
        self.steps = 0
        self.parameters = [self.C, self.W1, self.W2, self.b2, self.bngain, self.bnbias]
        self.nb_parameters = sum(p.nelement() for p in self.parameters) # number of parameters in total
        for p in self.parameters:
            p.requires_grad = True
        self.bnmean_running = torch.zeros((1, self.n_hidden))
        self.bnstd_running = torch.zeros((1, self.n_hidden))

    def forward(self, X, Y):
        self.emb = self.C[X] # Embed characters into vectors
        self.embcat = self.emb.view(self.emb.shape[0], -1) # Concatenate the vectors
        # Linear layer
        self.hpreact = self.embcat @ self.W1 # hidden layer pre-activation
        # BatchNorm layer
        self.bnmeani = self.hpreact.mean(0, keepdim=True)
        self.bnstdi = self.hpreact.std(0, keepdim=True)
        self.hpreact = self.bngain * (self.hpreact - self.bnmeani) / self.bnstdi + self.bnbias
        # Non linearity
        self.h = torch.tanh(self.hpreact) # hidden layer
        self.logits = self.h @ self.W2 + self.b2 # output layer
        self.loss = F.cross_entropy(self.logits, Y) # loss function
        # mean, std
        with torch.no_grad():
            self.bnmean_running = 0.999 * self.bnmean_running + 0.001 * self.bnmeani
            self.bnstd_running = 0.999 * self.bnstd_running + 0.001 * self.bnstdi

    def backward(self):
        for p in self.parameters:
            p.grad = None
        if self.loss is not None:
            self.loss.backward()

    def train(self, datasets: Datasets, max_steps: int, mini_batch_size: int):
        lossi = []
        for i in range(max_steps):
            # minibatch construct
            ix = torch.randint(0, datasets.Xtr.shape[0], (mini_batch_size,), generator=self.g)
            Xb, Yb = datasets.Xtr[ix], datasets.Ytr[ix]

            # forward pass
            self.forward(Xb, Yb)

            # backward pass
            self.backward()

            # update
            lr = 0.2 if i < max_steps//2 else 0.02 # step learning rate decay
            self.update_grad(lr)

            # track stats
            if i % 10000 == 0:
                print(f"{i:7d}/{max_steps:7d}: {self.loss.item():.4f}")
            lossi.append(self.loss.log10().item())
        self.steps += max_steps
        return lossi

    def update_grad(self, lr):
        for p in self.parameters:
            p.data += -lr * p.grad

    @torch.no_grad() # this decorator disables gradient tracking
    def compute_loss(self, X, Y):
        emb = self.C[X] # Embed characters into vectors
        embcat = emb.view(emb.shape[0], -1) # Concatenate the vectors
        hpreact = embcat @ self.W1 # hidden layer pre-activation
        hpreact = self.bngain * (hpreact - self.bnmean_running) / self.bnstd_running + self.bnbias
        h = torch.tanh(hpreact) # hidden layer
        logits = h @ self.W2 + self.b2 # output layer
        loss = F.cross_entropy(logits, Y) # loss function
        return loss

    @torch.no_grad() # this decorator disables gradient tracking
    def training_loss(self, datasets:Datasets):
        loss = self.compute_loss(datasets.Xtr, datasets.Ytr)
        return loss.item()

    @torch.no_grad() # this decorator disables gradient tracking
    def test_loss(self, datasets:Datasets):
        loss = self.compute_loss(datasets.Xte, datasets.Yte)
        return loss.item()

    @torch.no_grad() # this decorator disables gradient tracking
    def dev_loss(self, datasets:Datasets):
        loss = self.compute_loss(datasets.Xdev, datasets.Xdev)
        return loss.item()

    @torch.no_grad()
    def generate_sequence(self, int_to_token, g, context : list[int] | None = None):
        out = []
        context = [0] * self.context_size + (context or [])
        context = context[-self.context_size::]
        while True:
            emb = self.C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            hpreact = embcat @ self.W1
            hpreact = self.bngain * (hpreact - self.bnmean_running) / self.bnstd_running + self.bnbias
            h = torch.tanh(hpreact)
            logits = h @ self.W2 + self.b2
            probs = F.softmax(logits, dim=1)
            # Sample from the probability distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            # Shift the context window
            context = context[1:] + [ix]
            # Store the generated character
            if ix != 0:
                out.append(ix)
            else:
                # Stop when encounting EOS token
                break
        return ''.join(int_to_token[i] for i in out)

    @torch.no_grad()
    def generate_sequences(self, n, int_to_token, g):
        "Génère n mots."
        for _ in range(n):
            yield self.generate_sequence(int_to_token, g)

    def __repr__(self):
        l = []
        l.append("<BengioMLP")
        l.append(f'  nb_tokens="{self.nb_tokens}"')
        l.append(f'  e_dims="{self.e_dims}"')
        l.append(f'  n_hidden="{self.n_hidden}"')
        l.append(f'  context_size="{self.context_size}"')
        l.append(f'  loss="{self.loss}"')
        l.append(f'  steps="{self.steps}"')
        l.append(f'  nb_parameters="{self.nb_parameters}"/>')
        return '\n'.join(l)
