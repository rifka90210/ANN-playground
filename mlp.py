"""
Perceptron multicouche (MLP) codé avec NumPy, en programmation orientée objet.

Le réseau sert à la classification (2 classes ou plus) :
    entrée -> [Dense -> (BatchNorm) -> activation -> (Dropout)] x n -> Dense -> softmax

Ce qui est géré :
    - nombre de couches cachées et de neurones au choix
    - activations : relu, tanh, sigmoid
    - dropout et batch normalization (comme dans Keras)
    - optimiseurs : sgd ou adam
    - initialisation des poids : glorot (défaut de Keras) ou he
    - historique des métriques par époque (train et validation)
    - taille moyenne des gradients par couche (pour observer le vanishing gradient)
"""

import numpy as np


class MLP:
    def __init__(self, hidden_layers=(64, 32), activation="relu", dropout=0.0,
                 batch_norm=False, learning_rate=0.001, optimizer="adam",
                 init="glorot", random_state=42):
        self.hidden_layers = list(hidden_layers)
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.init = init
        self.rng = np.random.default_rng(random_state)

        # mêmes valeurs par défaut que Keras
        self.bn_momentum = 0.99
        self.bn_eps = 1e-3
        self.beta1, self.beta2, self.adam_eps = 0.9, 0.999, 1e-7

        self.layers = []          # une liste de dictionnaires, un par couche
        self.history = {}
        self.grad_history = []    # moyenne de |dW| par couche, pour chaque époque

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self, n_in, n_out):
        if self.init == "he":
            return self.rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out))
        limit = np.sqrt(6 / (n_in + n_out))                 # glorot uniforme
        return self.rng.uniform(-limit, limit, size=(n_in, n_out))

    def _build(self, n_features, n_classes):
        sizes = [n_features] + self.hidden_layers + [n_classes]
        self.layers = []
        for i in range(len(sizes) - 1):
            layer = {"W": self._init_weights(sizes[i], sizes[i + 1]),
                     "b": np.zeros((1, sizes[i + 1])),
                     "is_output": i == len(sizes) - 2}
            if self.batch_norm and not layer["is_output"]:
                layer["gamma"] = np.ones((1, sizes[i + 1]))
                layer["beta"] = np.zeros((1, sizes[i + 1]))
                layer["running_mean"] = np.zeros((1, sizes[i + 1]))
                layer["running_var"] = np.ones((1, sizes[i + 1]))
            self.layers.append(layer)

        # état de l'optimiseur Adam : une moyenne m et une variance v par paramètre
        self.t = 0
        for layer in self.layers:
            for name in self._param_names(layer):
                layer["m_" + name] = np.zeros_like(layer[name])
                layer["v_" + name] = np.zeros_like(layer[name])

    @staticmethod
    def _param_names(layer):
        return ["W", "b"] + (["gamma", "beta"] if "gamma" in layer else [])

    # ------------------------------------------------------------------
    # Fonctions d'activation et leurs dérivées
    # ------------------------------------------------------------------
    def _activate(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        raise ValueError(f"Activation inconnue : {self.activation}")

    def _activate_derivative(self, z, a):
        if self.activation == "relu":
            return (z > 0).astype(float)
        if self.activation == "tanh":
            return 1 - a ** 2
        return a * (1 - a)                                   # sigmoid

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)                 # évite les très grands exponentiels
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Propagation avant
    # ------------------------------------------------------------------
    def _forward(self, X, training=False):
        A = X
        for layer in self.layers:
            layer["A_prev"] = A
            Z = A @ layer["W"] + layer["b"]

            if layer["is_output"]:
                layer["A"] = self._softmax(Z)
                return layer["A"]

            if "gamma" in layer:                             # batch normalization
                if training:
                    mean, var = Z.mean(axis=0, keepdims=True), Z.var(axis=0, keepdims=True)
                    m = self.bn_momentum
                    layer["running_mean"] = m * layer["running_mean"] + (1 - m) * mean
                    layer["running_var"] = m * layer["running_var"] + (1 - m) * var
                else:
                    mean, var = layer["running_mean"], layer["running_var"]
                layer["inv_std"] = 1 / np.sqrt(var + self.bn_eps)
                layer["Z_hat"] = (Z - mean) * layer["inv_std"]
                Z = layer["gamma"] * layer["Z_hat"] + layer["beta"]

            layer["Z"] = Z
            A = self._activate(Z)
            layer["A_no_drop"] = A

            if training and self.dropout > 0:                # dropout "inversé", comme Keras
                layer["mask"] = (self.rng.random(A.shape) >= self.dropout) / (1 - self.dropout)
                A = A * layer["mask"]
            else:
                layer["mask"] = None
            layer["A"] = A
        return A

    # ------------------------------------------------------------------
    # Rétropropagation
    # ------------------------------------------------------------------
    def _backward(self, Y):
        n = Y.shape[0]
        dZ = (self.layers[-1]["A"] - Y) / n                  # softmax + cross-entropy

        for layer in reversed(self.layers):
            if not layer["is_output"]:
                dA = dZ
                if layer["mask"] is not None:
                    dA = dA * layer["mask"]
                dZ = dA * self._activate_derivative(layer["Z"], layer["A_no_drop"])

                if "gamma" in layer:                         # retour à travers la batch norm
                    Z_hat, inv_std = layer["Z_hat"], layer["inv_std"]
                    layer["d_gamma"] = (dZ * Z_hat).sum(axis=0, keepdims=True)
                    layer["d_beta"] = dZ.sum(axis=0, keepdims=True)
                    dZ_hat = dZ * layer["gamma"]
                    dZ = inv_std / n * (n * dZ_hat - dZ_hat.sum(axis=0, keepdims=True)
                                        - Z_hat * (dZ_hat * Z_hat).sum(axis=0, keepdims=True))

            layer["d_W"] = layer["A_prev"].T @ dZ
            layer["d_b"] = dZ.sum(axis=0, keepdims=True)
            dZ = dZ @ layer["W"].T                           # gradient envoyé à la couche d'avant

    # ------------------------------------------------------------------
    # Mise à jour des poids
    # ------------------------------------------------------------------
    def _update(self):
        self.t += 1
        lr = self.learning_rate
        for layer in self.layers:
            for name in self._param_names(layer):
                grad = layer["d_" + name]
                if self.optimizer == "sgd":
                    layer[name] -= lr * grad
                else:                                         # adam
                    layer["m_" + name] = self.beta1 * layer["m_" + name] + (1 - self.beta1) * grad
                    layer["v_" + name] = self.beta2 * layer["v_" + name] + (1 - self.beta2) * grad ** 2
                    m_hat = layer["m_" + name] / (1 - self.beta1 ** self.t)
                    v_hat = layer["v_" + name] / (1 - self.beta2 ** self.t)
                    layer[name] -= lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)

    # ------------------------------------------------------------------
    # Outils
    # ------------------------------------------------------------------
    @staticmethod
    def _one_hot(y, n_classes):
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1
        return Y

    @staticmethod
    def _cross_entropy(probas, Y):
        return -np.mean(np.sum(Y * np.log(np.clip(probas, 1e-7, 1)), axis=1))

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=0):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        self.n_classes = int(y.max()) + 1
        self._build(X.shape[1], self.n_classes)
        Y = self._one_hot(y, self.n_classes)

        self.history = {"loss": [], "accuracy": []}
        if X_val is not None:
            self.history.update({"val_loss": [], "val_accuracy": []})
        self.grad_history = []

        for epoch in range(epochs):
            order = self.rng.permutation(len(X))             # on mélange à chaque époque
            losses, accs, sizes = [], [], []
            grads = np.zeros(len(self.layers))
            n_batches = 0

            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                probas = self._forward(X[idx], training=True)
                self._backward(Y[idx])
                self._update()

                losses.append(self._cross_entropy(probas, Y[idx]))
                accs.append(np.mean(probas.argmax(axis=1) == y[idx]))
                sizes.append(len(idx))
                grads += [np.mean(np.abs(layer["d_W"])) for layer in self.layers]
                n_batches += 1

            # comme Keras : moyenne des batchs de l'époque, pondérée par leur taille
            self.history["loss"].append(np.average(losses, weights=sizes))
            self.history["accuracy"].append(np.average(accs, weights=sizes))
            self.grad_history.append(grads / n_batches)

            if X_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

            if verbose and (epoch + 1) % verbose == 0:
                msg = f"Époque {epoch + 1}/{epochs} - loss {self.history['loss'][-1]:.4f}"
                if X_val is not None:
                    msg += f" - val_loss {val_loss:.4f} - val_accuracy {val_acc:.4f}"
                print(msg)
        return self

    # ------------------------------------------------------------------
    # Prédiction et évaluation
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float), training=False)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        y = np.asarray(y, dtype=int)
        probas = self.predict_proba(X)
        loss = self._cross_entropy(probas, self._one_hot(y, self.n_classes))
        return loss, np.mean(probas.argmax(axis=1) == y)

    def count_params(self):
        return sum(layer[name].size for layer in self.layers for name in self._param_names(layer))
