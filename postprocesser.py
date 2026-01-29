from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import utils


class PostProcessor(ABC):
    """ABSTRACT CLASS"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass


class LabelProjection(PostProcessor):
    """Supervised projection for Classification tasks"""

    def fit(self, X, y):
        self.origin = X.mean(axis=0)
        self.axes = []
        unique_classes = np.unique(y)
        # print(f"[DEBUG] LabelProjection fit: Classes found: {unique_classes}")
        for c in unique_classes:
            cls = X[y == c]
            if len(cls) > 0:
                self.axes.append(cls.mean(axis=0) - self.origin)

        self.axes = np.array(self.axes)
        if self.axes.shape[0] == 0:
            print("[WARN] LabelProjection: No axes found! Fallback to random/identity?")
            # Fallback to avoid crash: just use random direction or identity
            self.axes = np.random.randn(1, X.shape[1])
        # print(f"[DEBUG] LabelProjection axes shape: {self.axes.shape}")

    def transform(self, X):
        return utils.embedding_to_proj(X, self.origin, self.axes)


class LDA(PostProcessor):
    """Statistical compression for supervised tasks"""

    def __init__(self, target_dim=None):
        self.lda = LinearDiscriminantAnalysis(n_components=target_dim)

    def fit(self, X, y):
        self.lda.fit(X, y)

    def transform(self, X):
        return self.lda.transform(X)


class PCA_PP(PostProcessor):
    """Statistical compression for general unsupervised tasks"""

    def __init__(self, target_dim=64):
        self.target_dim = target_dim
        self.pca = None

    def fit(self, X, y=None):
        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(X)
        # pass

    def transform(self, X):
        return self.pca.transform(X)


class PCA_ABTT(PostProcessor):
    """Statistical compression for general unsupervised tasks"""

    def __init__(self, D=5, target_dim=64):
        self.D = D
        self.target_dim = target_dim
        self.pca = None
        self.mean = None
        self.is_fitted = False

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_centered /= norms

        n_samples = X_centered.shape[0]
        n_features = X_centered.shape[1]
        n_components = min(self.target_dim, n_samples, n_features)

        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_centered)
        self.is_fitted = True

    def transform(self, X):
        if not self.is_fitted:
            return X

        X_centered = X - self.mean
        norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_centered /= norms

        X_pca = self.pca.transform(X_centered)

        return utils.all_but_the_top(X_pca)


# class AdaptivePostProcessor(PostProcessor):
#     """
#     Learned projection layer
#     Preserves semantic manifold via Relational Distillation
#     """

#     def __init__(self, target_dim=64, epochs=10):
#         self.target_dim = target_dim
#         self.epochs = epochs
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.proj = None | nn.Sequential

#     def fit(self, X, y=None):
#         input_dim = X.shape[1]
#         X_train = torch.from_numpy(X).float().to(self.device)

#         self.proj = nn.Sequential(
#             nn.Linear(input_dim, self.target_dim, bias=False),
#             nn.LayerNorm(self.target_dim),
#         ).to(self.device)

#         opt = optim.AdamW(self.proj.parameters(), lr=1e-3, weight_decay=0.01)

#         self.proj.train()
#         for _ in range(self.epochs):
#             X_norm = X_train / X_train.norm(dim=1, keepdim=True)
#             teacher_sim = X_norm @ X_norm.T

#             projected = self.proj(X_train)
#             projected_norm = projected / projected.norm(dim=1, keepdim=True)
#             student_sim = projected_norm @ projected_norm.T

#             loss = nn.functional.mse_loss(student_sim, teacher_sim)

#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#         # print(f"[ADAPTIVE] Fit complete. Final Distillation Loss: {loss.item():.6f}")

#     def transform(self, X):
#         self.proj.eval()
#         with torch.no_grad():
#             X_tensor = torch.from_numpy(X).float().to(self.device)
#             return self.proj(X_tensor).cpu().numpy()


class No_Train_Projection(PostProcessor):
    """
    Doesn't train the projection layer.
    Just create a projection layer and evaluate.
    """

    def __init__(self, target_dim=64, epochs=10, batch_size=128):
        self.target_dim = target_dim
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proj = None | nn.Sequential
        self.batch_size = batch_size

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        # X_train = torch.from_numpy(X).float().to(self.device)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.target_dim, bias=False),
            nn.LayerNorm(self.target_dim),
        ).to(self.device)

        # optimizer = optim.AdamW(self.proj.parameters(), lr=1e-3)
        # self.proj.train()
        # for _ in range(self.epochs):
        #     total_rel_loss = 0

        #     for (batch_X,) in dataloader:
        #         batch_X = batch_X.to(self.device)

        #         projected = self.proj(batch_X)

        #         X_norm = batch_X / (batch_X.norm(dim=1, keepdim=True) + 1e-8)
        #         P_norm = projected / (projected.norm(dim=1, keepdim=True) + 1e-8)

        #         # Similarity matrices
        #         input_sim = X_norm @ X_norm.T
        #         output_sim = P_norm @ P_norm.T

        #         loss = nn.functional.mse_loss(output_sim, input_sim)

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         total_rel_loss += loss.item()

        # print(
        #     f"[LOG] Fit complete. AVG. Distillation Loss: {total_rel_loss / self.epochs:.6f}"
        # )

    def transform(self, X):
        self.proj.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            return self.proj(X_tensor).cpu().numpy()


class AdaptivePostProcessor(PostProcessor):
    """
    Learned projection layer
    Preserves semantic manifold via Relational Distillation
    """

    def __init__(self, target_dim=64, epochs=10, batch_size=128):
        self.target_dim = target_dim
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proj = None | nn.Sequential
        self.batch_size = batch_size

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        # X_train = torch.from_numpy(X).float().to(self.device)

        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.target_dim, bias=False),
            nn.LayerNorm(self.target_dim),
        ).to(self.device)

        optimizer = optim.AdamW(self.proj.parameters(), lr=1e-3)
        self.proj.train()
        for _ in range(self.epochs):
            total_rel_loss = 0

            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)

                projected = self.proj(batch_X)

                X_norm = batch_X / (batch_X.norm(dim=1, keepdim=True) + 1e-8)
                P_norm = projected / (projected.norm(dim=1, keepdim=True) + 1e-8)

                # Similarity matrices
                input_sim = X_norm @ X_norm.T
                output_sim = P_norm @ P_norm.T

                loss = nn.functional.mse_loss(output_sim, input_sim)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_rel_loss += loss.item()

        # for _ in range(self.epochs):
        #     X_norm = X_train / X_train.norm(dim=1, keepdim=True)
        #     input_sim = X_norm @ X_norm.T

        #     projected = self.proj(X_train)
        #     projected_norm = projected / projected.norm(dim=1, keepdim=True)
        #     output_sim = projected_norm @ projected_norm.T

        #     # Distillation Loss (MSE between similarity matrices)
        #     loss = nn.functional.mse_loss(output_sim, input_sim)

        #     opt.zero_grad()
        #     loss.backward()
        #     opt.step()

        # print(f"[LOG] Fit complete. Final Distillation Loss: {loss.item():.6f}")
        print(
            f"[LOG] Fit complete. AVG. Distillation Loss: {total_rel_loss / self.epochs:.6f}"
        )

    def transform(self, X):
        self.proj.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            return self.proj(X_tensor).cpu().numpy()


class DisentangledAdaptivePostProcessor(PostProcessor):
    """
    Learned projection layer
    Preserves semantic manifold via Relational Distillation
    + Covariance Disentanglement
    """

    def __init__(
        self, target_dim=64, epochs=10, batch_size=128, disentangle_weight=0.1
    ):
        self.target_dim = target_dim
        self.epochs = epochs
        self.beta = disentangle_weight  # Strength of disentanglement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.proj = None | nn.Sequential

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        # Old code
        # X_train = torch.from_numpy(X).float().to(self.device)

        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.target_dim, bias=False),
            nn.LayerNorm(self.target_dim),
        ).to(self.device)

        optimizer = optim.AdamW(self.proj.parameters(), lr=1e-3)

        self.proj.train()

        # Need to do batching size of loss calculation reach 9GB of memory
        for _ in range(self.epochs):
            total_rel_loss = 0
            total_dis_loss = 0

            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)

                projected = self.proj(batch_X)

                X_norm = batch_X / (batch_X.norm(dim=1, keepdim=True) + 1e-8)
                P_norm = projected / (projected.norm(dim=1, keepdim=True) + 1e-8)

                input_sim = X_norm @ X_norm.T
                output_sim = P_norm @ P_norm.T

                rel_loss = nn.functional.mse_loss(output_sim, input_sim)

                S_centered = projected - projected.mean(dim=0)
                cov_matrix = (S_centered.T @ S_centered) / (self.batch_size - 1)
                target_identity = torch.eye(self.target_dim, device=self.device)

                disentangle_loss = nn.functional.mse_loss(cov_matrix, target_identity)

                loss = rel_loss + (self.beta * disentangle_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_rel_loss += rel_loss.item()
                total_dis_loss += disentangle_loss.item()

        # print(
        #     f"[LOG] Fit complete. AVG. Distillation Loss: {total_rel_loss / self.epochs:.6f}, AVG. Disentanglement Loss: {total_dis_loss / self.epochs:.6f}"
        # )

    def transform(self, X, batch_size=None):
        if self.proj is None:
            raise RuntimeError("Model not fitted yet.")
        self.proj.eval()
        bs = batch_size if batch_size else self.batch_size
        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

        outputs = []
        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)
                out = self.proj(batch_X)
                outputs.append(out.cpu().numpy())

        return np.vstack(outputs)
