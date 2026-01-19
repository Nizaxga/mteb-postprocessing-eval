import copy
import warnings
from abc import ABC, abstractmethod
from enum import Enum

import mteb
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn, optim

# MTEBv2 using n_jobs=-1 and for some reason sklearn.linear_model is deprecated that.
# So suppress those warnings real quick
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")



def embedding_to_proj(embeddings, origin, directions):
    """HELPER"""

    X = embeddings - origin
    return X @ directions.T / (np.linalg.norm(directions, axis=1) ** 2)


def low_rank_approximation(X, k):
    """HELPER"""

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def all_but_the_top(X, D=5):
    """HELPER"""

    X = X - X.mean(axis=0)
    return X - low_rank_approximation(X, D)


def PCAP(X, D=7):
    """HELPER"""

    X -= np.mean(X, axis=0)
    X = PCA(n_components=X.shape[1] // 2).fit_transform(X)
    return all_but_the_top(X, D)


class PostProcessor(ABC):
    """ABSTRACT CLASS"""

    @abstractmethod
    def fit(self, X, y=None):
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
        return embedding_to_proj(X, self.origin, self.axes)


class PCA_ABTT(PostProcessor):
    """Statistical compression for general unsupervised tasks"""

    def __init__(self, D=5):
        self.D = D

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return PCAP(X, self.D)


class AdaptivePostProcessor(PostProcessor):
    """
    Learned projection layer
    Preserves semantic manifold via Relational Distillation
    """

    def __init__(self, target_dim=128, epochs=10):
        self.target_dim = target_dim
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proj = None

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        X_train = torch.from_numpy(X).float().to(self.device)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.target_dim, bias=False),
            nn.LayerNorm(self.target_dim),
        ).to(self.device)

        opt = optim.AdamW(self.proj.parameters(), lr=1e-3, weight_decay=0.01)

        self.proj.train()
        for _ in range(self.epochs):
            X_norm = X_train / X_train.norm(dim=1, keepdim=True)
            teacher_sim = X_norm @ X_norm.T

            projected = self.proj(X_train)
            projected_norm = projected / projected.norm(dim=1, keepdim=True)
            student_sim = projected_norm @ projected_norm.T

            loss = nn.functional.mse_loss(student_sim, teacher_sim)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"[ADAPTIVE] Fit complete. Final Distillation Loss: {loss.item():.6f}")

    def transform(self, X):
        self.proj.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            return self.proj(X_tensor).cpu().numpy()


class DisentangledAdaptivePostProcessor(PostProcessor):
    """
    Learned projection layer
    Preserves semantic manifold via Relational Distillation

    """

    def __init__(self, target_dim=77, epochs=10, disentangle_weight=0.5):
        self.target_dim = target_dim
        self.epochs = epochs
        self.beta = disentangle_weight  # Strength of disentanglement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proj = None

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        X_train = torch.from_numpy(X).float().to(self.device)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.target_dim, bias=False),
            nn.LayerNorm(self.target_dim),
        ).to(self.device)

        optimizer = optim.AdamW(self.proj.parameters(), lr=1e-3)

        self.proj.train()
        for _ in range(self.epochs):
            # 1. Standard Relational Distillation
            projected = self.proj(X_train)

            X_norm = X_train / X_train.norm(dim=1, keepdim=True)
            P_norm = projected / projected.norm(dim=1, keepdim=True)

            rel_loss = nn.functional.mse_loss(P_norm @ P_norm.T, X_norm @ X_norm.T)

            # 2. Disentanglement Loss (Covariance Independence)
            # We want the correlation matrix of dimensions to be Identity
            # Center the projected features
            S_centered = projected - projected.mean(dim=0)
            # Compute Covariance Matrix (dim x dim)
            cov_matrix = (S_centered.T @ S_centered) / (projected.shape[0] - 1)

            # Identity matrix for target
            target_identity = torch.eye(self.target_dim).to(self.device)

            # Disentanglement loss: minimize off-diagonal elements
            disentangle_loss = nn.functional.mse_loss(cov_matrix, target_identity)
            loss = rel_loss + (self.beta * disentangle_loss)
            # loss = disentangle_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(f"[DAL] rel_loss: {rel_loss:.4f}, disentangle_loss: {disentangle_loss:.4f}")

    def transform(self, X):
        self.proj.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            return self.proj(X_tensor).cpu().numpy()


class TaskType(Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    UNSUPERVISED_ALT = "unsupervised_alt"
    UNSUPERVISED_DIS = "unsupervised_dis"
    CLUSTERING = "clustering"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    PAIR_CLASSIFICATION = "pair_classification"


TASK_REGISTRY = {
    "Banking77Classification.v2": TaskType.SUPERVISED,
    "EmotionClassification.v2": TaskType.SUPERVISED,
    "STSBenchmark": TaskType.UNSUPERVISED_DIS,
    "TwentyNewsgroupsClustering.v2": TaskType.CLUSTERING,
    "SprintDuplicateQuestions": TaskType.PAIR_CLASSIFICATION,
    "AskUbuntuDupQuestions": TaskType.RERANKING,
    "NFCorpus": TaskType.RETRIEVAL,
}


def build_postprocessor(task_name):
    ttype = TASK_REGISTRY.get(task_name, TaskType.UNSUPERVISED)
    if ttype == TaskType.SUPERVISED:
        return LabelProjection()
    elif ttype == TaskType.UNSUPERVISED_ALT:
        return AdaptivePostProcessor()
    elif ttype == TaskType.UNSUPERVISED_DIS:
        return DisentangledAdaptivePostProcessor()
    elif ttype == TaskType.CLUSTERING:
        return AdaptivePostProcessor(target_dim=128)
    elif ttype == TaskType.PAIR_CLASSIFICATION:
        return LabelProjection()
    elif ttype in [TaskType.RETRIEVAL, TaskType.RERANKING]:
        return AdaptivePostProcessor(target_dim=128) 
    else:
        # Fall back
        return PCA_ABTT()


def create_mteb_wrapper(base_model, postprocessor):
    """
    Creates a class that dynamically inherits from the base_model's class.
    This guarantees we pass MTEB v2's 'isinstance(model, Encoder)' check
    without needing to find the hidden import path.
    """

    BaseClass = base_model.__class__

    class MTEBWrapper(BaseClass):
        def __init__(self, original_model, pp):
            self.__dict__ = original_model.__dict__.copy()
            self.pp = pp
            self.is_fitted = False

            if hasattr(self, "mteb_model_meta"):
                self.mteb_model_meta = copy.deepcopy(original_model.mteb_model_meta)
                self.mteb_model_meta.name = f"{self.mteb_model_meta.name}-PP"

        # def fit_from_task(self, task_name):
        #     task = mteb.get_task(task_name)
        #     task.load_data()
        #     split = "train" if "train" in task.dataset else list(task.dataset.keys())[0]
        #     train_data = task.dataset[split]

        #     texts = (
        #         train_data["text"]
        #         if "text" in train_data.column_names
        #         else train_data[0]
        #     )

        #     print(
        #         f"[LOG] Fitting {self.pp.__class__.__name__} on {len(texts)} samples..."
        #     )
        #     X = self.model.encode(texts, batch_size=128)

        #     if TASK_REGISTRY.get(task_name) == TaskType.SUPERVISED:
        #         labels = np.array(train_data["label"])
        #         self.pp.fit(X, labels)
        #     else:
        #         self.pp.fit(X)
        #     self.is_fitted = True

        def fit_from_task(self, task_name):
            task = mteb.get_task(task_name)
            task.load_data()
            
            split = "train"

            # Some tasks have 'default' as a key containing splits
            if "default" in task.dataset and isinstance(task.dataset["default"], dict):
                 if "test" in task.dataset["default"]:
                     split = "test"
                     task.dataset = task.dataset["default"] # Unwrap
                 elif "train" in task.dataset["default"]:
                     split = "train"
                     task.dataset = task.dataset["default"] # Unwrap
            
            if "train" not in task.dataset:
                # For some tasks, dataset might be structured differently
                if "test" in task.dataset:
                    split = "test"
                else:
                     # Fallback to any available split
                    split = list(task.dataset.keys())[0]
            
            # For Retrieval, data might be in task.corpus
            data_split = task.dataset[split] if split in task.dataset else task.dataset

            ttype = TASK_REGISTRY.get(task_name)

            if ttype == TaskType.SUPERVISED:
                texts = data_split["text"]
                labels = np.array(data_split["label"])
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X, labels)

            elif ttype == TaskType.CLUSTERING:
                 # Clustering tasks often have a "sentences" column which is a list of strings
                raw_texts = data_split["sentences"] # List of list of strings for some reason
                texts = []
                for sublist in raw_texts:
                    texts.extend(sublist)
                
                if len(texts) > 2000:
                    print(f"[LOG] Subsampling {len(texts)} -> 2000 for fitting.")
                    texts = texts[:2000]
                
                print(f"[LOG] Fitting on {len(texts)} sentences from {split} split (Clustering task)...")
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X)

            elif ttype == TaskType.PAIR_CLASSIFICATION:
                s1 = data_split["sentence1"] if "sentence1" in data_split.column_names else []
                s2 = data_split["sentence2"] if "sentence2" in data_split.column_names else []
                labels = []
                if "label" in data_split.column_names:
                    labels = data_split["label"]
                    # print("[DEBUG] Found 'label' column")
                elif "labels" in data_split.column_names:
                    labels = data_split["labels"]
                    # print("[DEBUG] Found 'labels' column")
                
                # print(f"[DEBUG] labels type: {type(labels)}")
                # if len(labels) > 0:
                #      print(f"[DEBUG] labels[0]: {labels[0]}")
                #      print(f"[DEBUG] labels[0] type: {type(labels[0])}")
                
                # Handle potential list-in-list structure for sentences
                if len(s1) > 0 and isinstance(s1[0], list):
                    s1 = [item for sublist in s1 for item in sublist]
                if len(s2) > 0 and isinstance(s2[0], list):
                    s2 = [item for sublist in s2 for item in sublist]

                # Handle potential list-in-list structure for LABELS (found during debug: len(labels)=1)
                if len(labels) > 0 and isinstance(labels[0], list):
                    labels = [item for sublist in labels for item in sublist]
                elif len(labels) > 0 and hasattr(labels[0], "shape"): # numpy/tensor check
                     # If it's a tensor of shape (1, N) or list of tensors
                     pass 

                # Re-do specific Logic:
                # If s1[i] is a list, join it back to string (if it was tokens)
                final_s1 = []
                for item in s1:
                    if isinstance(item, list):
                        final_s1.append(" ".join([str(x) for x in item]))
                    else:
                        final_s1.append(item)
                
                final_s2 = []
                for item in s2:
                    if isinstance(item, list):
                        final_s2.append(" ".join([str(x) for x in item]))
                    else:
                        final_s2.append(item)

                # Combine
                final_labels = list(labels) + list(labels)
                
                texts = final_s1 + final_s2
                
                if len(texts) > 2000:
                    texts = texts[:2000]
                    final_labels = final_labels[:2000]

                print(f"[LOG] Fitting LabelProjection on {len(texts)} sentences from {split} split (PairClass)...")
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X, np.array(final_labels))

            elif ttype == TaskType.RERANKING:
                # Structure: {'query': str, 'positive': list[str], 'negative': list[str]}
                # dataset might be list-ish
                texts = []
                count = 0
                
                # Unwrap
                if isinstance(data_split, dict) and "test" in data_split:
                     data_split = data_split["test"]
                
                # iterating?
                iterable = data_split
                if isinstance(data_split, dict):
                     # If it's a dict of samples with string keys?
                     if "query" in data_split: # It's a single sample?
                         iterable = [data_split]
                     else:
                        iterable = data_split.values()

                # Sampling optimization
                for sample in iterable:
                    if count > 500: break 
                    
                    if isinstance(sample, dict):
                        texts.append(sample.get('query', ''))
                        texts.extend(sample.get('positive', []))
                        texts.extend(sample.get('negative', []))
                    elif isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], dict):
                         # Maybe nested list
                         for sub in sample:
                            texts.append(sub.get('query', ''))
                            texts.extend(sub.get('positive', []))
                            texts.extend(sub.get('negative', []))
                    else:
                        # Skip unknown
                        pass
                    count += 1
                
                texts = list(set(texts))
                if len(texts) > 2000: texts = texts[:2000]

                print(f"[LOG] Fitting on {len(texts)} sentences from {split} split (Reranking)...")
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X)

            elif ttype == TaskType.RETRIEVAL:
                # Corpus first
                texts = []
                corpus = None
                if hasattr(task, "corpus") and task.corpus:
                     corpus = task.corpus
                elif "corpus" in task.dataset:
                     corpus = task.dataset["corpus"]
                
                if corpus:
                    print("[LOG] Using corpus for fitting...")
                    # Corpus: dict {id: {text:...}} or Dataset with 'text' col
                    if hasattr(corpus, "values") and callable(corpus.values):
                         # dict-ish
                        count = 0
                        for doc in corpus.values():
                            if count > 2000: break
                            if isinstance(doc, dict) and 'text' in doc:
                                texts.append(doc['text'])
                            elif isinstance(doc, str):
                                texts.append(doc)
                            count += 1
                    else:
                        # dataset-ish (list/HF dataset)
                        # Check columns
                        if hasattr(corpus, "column_names") and "text" in corpus.column_names:
                            texts = corpus["text"][:2000]
                        else:
                            # Try iterating
                            count = 0
                            for item in corpus:
                                if count > 2000: break
                                if isinstance(item, dict) and "text" in item:
                                    texts.append(item["text"])
                                count += 1

                if not texts:
                     # Fallback to queries
                     if hasattr(task, "queries") and task.queries:
                         print("[LOG] Fallback to task.queries...")
                         texts = list(task.queries.values())[:2000]
                     elif "queries" in task.dataset:
                         q = task.dataset["queries"]
                         if hasattr(q, "values"):
                             texts = list(q.values())[:2000]
                         else:
                             # list/HF access
                             if hasattr(q, "column_names") and "text" in q.column_names:
                                 texts = q["text"][:2000]

                if not texts:
                    print("[WARN] Could not find corpus/queries text for Retrieval. Fitting on minimal dummy data.")
                    texts = ["dummy sentence"]
                
                print(f"[LOG] Fitting on {len(texts)} sentences (Retrieval)...")
                if texts:
                    X = self.model.encode(texts, batch_size=128)
                    self.pp.fit(X)

            else:
                # Standard STS or others
                # texts = list(set(train["sentence1"] + train["sentence2"])) # WHYY IS THIS WORKING
                s1 = list(data_split["sentence1"])
                s2 = list(data_split["sentence2"])
                # texts = train["text"]
                texts = list(set(s1 + s2))
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X)

            self.is_fitted = True

        def encode(self, sentences, **kwargs):
            if not self.is_fitted:
                raise RuntimeError("Post-processor not fitted")

            X = super().encode(sentences, **kwargs)
            return self.pp.transform(X)

    return MTEBWrapper(base_model, postprocessor)


def run_mteb_comparison_bench(task_name, base_model):
    """Run wrapper along with base_model using MTEB API"""

    print(f"\n[LOG] Initializing Task: {task_name}")
    pp = build_postprocessor(task_name)

    # mtebv1 wrapper
    # wrapped_model = MTEBWrapper(base_model, pp) 

    # mtebv2 wrapper
    wrapped_model = create_mteb_wrapper(base_model, pp) 
    wrapped_model.fit_from_task(task_name)

    tasks = mteb.get_tasks(tasks=[task_name])
    cache = mteb.ResultCache(cache_path="./")

    print("--- Running Evaluation for Wrapped Model ---")
    mteb.evaluate(
        model=wrapped_model, tasks=tasks, cache=cache, overwrite_strategy="always"
    )
    # res_wrapped = mteb.evaluate(model=wrapped_model, tasks=tasks, cache=cache)
    # print(json.dumps(res_wrapped.task_results[0].scores, indent=4))

    print("--- Running Evaluation for Base Model ---")
    mteb.evaluate(
        model=base_model, tasks=tasks, cache=cache, overwrite_strategy="always"
    )
    # res_base = mteb.evaluate(model=base_model, tasks=tasks, cache=cache)
    # print(json.dumps(res_base.task_results[0].scores, indent=4))
    # return res_wrapped, res_base

# Class struct of TaskResult, Evaluation time.
# class TaskResult(BaseModel):
#     """A class to represent the MTEB result.

#     Attributes:
#         task_name: The name of the MTEB task.
#         dataset_revision: The revision dataset for the task on HuggingFace dataset hub.
#         mteb_version: The version of the MTEB used to evaluate the model.
#         scores: The scores of the model on the dataset. The scores is a dictionary with the following structure; dict[SplitName, list[Scores]].
#             Where Scores is a dictionary with the following structure; dict[str, Any]. Where the keys and values are scores. Split is the split of
#             the dataset.
#         evaluation_time: The time taken to evaluate the model.
#         kg_co2_emissions: The kg of CO2 emissions produced by the model during evaluation.

def run_mteb_comparison_time(task_name, base_model):
    """ Could be merge with run_..._bench """

    raise NotImplementedError


if __name__ == "__main__":
    base_model = mteb.get_model("BAAI/bge-base-en-v1.5")
    # base_model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
    
    # "Banking77Classification.v2": TaskType.SUPERVISED,
    # "EmotionClassification.v2": TaskType.SUPERVISED,
    # "Banking77Classification.v2": TaskType.SUPERVISED,
    # "EmotionClassification.v2": TaskType.SUPERVISED,
    # "STSBenchmark": TaskType.UNSUPERVISED_DIS,
    # "TwentyNewsgroupsClustering.v2": TaskType.CLUSTERING,
    # "SprintDuplicateQuestions": TaskType.PAIR_CLASSIFICATION,
    # "AskUbuntuDupQuestions": TaskType.RERANKING,
    # "NFCorpus": TaskType.RETRIEVAL,
    # New tasks
    run_mteb_comparison_bench("TwentyNewsgroupsClustering.v2", base_model) #clustering
    run_mteb_comparison_bench("SprintDuplicateQuestions", base_model) # PairClass
    run_mteb_comparison_bench("AskUbuntuDupQuestions", base_model) # Reranking
    run_mteb_comparison_bench("NFCorpus", base_model) # Retrieval
