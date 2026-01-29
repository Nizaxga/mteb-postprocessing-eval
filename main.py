import copy
import os
import warnings
from enum import Enum

import mteb
import numpy as np

import postprocesser

# MTEBv2 using n_jobs=-1 and for some reason sklearn.linear_model is deprecated that parameter.
# So suppress those warnings real quick
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")
DIR = "./PCA_ABTT_64"


class TaskType(Enum):
    CLASSIFICATION = "classification task"
    CLUSTERING = "clustering"
    STS = "sts"
    PAIR_CLASSIFICATION = "pair_classification"
    RETRIEVAL = "retrieval"
    FALLBACK = "none"


TASK_REGISTRY = {
    "Banking77Classification.v2": TaskType.CLASSIFICATION,
    "EmotionClassification.v2": TaskType.CLASSIFICATION,
    "SprintDuplicateQuestions": TaskType.PAIR_CLASSIFICATION,
    "TwentyNewsgroupsClustering.v2": TaskType.CLUSTERING,
    "STSBenchmark": TaskType.STS,
    "NFCorpus": TaskType.RETRIEVAL,
}


def build_postprocessor(task_name):
    ttype = TASK_REGISTRY.get(task_name, TaskType.FALLBACK)
    if ttype == TaskType.FALLBACK:
        return postprocesser.PCA_PP()
    elif (
        ttype == TaskType.CLASSIFICATION
        or ttype == TaskType.CLUSTERING
        or ttype == TaskType.PAIR_CLASSIFICATION
    ):
        return postprocesser.LDA()
    elif ttype == TaskType.RETRIEVAL or ttype == TaskType.STS:
        return postprocesser.DisentangledAdaptivePostProcessor()


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
                self.mteb_model_meta.name = (
                    f"{self.mteb_model_meta.name}-{self.pp.__class__.__name__}"
                )

        def fit_from_task(self, task_name):
            task = mteb.get_task(task_name)
            task.load_data()

            split = "train"

            # Some tasks have 'default' as a key containing splits
            if "default" in task.dataset and isinstance(task.dataset["default"], dict):
                if "test" in task.dataset["default"]:
                    split = "test"
                    task.dataset = task.dataset["default"]  # Unwrap
                elif "train" in task.dataset["default"]:
                    split = "train"
                    task.dataset = task.dataset["default"]  # Unwrap

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

            if ttype == TaskType.CLASSIFICATION:
                texts = data_split["text"]
                labels = np.array(data_split["label"])
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X, labels)

            elif ttype == TaskType.CLUSTERING:
                # Clustering tasks often have a "sentences" column which is a list of strings
                raw_texts = data_split[
                    "sentences"
                ]  # List of list of strings for some reason
                texts = []
                for sublist in raw_texts:
                    texts.extend(sublist)

                if len(texts) > 2000:
                    # print(f"[LOG] Subsampling {len(texts)} -> 2000 for fitting.")
                    texts = texts[:2000]

                print(
                    # f"[LOG] Fitting on {len(texts)} sentences from {split} split (Clustering task)..."
                )
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X)

            elif ttype == TaskType.PAIR_CLASSIFICATION:
                s1 = (
                    data_split["sentence1"]
                    if "sentence1" in data_split.column_names
                    else []
                )
                s2 = (
                    data_split["sentence2"]
                    if "sentence2" in data_split.column_names
                    else []
                )
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
                elif len(labels) > 0 and hasattr(
                    labels[0], "shape"
                ):  # numpy/tensor check
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

                print(
                    # f"[LOG] Fitting LabelProjection on {len(texts)} sentences from {split} split (PairClass)..."
                )
                X = self.model.encode(texts, batch_size=128)
                self.pp.fit(X, np.array(final_labels))

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
                            if count > 2000:
                                break
                            if isinstance(doc, dict) and "text" in doc:
                                texts.append(doc["text"])
                            elif isinstance(doc, str):
                                texts.append(doc)
                            count += 1
                    else:
                        # dataset-ish (list/HF dataset)
                        # Check columns
                        if (
                            hasattr(corpus, "column_names")
                            and "text" in corpus.column_names
                        ):
                            texts = corpus["text"][:2000]
                        else:
                            # Try iterating
                            count = 0
                            for item in corpus:
                                if count > 2000:
                                    break
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
                    print(
                        "[WARN] Could not find corpus/queries text for Retrieval. Fitting on minimal dummy data."
                    )
                    texts = ["dummy sentence"]

                print(f"[LOG] Fitting on {len(texts)} sentences (Retrieval)...")
                if texts:
                    X = self.model.encode(texts, batch_size=128)
                    self.pp.fit(X)

            else:
                # Standard STS or others
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
            # return PCA(n_components=128).fit_transform(X)

    return MTEBWrapper(base_model, postprocessor)


def run_mteb_comparison_bench(task_name, base_model):
    """Run wrapper along with base_model using MTEB API"""

    pp = build_postprocessor(task_name)
    print(f"[LOG] using {TASK_REGISTRY[task_name]}")

    # mtebv2 wrapper
    wrapped_model = create_mteb_wrapper(base_model, pp)
    wrapped_model.fit_from_task(task_name)

    tasks = mteb.get_tasks(tasks=[task_name])
    os.makedirs(DIR, exist_ok=True)
    cache = mteb.ResultCache(cache_path=DIR)

    print("--- Running Evaluation for Wrapped Model ---")
    mteb.evaluate(
        model=wrapped_model, tasks=tasks, cache=cache, overwrite_strategy="always"
    )


if __name__ == "__main__":
    MODEL = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",  # SBERT
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/paraphrase-mpnet-base-v2",
    ]
    TASK = [
        "Banking77Classification.v2",  # Cliassification
        "EmotionClassification.v2",  # Classifiaction
        "STSBenchmark",  # STS
        "TwentyNewsgroupsClustering.v2",  # Clustering
        "SprintDuplicateQuestions",  # PairClassification
        # "NFCorpus",  # Retrieval
    ]
    for model in MODEL:
        print(f"[LOG] Load model {model}")
        base_model = mteb.get_model(model)
        for task in TASK:
            print(f"[LOG] Load task {task}")
            run_mteb_comparison_bench(task, base_model)
