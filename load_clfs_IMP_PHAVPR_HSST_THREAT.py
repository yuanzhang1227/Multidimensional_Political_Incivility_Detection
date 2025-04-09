#!/usr/bin/env python
# -*- coding: utf-8 -*-


import joblib
import sqlite3
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

clf_LR_IMP = joblib.load(
    "./CLF/LR_ROUND2_impoliteness_ON_paraphrase-multilingual-mpnet-base-v2.joblib")

clf_LR_PHAVPR1  = joblib.load(
    "./CLF/LR_ROUND2_PHAVPR1_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_PHAVPR2  = joblib.load(
    "./CLF/LR_ROUND2_PHAVPR2_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_PHAVPR3  = joblib.load(
    "./CLF/LR_ROUND2_PHAVPR3_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_PHAVPR4  = joblib.load(
    "./CLF/LR_ROUND2_PHAVPR4_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_PHAVPR5  = joblib.load(
    "./CLF/LR_ROUND2_PHAVPR5_ON_paraphrase-multilingual-mpnet-base-v2.joblib")

clf_LR_HSST1 = joblib.load(
    "./CLF/LR_ROUND2_HSST1_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_HSST2 = joblib.load(
    "./CLF/LR_ROUND2_HSST2_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_HSST3 = joblib.load(
    "./CLF/LR_ROUND2_HSST3_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_HSST4 = joblib.load(
    "./CLF/LR_ROUND2_HSST4_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_HSST5 = joblib.load(
    "./CLF/LR_ROUND2_HSST5_ON_paraphrase-multilingual-mpnet-base-v2.joblib")

clf_LR_THREAT1 = joblib.load(
    "./CLF/LR_ROUND2_THREAT1_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_THREAT2 = joblib.load(
    "./CLF/LR_ROUND2_THREAT2_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_THREAT3 = joblib.load(
    "./CLF/LR_ROUND2_THREAT3_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_THREAT4 = joblib.load(
    "./CLF/LR_ROUND2_THREAT4_ON_paraphrase-multilingual-mpnet-base-v2.joblib")
clf_LR_THREAT5 = joblib.load(
    "./CLF/LR_ROUND2_THREAT5_ON_paraphrase-multilingual-mpnet-base-v2.joblib")


list_of_clf_LR_THREAT = [clf_LR_THREAT1,
                         clf_LR_THREAT2,
                         clf_LR_THREAT3,
                         clf_LR_THREAT4,
                         clf_LR_THREAT5]

list_of_clf_LR_HSST = [clf_LR_HSST1,
                       clf_LR_HSST2,
                       clf_LR_HSST3,
                       clf_LR_HSST4,
                       clf_LR_HSST5]

list_of_clf_LR_PHAVPR = [clf_LR_PHAVPR1,
                       clf_LR_PHAVPR2,
                       clf_LR_PHAVPR3,
                       clf_LR_PHAVPR4,
                       clf_LR_PHAVPR5] 


# funcs

def get_norm_vecs(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Normalize the vectors to unit vectors
    unit_vectors = vectors / norms

    return unit_vectors

def get_text(lof_json_strings):
    # we create an empty one if something goes wrong
    return [json.loads(json_str).get("message", "") for json_str in lof_json_strings]


def predict_LR(embs, LR=clf_LR_IMP, cat="IMP"):

    # predict:
    print(f"predicting {cat}...")
    labels = LR.predict(embs)
    print(f"predicting probas {cat}...")
    probas = LR.predict_proba(embs)

    # Split into two arrays
    probas_class_0 = probas[:, 0]  # Probabilities for class 0
    probas_class_1 = probas[:, 1]  # Probabilities for class 1

    return labels, probas_class_0, probas_class_1


def get_majority_vote(votes, verbosity):
    # list_votes = votes.tolist()
    if verbosity:
        print(f"I got {votes}")
    res = max(votes, key=votes.count)
    if verbosity:
        print(f"{votes=}")
        print(f"{res}=")

    return res


def predict_list_of_LRs(embs, list_of_LRs=list_of_clf_LR_THREAT, cat="THREAT", verbosity=0):

    probas_class_0_list = []
    probas_class_1_list = []
    labels_list = []

    for clf in list_of_LRs:
        # predict:
        print(f"predicting {cat}...")
        labels = clf.predict(embs)
        labels_list.append(labels)

        print(f"predicting probas {cat}...")
        probas = clf.predict_proba(embs)

        # Split into two arrays
        probas_class_0_list.append(probas[:, 0])  # Probabilities for class 0
        probas_class_1_list.append(probas[:, 1])  # Probabilities for class 1

    # now aggregate / ensemble
    if verbosity:
        q = labels_list[0]
        print(f"NOW got {q=}")
    majority_vote_labels = [get_majority_vote(
        votes, verbosity=verbosity) for votes in zip(*labels_list)]

    mean_probas_class_0 = [np.mean(probas_class_0)
                           for probas_class_0 in zip(*probas_class_0_list)]
    mean_probas_class_1 = [np.mean(probas_class_1)
                           for probas_class_1 in zip(*probas_class_1_list)]

    return majority_vote_labels, mean_probas_class_0, mean_probas_class_1
    


def predict_for_lists_of_text(given_list_of_text,
                              model=model,
                              LR_IMP = clf_LR_IMP,
                              LRs_PHAVPR = list_of_clf_LR_PHAVPR,
                              LRs_HSST=list_of_clf_LR_HSST,
                              LRs_THREAT=list_of_clf_LR_THREAT
                              ):
    # encode
    print("encoding ...")
    embs = model.encode(given_list_of_text, batch_size=8,
                        show_progress_bar=True)
    
    # predict IMP:
    labels_IMP, probas_class_0_IMP, probas_class_1_IMP = predict_LR(
        embs, LR=LR_IMP, cat="IMP")
    
    # here come the ensembles
    # predict PHAVPR
    labels_PHAVPR, mean_probas_class_0_PHAVPR, mean_probas_class_1_PHAVPR = predict_list_of_LRs(
        embs, list_of_LRs=LRs_PHAVPR, cat="PHAVPR", verbosity=0)

    # predict THREAT
    labels_THREAT, mean_probas_class_0_THREAT, mean_probas_class_1_THREAT = predict_list_of_LRs(
        embs, list_of_LRs=LRs_THREAT, cat="THREAT", verbosity=0)

    # predict HSST
    labels_HSST, mean_probas_class_0_HSST, mean_probas_class_1_HSST = predict_list_of_LRs(
        embs, list_of_LRs=LRs_HSST, cat="HSST", verbosity=0)

    res = dict()
    
    res["labels_IMP"] = labels_IMP
    res["probas_class_0_IMP"] = probas_class_0_IMP
    res["probas_class_1_IMP"] = probas_class_1_IMP

    res["labels_PHAVPR"] = labels_PHAVPR
    res["probas_class_0_PHAVPR"] = mean_probas_class_0_PHAVPR
    res["probas_class_1_PHAVPR"] = mean_probas_class_1_PHAVPR

    res["labels_HSST"] = labels_HSST
    res["probas_class_0_HSST"] = mean_probas_class_0_HSST
    res["probas_class_1_HSST"] = mean_probas_class_1_HSST
    
    res["labels_THREAT"] = labels_THREAT
    res["probas_class_0_THREAT"] = mean_probas_class_0_THREAT
    res["probas_class_1_THREAT"] = mean_probas_class_1_THREAT
    
    return res


# # === Multidimensional Incivility Prediction Example ===

# # Define a list of texts to classify
# TEXT_LIST = ['text1', 'text2', 'text3', 'text4', 'text5', ...]  # Add your texts here

# # Predict incivility dimensions using pre-trained models
# Prediction_df = predict_for_lists_of_text(
#     TEXT_LIST,
#     model=model,
#     LR_IMP=clf_LR_IMP,
#     LRs_PHAVPR=list_of_clf_LR_PHAVPR,
#     LRs_HSST=list_of_clf_LR_HSST,
#     LRs_THREAT=list_of_clf_LR_THREAT
# )

# # Save the predictions to a CSV file
# Prediction_df.to_csv('Prediction_Multidimensional_Incivility.csv', index=False)

# # --- Optional: Predict only one dimension ---
# # If you only want to classify a single dimension (e.g., Impoliteness),
# # just keep that classifier and comment out the others:
# # Prediction_df = predict_for_lists_of_text(
# #     TEXT_LIST,
# #     model=model,
# #     LR_IMP=clf_LR_IMP
# # )
