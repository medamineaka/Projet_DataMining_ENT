# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : preprocessing.py
#  Rôle    : Chargement, nettoyage, encodage, SMOTE
# =============================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────────────────────
CHEMIN_DATA    = os.path.join('data', 'Diagnostic_error_of_ENT_diseases.xlsx')
CHEMIN_OUTPUT  = os.path.join('outputs', 'tables')
CHEMIN_MODELS  = os.path.join('outputs', 'models')


# ==============================================================
#  ÉTAPE 1 — CHARGEMENT DES DONNÉES
# ==============================================================
def charger_donnees(chemin=CHEMIN_DATA):
    """
    Charge le fichier Excel du dataset ORL.
    Retourne un DataFrame pandas brut.
    """
    print("=" * 60)
    print("ÉTAPE 1 : Chargement des données")
    print("=" * 60)

    df = pd.read_excel(chemin)

    print(f"  ✓ Dataset chargé : {df.shape[0]} patients, {df.shape[1]} variables")
    print(f"  ✓ Colonnes       : {list(df.columns)}")
    print(f"\n  Distribution variable cible :")
    print(df['referral appropriateness'].value_counts().to_string())

    return df


# ==============================================================
#  ÉTAPE 2 — NETTOYAGE DES DONNÉES
# ==============================================================
def nettoyer_donnees(df):
    """
    Nettoie les incohérences textuelles du dataset :
      - Supprime les espaces en début/fin de chaîne
      - Uniformise les casses (minuscules)
      - Corrige les fautes de frappe détectées
      - Supprime les colonnes non exploitables
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Nettoyage des données")
    print("=" * 60)

    df = df.copy()

    # ── 2.1 Renommer les colonnes (espaces parasites)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'UTH department ': 'UTH department'})

    # ── 2.2 Nettoyer toutes les colonnes texte
    colonnes_texte = df.select_dtypes(include=['object', 'str']).columns
    for col in colonnes_texte:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # ── 2.3 Corriger les fautes de frappe connues
    corrections = {
        'attendance': {
            'first attendane'  : 'first attendance',   # faute de frappe
            'review '          : 'review',              # espace résiduel
            'first attendance ': 'first attendance',    # espace résiduel
        },
        'specialtyentdx1': {
            'otology'          : 'otology',             # déjà bon
            'Otology'          : 'otology',             # majuscule
            'head and neck'    : 'head and neck',
            'head and Neck'    : 'head and neck',
        },
        'entdiag1': {
            'tonsillar hypertrophy ': 'tonsillar hypertrophy',
        }
    }

    nb_corrections = 0
    for col, mapping in corrections.items():
        if col in df.columns:
            avant = df[col].copy()
            df[col] = df[col].replace(mapping)
            n = (avant != df[col]).sum()
            if n > 0:
                print(f"  ✓ {col} : {n} valeur(s) corrigée(s)")
                nb_corrections += n

    print(f"  → Total corrections textuelles : {nb_corrections}")

    # ── 2.4 Supprimer les colonnes non exploitables
    #   'additional entdiag1 information' : texte libre, trop hétérogène
    #   'reason'                          : data leakage (explique pourquoi
    #                                       la référence est inappropriée)
    colonnes_a_supprimer = ['additional entdiag1 information', 'reason']
    df = df.drop(columns=colonnes_a_supprimer, errors='ignore')
    print(f"  ✓ Colonnes supprimées : {colonnes_a_supprimer}")
    print(f"  ✓ Dataset après nettoyage : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    return df


# ==============================================================
#  ÉTAPE 3 — CRÉATION DE NOUVELLES VARIABLES
# ==============================================================
def creer_variables(df):
    """
    Crée des variables dérivées utiles pour l'analyse :
      - age_group : tranche d'âge médicale (6 catégories)
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Création de nouvelles variables")
    print("=" * 60)

    df = df.copy()

    # ── 3.1 Tranche d'âge médicale
    bins   = [0, 2, 5, 12, 18, 45, 65, 120]
    labels = ['nourrisson', 'tout-petit', 'enfant',
              'adolescent', 'adulte', 'adulte-mature', 'senior']

    df['age_group'] = pd.cut(
        df['age'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    ).astype(str)

    print("  ✓ Variable 'age_group' créée :")
    print(df['age_group'].value_counts().to_string())

    return df


# ==============================================================
#  ÉTAPE 4 — SÉLECTION DES FEATURES
# ==============================================================
def selectionner_features(df):
    """
    Sélectionne les variables explicatives (X) et la cible (y).
    Variables retenues pour la modélisation :
      - age, age_group, UTH referral, UTH department,
        facility ref level, refdiag1, reason for referral,
        entdiag1, specialtyentdx1, attendance
    Variable cible : referral appropriateness
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Sélection des features")
    print("=" * 60)

    features = [
        'age',
        'age_group',
        'UTH referral?',
        'UTH department',
        'facility ref level',
        'refdiag1',
        'reason for referral',
        'entdiag1',
        'specialtyentdx1',
        'attendance'
    ]

    cible = 'referral appropriateness'

    # Vérification que toutes les colonnes existent
    features_ok = [f for f in features if f in df.columns]
    features_manquantes = [f for f in features if f not in df.columns]
    if features_manquantes:
        print(f"  ⚠ Features manquantes ignorées : {features_manquantes}")

    X = df[features_ok].copy()
    y = df[cible].copy()

    print(f"  ✓ Features sélectionnées ({len(features_ok)}) : {features_ok}")
    print(f"  ✓ Variable cible         : {cible}")
    print(f"  ✓ Taille X               : {X.shape}")

    return X, y, features_ok


# ==============================================================
#  ÉTAPE 5 — ENCODAGE DES VARIABLES CATÉGORIELLES
# ==============================================================
def encoder_donnees(X, y):
    """
    Encode toutes les variables catégorielles avec LabelEncoder.
    Sauvegarde les encodeurs pour pouvoir inverser les labels plus tard.
    Retourne X encodé, y encodé, et le dictionnaire des encodeurs.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Encodage des variables")
    print("=" * 60)

    X_enc = X.copy()
    encodeurs = {}

    # ── 5.1 Encoder les colonnes catégorielles de X
    colonnes_cat = X_enc.select_dtypes(include=['object', 'str']).columns.tolist()

    for col in colonnes_cat:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encodeurs[col] = le
        print(f"  ✓ {col:30s} → {len(le.classes_)} classes encodées")

    # ── 5.2 Encoder la variable cible y
    le_cible = LabelEncoder()
    y_enc = le_cible.fit_transform(y.astype(str))
    encodeurs['target'] = le_cible

    print(f"\n  ✓ Variable cible encodée :")
    for i, cls in enumerate(le_cible.classes_):
        n = (y_enc == i).sum()
        print(f"      {i} = '{cls}' ({n} patients)")

    # ── 5.3 Vérification des valeurs nulles après encodage
    nulls = X_enc.isnull().sum().sum()
    print(f"\n  ✓ Valeurs nulles après encodage : {nulls}")

    return X_enc, y_enc, encodeurs


# ==============================================================
#  ÉTAPE 6 — TRAIN / TEST SPLIT STRATIFIÉ
# ==============================================================
def split_donnees(X_enc, y_enc, test_size=0.25, random_state=42):
    """
    Découpe les données en ensemble d'entraînement (75%) et de test (25%).
    Stratifié pour conserver la proportion des 3 classes.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Train / Test Split (75% / 25%)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc        # important pour les 3 classes déséquilibrées
    )

    print(f"  ✓ X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    print(f"  ✓ Distribution y_train : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  ✓ Distribution y_test  : {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    return X_train, X_test, y_train, y_test


# ==============================================================
#  ÉTAPE 7 — SMOTE (Suréchantillonnage des minorités)
# ==============================================================
def appliquer_smote(X_train, y_train, random_state=42):
    """
    Applique SMOTE uniquement sur le jeu d'entraînement
    pour équilibrer les 3 classes.
    SMOTE génère des exemples synthétiques de la classe minoritaire.

    AVANT SMOTE  → inappropriate : ~583  |  appropriate : ~448  |  not applicable : ~127
    APRÈS SMOTE  → toutes les classes équilibrées (~583 chacune)
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : SMOTE – Équilibrage des classes")
    print("=" * 60)

    print("  AVANT SMOTE :")
    classes, counts = np.unique(y_train, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"    Classe {c} : {n} exemples")

    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("\n  APRÈS SMOTE :")
    classes, counts = np.unique(y_train_bal, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"    Classe {c} : {n} exemples")

    print(f"\n  ✓ X_train équilibré : {X_train_bal.shape}")
    gain = X_train_bal.shape[0] - X_train.shape[0]
    print(f"  ✓ Exemples synthétiques ajoutés : {gain}")

    return X_train_bal, y_train_bal


# ==============================================================
#  ÉTAPE 8 — TABLEAU RÉCAPITULATIF SMOTE (pour le rapport)
# ==============================================================
def generer_tableau_smote(y_train, y_train_bal, encodeurs):
    """
    Génère un tableau CSV comparatif avant/après SMOTE.
    Ce tableau est directement utilisable dans le rapport.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 8 : Tableau récapitulatif SMOTE")
    print("=" * 60)

    le_cible = encodeurs['target']

    avant  = dict(zip(*np.unique(y_train,     return_counts=True)))
    apres  = dict(zip(*np.unique(y_train_bal, return_counts=True)))

    lignes = []
    for code in sorted(avant.keys()):
        nom   = le_cible.inverse_transform([code])[0]
        n_av  = avant.get(code, 0)
        n_ap  = apres.get(code, 0)
        delta = n_ap - n_av
        lignes.append({
            'Classe'              : nom,
            'Avant SMOTE'         : n_av,
            'Après SMOTE'         : n_ap,
            'Exemples ajoutés'    : delta,
            'Ratio équilibrage %' : round(n_ap / sum(apres.values()) * 100, 1)
        })

    tableau = pd.DataFrame(lignes)
    chemin_csv = os.path.join(CHEMIN_OUTPUT, 'smote_comparaison.csv')
    tableau.to_csv(chemin_csv, index=False, encoding='utf-8-sig')

    print(tableau.to_string(index=False))
    print(f"\n  ✓ Tableau sauvegardé → {chemin_csv}")

    return tableau


# ==============================================================
#  ÉTAPE 9 — SAUVEGARDE DES DONNÉES PRÉTRAITÉES
# ==============================================================
def sauvegarder_donnees(X_train_bal, X_test, y_train_bal, y_test,
                         X_train, y_train, encodeurs, noms_features):
    """
    Sauvegarde tous les objets nécessaires pour les scripts suivants :
      - X_train_bal, y_train_bal : données équilibrées (SMOTE) pour entraîner
      - X_test, y_test           : données de test (jamais touchées par SMOTE)
      - X_train, y_train         : données originales avant SMOTE
      - encodeurs                : dict des LabelEncoders
      - noms_features            : liste des noms de colonnes
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 9 : Sauvegarde des données prétraitées")
    print("=" * 60)

    donnees = {
        'X_train_bal'   : X_train_bal,
        'X_test'        : X_test,
        'y_train_bal'   : y_train_bal,
        'y_test'        : y_test,
        'X_train'       : X_train,
        'y_train'       : y_train,
        'encodeurs'     : encodeurs,
        'noms_features' : noms_features
    }

    chemin_pkl = os.path.join(CHEMIN_MODELS, 'donnees_preprocessees.pkl')
    with open(chemin_pkl, 'wb') as f:
        pickle.dump(donnees, f)

    print(f"  ✓ Données sauvegardées → {chemin_pkl}")
    print(f"  ✓ Contenu : {list(donnees.keys())}")

    return chemin_pkl


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_preprocessing():
    """
    Exécute toutes les étapes de preprocessing dans l'ordre.
    Retourne les données prêtes pour la modélisation.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║    PREPROCESSING – Dataset Erreurs Diagnostic ORL       ║")
    print("╚" + "═" * 58 + "╝")

    # ── Étapes séquentielles
    df               = charger_donnees()
    df               = nettoyer_donnees(df)
    df               = creer_variables(df)
    X, y, features   = selectionner_features(df)
    X_enc, y_enc, encodeurs = encoder_donnees(X, y)
    X_train, X_test, y_train, y_test = split_donnees(X_enc, y_enc)
    X_train_bal, y_train_bal = appliquer_smote(X_train, y_train)
    generer_tableau_smote(y_train, y_train_bal, encodeurs)
    sauvegarder_donnees(X_train_bal, X_test, y_train_bal, y_test,
                        X_train, y_train, encodeurs, features)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  PREPROCESSING TERMINÉ – Données prêtes pour les algos  ║")
    print("╚" + "═" * 58 + "╝\n")

    return {
        'X_train_bal'   : X_train_bal,
        'X_test'        : X_test,
        'y_train_bal'   : y_train_bal,
        'y_test'        : y_test,
        'encodeurs'     : encodeurs,
        'noms_features' : features
    }


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    donnees = pipeline_preprocessing()