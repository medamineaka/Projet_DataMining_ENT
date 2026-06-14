# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : utils.py
#  Rôle    : Fonctions utilitaires partagées entre tous les scripts
# =============================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Créer les dossiers si nécessaire
for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Style matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# ==============================================================
#  CHARGEMENT / SAUVEGARDE DES DONNÉES
# ==============================================================

def charger_donnees_pretraitees():
    """
    Charge les données prétraitées depuis le fichier .pkl
    généré par preprocessing.py
    """
    chemin = os.path.join(MODELS_DIR, 'donnees_preprocessees.pkl')

    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Fichier non trouvé : {chemin}\n"
            f"Exécutez d'abord preprocessing.py"
        )

    with open(chemin, 'rb') as f:
        data = pickle.load(f)

    return {
        'X_train_bal': data['X_train_bal'],
        'X_test': data['X_test'],
        'y_train_bal': data['y_train_bal'],
        'y_test': data['y_test'],
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'encodeurs': data['encodeurs'],
        'noms_features': data['noms_features']
    }


def sauvegarder_modele(modele, nom_fichier):
    """
    Sauvegarde un modèle entraîné au format .pkl
    """
    chemin = os.path.join(MODELS_DIR, f'{nom_fichier}.pkl')
    with open(chemin, 'wb') as f:
        pickle.dump(modele, f)
    print(f"  ✓ Modèle sauvegardé → {chemin}")
    return chemin


def charger_modele(nom_fichier):
    """
    Charge un modèle sauvegardé
    """
    chemin = os.path.join(MODELS_DIR, f'{nom_fichier}.pkl')
    with open(chemin, 'rb') as f:
        return pickle.load(f)


# ==============================================================
#  ÉVALUATION DES MODÈLES
# ==============================================================

def evaluer_modele(y_true, y_pred, y_proba=None, noms_classes=None):
    """
    Calcule toutes les métriques d'évaluation et retourne un dict.

    Paramètres:
        y_true : labels réels
        y_pred : labels prédits
        y_proba : probabilités prédites (optionnel, pour ROC-AUC)
        noms_classes : liste des noms de classes
    """
    if noms_classes is None:
        noms_classes = ['inappropriate', 'appropriate', 'not applicable']

    # Métriques de base
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # ROC-AUC (si probabilités disponibles et classification binaire)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                # Multiclass : one-vs-rest
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)
                metrics['roc_auc'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            metrics['roc_auc'] = None
            print(f"  ⚠ ROC-AUC non calculable : {e}")

    # Classification report détaillé
    report = classification_report(
        y_true, y_pred,
        target_names=noms_classes[:len(np.unique(y_true))],
        output_dict=True,
        zero_division=0
    )

    return {
        'metrics': metrics,
        'report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def afficher_metrics(metrics_dict):
    """
    Affiche les métriques de manière formatée
    """
    m = metrics_dict['metrics']
    print("\n  ╔" + "═" * 40 + "╗")
    print("  ║         MÉTRIQUES D'ÉVALUATION        ║")
    print("  ╠" + "═" * 40 + "╣")
    print(f"  ║  Accuracy   : {m['accuracy']:.4f}              ║")
    print(f"  ║  Precision  : {m['precision']:.4f}              ║")
    print(f"  ║  Recall     : {m['recall']:.4f}              ║")
    print(f"  ║  F1-Score   : {m['f1_score']:.4f}              ║")
    if 'roc_auc' in m and m['roc_auc'] is not None:
        print(f"  ║  ROC-AUC    : {m['roc_auc']:.4f}              ║")
    print("  ╚" + "═" * 40 + "╝")


# ==============================================================
#  VISUALISATIONS COMMUNES
# ==============================================================

def plot_confusion_matrix(cm, noms_classes, titre, nom_fichier):
    """
    Affiche et sauvegarde une matrice de confusion
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=noms_classes,
        yticklabels=noms_classes,
        linewidths=0.5, linecolor='gray'
    )
    plt.title(titre, fontsize=14, pad=15)
    plt.ylabel('Valeur réelle', fontsize=12)
    plt.xlabel('Valeur prédite', fontsize=12)
    plt.tight_layout()

    chemin = os.path.join(FIGURES_DIR, f'{nom_fichier}.png')
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Matrice de confusion sauvegardée → {chemin}")
    return chemin


def plot_roc_curve(y_true, y_proba, titre, nom_fichier):
    """
    Affiche et sauvegarde la courbe ROC
    """
    # Binaire uniquement
    if len(np.unique(y_true)) != 2:
        print("  ⚠ Courbe ROC uniquement pour classification binaire")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs (1 - Spécificité)', fontsize=11)
    plt.ylabel('Taux de vrais positifs (Sensibilité)', fontsize=11)
    plt.title(titre, fontsize=14, pad=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chemin = os.path.join(FIGURES_DIR, f'{nom_fichier}.png')
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Courbe ROC sauvegardée → {chemin}")
    return chemin


def plot_comparaison_modeles(results_df, nom_fichier='comparaison_modeles'):
    """
    Barplot comparatif de plusieurs modèles
    results_df : DataFrame avec index=noms_modèles, colonnes=métriques
    """
    plt.figure(figsize=(12, 7))
    results_df.plot(kind='bar', rot=0, width=0.8, colormap='Set2')
    plt.title('Comparaison des modèles de classification', fontsize=14, pad=15)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Modèle', fontsize=12)
    plt.legend(title='Métrique', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.05)
    plt.tight_layout()

    chemin = os.path.join(FIGURES_DIR, f'{nom_fichier}.png')
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Comparaison sauvegardée → {chemin}")
    return chemin


# ==============================================================
#  TABLEAUX DE RÉSULTATS
# ==============================================================

def sauvegarder_tableau(df, nom_fichier, index=False):
    """
    Sauvegarde un DataFrame au format CSV
    """
    chemin = os.path.join(TABLES_DIR, f'{nom_fichier}.csv')
    df.to_csv(chemin, index=index, encoding='utf-8-sig')
    print(f"  ✓ Tableau sauvegardé → {chemin}")
    return chemin


def creer_tableau_smote(y_avant, y_apres, encodeurs):
    """
    Crée le tableau comparatif avant/après SMOTE
    """
    le_cible = encodeurs['target']

    avant = dict(zip(*np.unique(y_avant, return_counts=True)))
    apres = dict(zip(*np.unique(y_apres, return_counts=True)))

    lignes = []
    for code in sorted(avant.keys()):
        nom = le_cible.inverse_transform([code])[0]
        n_av = avant.get(code, 0)
        n_ap = apres.get(code, 0)
        lignes.append({
            'Classe': nom,
            'Avant SMOTE': n_av,
            'Après SMOTE': n_ap,
            'Exemples ajoutés': n_ap - n_av,
            'Ratio %': round(n_ap / sum(apres.values()) * 100, 1)
        })

    return pd.DataFrame(lignes)


# ==============================================================
#  UTILITAIRES DIVERS
# ==============================================================

def get_noms_classes(encodeurs):
    """
    Retourne les noms des classes depuis l'encodeur de la cible
    """
    le = encodeurs.get('target')
    if le is None:
        return ['classe_0', 'classe_1', 'classe_2']
    return list(le.classes_)


def timer(func):
    """
    Décorateur pour mesurer le temps d'exécution
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        debut = time.time()
        result = func(*args, **kwargs)
        fin = time.time()
        duree = fin - debut
        print(f"\n  ⏱️ Temps d'exécution : {duree:.2f} secondes")
        return result

    return wrapper


def print_section(titre):
    """
    Affiche un encadré de section
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + titre.center(58) + "║")
    print("╚" + "═" * 58 + "╝")


def print_sous_section(titre):
    """
    Affiche un sous-titre
    """
    print("\n" + "─" * 60)
    print(f"  {titre}")
    print("─" * 60)


# ==============================================================
#  FONCTIONS SPÉCIFIQUES AUX ALGORITHMES
# ==============================================================

def extraire_regles_arbre(modele, noms_features, noms_classes, max_depth=3):
    """
    Extrait les règles de décision d'un arbre sous forme textuelle
    """
    from sklearn.tree import export_text

    try:
        regles = export_text(
            modele,
            feature_names=noms_features,
            max_depth=max_depth
        )
        return regles
    except Exception as e:
        print(f"  ⚠ Erreur extraction règles : {e}")
        return None


def plot_feature_importance(importance_df, titre, nom_fichier):
    """
    Barplot horizontal de feature importance (pour Random Forest)
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(10),
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title(titre, fontsize=14, pad=15)
    plt.xlabel('Importance', fontsize=11)
    plt.ylabel('Variable', fontsize=11)
    plt.tight_layout()

    chemin = os.path.join(FIGURES_DIR, f'{nom_fichier}.png')
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Feature importance sauvegardée → {chemin}")
    return chemin


def plot_dendrogram(linked, titre, nom_fichier, max_d=None):
    """
    Affiche et sauvegarde un dendrogramme CAH
    """
    from scipy.cluster.hierarchy import dendrogram

    plt.figure(figsize=(18, 8))
    dendrogram(
        linked,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=8
    )
    if max_d:
        plt.axhline(y=max_d, c='r', linestyle='--', label=f'Coupe à {max_d}')
        plt.legend()
    plt.title(titre, fontsize=14, pad=15)
    plt.ylabel('Distance (Ward)', fontsize=11)
    plt.tight_layout()

    chemin = os.path.join(FIGURES_DIR, f'{nom_fichier}.png')
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Dendrogramme sauvegardé → {chemin}")
    return chemin


# ==============================================================
#  POINT D'ENTRÉE (test rapide)
# ==============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("UTILS.PY – Fonctions utilitaires du projet ORL")
    print("=" * 60)
    print(f"\n  Dossiers configurés :")
    print(f"    BASE_DIR   : {BASE_DIR}")
    print(f"    DATA_DIR   : {DATA_DIR}")
    print(f"    OUTPUT_DIR : {OUTPUT_DIR}")
    print(f"    FIGURES    : {FIGURES_DIR}")
    print(f"    TABLES     : {TABLES_DIR}")
    print(f"    MODELS     : {MODELS_DIR}")

    # Test chargement données
    try:
        data = charger_donnees_pretraitees()
        print(f"\n  ✓ Données chargées avec succès")
        print(f"    X_train_bal : {data['X_train_bal'].shape}")
        print(f"    X_test      : {data['X_test'].shape}")
    except FileNotFoundError as e:
        print(f"\n  ⚠ {e}")