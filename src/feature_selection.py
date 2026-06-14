# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : feature_selection.py
#  Rôle    : Sélection de variables (Feature Selection)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("viridis")

# ──────────────────────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────────────────────
CHEMIN_MODELS  = os.path.join('outputs', 'models')
CHEMIN_OUTPUT  = os.path.join('outputs', 'figures')
CHEMIN_TABLES  = os.path.join('outputs', 'tables')

os.makedirs(CHEMIN_OUTPUT, exist_ok=True)
os.makedirs(CHEMIN_TABLES, exist_ok=True)


# ==============================================================
#  ÉTAPE 1 — CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ==============================================================
def charger_donnees():
    """
    Charge les données prétraitées depuis le fichier .pkl
    généré par preprocessing.py
    """
    print("=" * 60)
    print("ÉTAPE 1 : Chargement des données prétraitées")
    print("=" * 60)

    chemin_pkl = os.path.join(CHEMIN_MODELS, 'donnees_preprocessees.pkl')

    with open(chemin_pkl, 'rb') as f:
        data = pickle.load(f)

    X_train_bal = data['X_train_bal']
    X_test        = data['X_test']
    y_train_bal   = data['y_train_bal']
    y_test        = data['y_test']
    noms_features = data['noms_features']
    encodeurs     = data['encodeurs']

    print(f"  ✓ Données chargées depuis {chemin_pkl}")
    print(f"  ✓ X_train_bal : {X_train_bal.shape}")
    print(f"  ✓ X_test      : {X_test.shape}")
    print(f"  ✓ Features    : {noms_features}")

    return X_train_bal, X_test, y_train_bal, y_test, noms_features, encodeurs


# ==============================================================
#  ÉTAPE 2 — FEATURE IMPORTANCE (RANDOM FOREST)
# ==============================================================
def feature_importance_rf(X_train, y_train, noms_features):
    """
    Calcule l'importance des variables avec Random Forest.
    C'est la méthode la plus fiable pour ce dataset.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Feature Importance (Random Forest)")
    print("=" * 60)

    # Modèle rapide pour l'importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Importance
    importance = pd.DataFrame({
        'feature': noms_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  Importance des variables :")
    print(importance.to_string(index=False))

    # Visualisation
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance – Random Forest', fontsize=14)
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'feature_importance_rf.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : feature_importance_rf.png")

    # Sauvegarde
    importance.to_csv(os.path.join(CHEMIN_TABLES, 'feature_importance.csv'), index=False)

    return importance


# ==============================================================
#  ÉTAPE 3 — SELECTKBEST (CHI2)
# ==============================================================
def select_kbest_chi2(X_train, y_train, noms_features, k=5):
    """
    Sélectionne les k meilleures variables avec le test du Chi2.
    Adapté aux variables catégorielles encodées.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : SelectKBest (Chi2)")
    print("=" * 60)

    # Chi2 nécessite des valeurs positives
    X_train_pos = np.abs(X_train)

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_train_pos, y_train)

    scores = pd.DataFrame({
        'feature': noms_features,
        'chi2_score': selector.scores_,
        'selected': selector.get_support()
    }).sort_values('chi2_score', ascending=False)

    print("\n  Scores Chi2 :")
    print(scores.to_string(index=False))

    # Variables sélectionnées
    selected_features = [noms_features[i] for i in range(len(noms_features))
                         if selector.get_support()[i]]
    print(f"\n  ✓ Variables sélectionnées ({k}) : {selected_features}")

    # Visualisation
    plt.figure(figsize=(10, 6))
    sns.barplot(data=scores, x='chi2_score', y='feature', palette='coolwarm')
    plt.title('Scores Chi2 – Sélection de variables', fontsize=14)
    plt.xlabel('Score Chi2')
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'chi2_scores.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : chi2_scores.png")

    scores.to_csv(os.path.join(CHEMIN_TABLES, 'chi2_scores.csv'), index=False)

    return selected_features, scores


# ==============================================================
#  ÉTAPE 4 — MUTUAL INFORMATION
# ==============================================================
def mutual_information(X_train, y_train, noms_features):
    """
    Calcule l'information mutuelle entre chaque variable et la cible.
    Capture les relations non-linéaires.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Mutual Information")
    print("=" * 60)

    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    mi_df = pd.DataFrame({
        'feature': noms_features,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)

    print("\n  Scores d'information mutuelle :")
    print(mi_df.to_string(index=False))

    # Visualisation
    plt.figure(figsize=(10, 6))
    sns.barplot(data=mi_df, x='mutual_info', y='feature', palette='magma')
    plt.title('Information Mutuelle – Dépendance avec la cible', fontsize=14)
    plt.xlabel('Score MI')
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'mutual_information.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : mutual_information.png")

    mi_df.to_csv(os.path.join(CHEMIN_TABLES, 'mutual_information.csv'), index=False)

    return mi_df


# ==============================================================
#  ÉTAPE 5 — RFE (RECURSIVE FEATURE ELIMINATION)
# ==============================================================
def rfe_selection(X_train, y_train, noms_features, n_features=5):
    """
    Élimination récursive de variables avec Decision Tree.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : RFE – Recursive Feature Elimination")
    print("=" * 60)

    estimator = DecisionTreeClassifier(random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X_train, y_train)

    ranking = pd.DataFrame({
        'feature': noms_features,
        'ranking': selector.ranking_,
        'selected': selector.support_
    }).sort_values('ranking')

    print("\n  Classement RFE (1 = meilleur) :")
    print(ranking.to_string(index=False))

    selected_rfe = [noms_features[i] for i in range(len(noms_features))
                    if selector.support_[i]]
    print(f"\n  ✓ Variables sélectionnées par RFE ({n_features}) : {selected_rfe}")

    ranking.to_csv(os.path.join(CHEMIN_TABLES, 'rfe_ranking.csv'), index=False)

    return selected_rfe, ranking


# ==============================================================
#  ÉTAPE 6 — SYNTHÈSE ET COMPARAISON DES MÉTHODES
# ==============================================================
def synthese_selection(importance, chi2_scores, mi_df, rfe_ranking, noms_features):
    """
    Compare les 4 méthodes de sélection et propose un consensus.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Synthèse – Consensus des méthodes")
    print("=" * 60)

    # Normaliser les scores pour comparaison
    imp_norm = importance.copy()
    imp_norm['score'] = imp_norm['importance'] / imp_norm['importance'].max()

    chi2_norm = chi2_scores.copy()
    chi2_norm['score'] = chi2_norm['chi2_score'] / chi2_norm['chi2_score'].max()

    mi_norm = mi_df.copy()
    mi_norm['score'] = mi_norm['mutual_info'] / mi_norm['mutual_info'].max()

    # Fusion
    synthese = pd.DataFrame({'feature': noms_features})
    synthese = synthese.merge(imp_norm[['feature', 'score']], on='feature', how='left')
    synthese = synthese.rename(columns={'score': 'rf_importance'})
    synthese = synthese.merge(chi2_norm[['feature', 'score']], on='feature', how='left')
    synthese = synthese.rename(columns={'score': 'chi2_score'})
    synthese = synthese.merge(mi_norm[['feature', 'score']], on='feature', how='left')
    synthese = synthese.rename(columns={'score': 'mutual_info'})
    synthese = synthese.merge(rfe_ranking[['feature', 'ranking']], on='feature', how='left')
    synthese['rfe_score'] = 1 / synthese['ranking']  # Inverser le ranking

    # Score moyen
    synthese['score_moyen'] = synthese[['rf_importance', 'chi2_score', 'mutual_info', 'rfe_score']].mean(axis=1)
    synthese = synthese.sort_values('score_moyen', ascending=False)

    print("\n  Tableau de synthèse (scores normalisés) :")
    print(synthese.round(3).to_string(index=False))

    # Visualisation heatmap
    plt.figure(figsize=(10, 6))
    heatmap_data = synthese.set_index('feature')[['rf_importance', 'chi2_score', 'mutual_info', 'rfe_score']]
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=0.5)
    plt.title('Comparaison des méthodes de sélection de variables', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'synthese_feature_selection.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : synthese_feature_selection.png")

    # Top 5 variables consensus
    top5 = synthese.head(5)['feature'].tolist()
    print(f"\n  ⭐ TOP 5 VARIABLES (consensus) : {top5}")

    synthese.to_csv(os.path.join(CHEMIN_TABLES, 'synthese_feature_selection.csv'), index=False)

    return synthese, top5


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_feature_selection():
    """
    Exécute toute la sélection de variables.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║      FEATURE SELECTION – Sélection de Variables ORL     ║")
    print("╚" + "═" * 58 + "╝")

    X_train_bal, X_test, y_train_bal, y_test, noms_features, encodeurs = charger_donnees()

    importance = feature_importance_rf(X_train_bal, y_train_bal, noms_features)
    selected_chi2, chi2_scores = select_kbest_chi2(X_train_bal, y_train_bal, noms_features, k=5)
    mi_df = mutual_information(X_train_bal, y_train_bal, noms_features)
    selected_rfe, rfe_ranking = rfe_selection(X_train_bal, y_train_bal, noms_features, n_features=5)
    synthese, top5 = synthese_selection(importance, chi2_scores, mi_df, rfe_ranking, noms_features)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  FEATURE SELECTION TERMINÉ – Variables clés identifiées  ║")
    print("╚" + "═" * 58 + "╝\n")

    return synthese, top5


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    synthese, top5 = pipeline_feature_selection()