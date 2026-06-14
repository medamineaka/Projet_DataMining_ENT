# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : comparison.py
#  Rôle    : Comparaison finale de tous les modèles
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
import time
import sys

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (14, 10)
sns.set_palette("Set2")

# ──────────────────────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────────────────────
CHEMIN_MODELS = os.path.join('outputs', 'models')
CHEMIN_OUTPUT = os.path.join('outputs', 'figures')
CHEMIN_TABLES = os.path.join('outputs', 'tables')

os.makedirs(CHEMIN_OUTPUT, exist_ok=True)
os.makedirs(CHEMIN_TABLES, exist_ok=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import (
    charger_donnees_pretraitees, charger_modele, evaluer_modele,
    afficher_metrics, plot_comparaison_modeles, sauvegarder_tableau,
    print_section
)

warnings.filterwarnings('ignore')
# ==============================================================
#  ÉTAPE 1 — CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ==============================================================
def charger_donnees():
    """
    Charge les données prétraitées depuis le fichier .pkl
    """
    print("=" * 60)
    print("ÉTAPE 1 : Chargement des données prétraitées")
    print("=" * 60)

    chemin_pkl = os.path.join(CHEMIN_MODELS, 'donnees_preprocessees.pkl')

    with open(chemin_pkl, 'rb') as f:
        data = pickle.load(f)

    X_train_bal = data['X_train_bal']
    X_test = data['X_test']
    y_train_bal = data['y_train_bal']
    y_test = data['y_test']
    noms_features = data['noms_features']

    print(f"  ✓ Données chargées : train={X_train_bal.shape}, test={X_test.shape}")
    print(f"  ✓ Classes : {np.unique(y_test)}")

    return X_train_bal, X_test, y_train_bal, y_test, noms_features


# ==============================================================
#  ÉTAPE 2 — ENTRAÎNEMENT DE TOUS LES MODÈLES
# ==============================================================
def entrainer_modeles(X_train, y_train, X_test, y_test):
    """
    Entraîne les 4 modèles de classification et mesure les temps.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Entraînement des modèles")
    print("=" * 60)

    modeles = {}

    # ── 2.1 Decision Tree
    print("\n  → Decision Tree ...")
    debut = time.time()
    dt = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
    dt.fit(X_train, y_train)
    temps_dt = time.time() - debut
    y_pred_dt = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)
    modeles['Decision Tree'] = {
        'model': dt, 'y_pred': y_pred_dt, 'y_proba': y_proba_dt,
        'temps': temps_dt
    }
    print(f"     ✓ Entraîné en {temps_dt:.3f}s")

    # ── 2.2 Random Forest
    print("\n  → Random Forest ...")
    debut = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    temps_rf = time.time() - debut
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    modeles['Random Forest'] = {
        'model': rf, 'y_pred': y_pred_rf, 'y_proba': y_proba_rf,
        'temps': temps_rf
    }
    print(f"     ✓ Entraîné en {temps_rf:.3f}s")

    # ── 2.3 Naive Bayes
    print("\n  → Naive Bayes ...")
    debut = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    temps_nb = time.time() - debut
    y_pred_nb = nb.predict(X_test)
    y_proba_nb = nb.predict_proba(X_test)
    modeles['Naive Bayes'] = {
        'model': nb, 'y_pred': y_pred_nb, 'y_proba': y_proba_nb,
        'temps': temps_nb
    }
    print(f"     ✓ Entraîné en {temps_nb:.3f}s")

    # ── 2.4 KNN (optionnel, si vous l'avez gardé)
    # Si vous avez supprimé KNN, commentez cette section
    from sklearn.neighbors import KNeighborsClassifier
    print("\n  → KNN (k=5) ...")
    debut = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    temps_knn = time.time() - debut
    y_pred_knn = knn.predict(X_test)
    y_proba_knn = knn.predict_proba(X_test)
    modeles['KNN'] = {
        'model': knn, 'y_pred': y_pred_knn, 'y_proba': y_proba_knn,
        'temps': temps_knn
    }
    print(f"     ✓ Entraîné en {temps_knn:.3f}s")

    return modeles


# ==============================================================
#  ÉTAPE 3 — CALCUL DES MÉTRIQUES
# ==============================================================
def calculer_metriques(modeles, y_test):
    """
    Calcule toutes les métriques pour chaque modèle.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Calcul des métriques d'évaluation")
    print("=" * 60)

    results = []
    n_classes = len(np.unique(y_test))

    for nom, info in modeles.items():
        y_pred = info['y_pred']
        y_proba = info['y_proba']
        temps = info['temps']

        # Métriques de base
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # AUC-ROC (multiclasse : one-vs-rest)
        try:
            if n_classes > 2:
                y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
                auc = roc_auc_score(y_test_binarized, y_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
        except:
            auc = 0.5  # Si calcul impossible

        results.append({
            'Modèle': nom,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'AUC-ROC': auc,
            'Temps (s)': temps
        })

        print(f"\n  {nom} :")
        print(f"    Accuracy  = {acc:.4f}")
        print(f"    Precision = {prec:.4f}")
        print(f"    Recall    = {rec:.4f}")
        print(f"    F1-score  = {f1:.4f}")
        print(f"    AUC-ROC   = {auc:.4f}")
        print(f"    Temps     = {temps:.3f}s")

    df_results = pd.DataFrame(results)
    return df_results


# ==============================================================
#  ÉTAPE 4 — TABLEAU COMPARATIF FINAL
# ==============================================================
def tableau_comparatif(df_results):
    """
    Génère le tableau comparatif final pour le rapport.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Tableau comparatif final")
    print("=" * 60)

    # Arrondir pour le rapport
    df_display = df_results.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']:
        df_display[col] = df_display[col].round(4)
    df_display['Temps (s)'] = df_display['Temps (s)'].round(3)

    print("\n" + df_display.to_string(index=False))

    # Sauvegarde
    df_display.to_csv(os.path.join(CHEMIN_TABLES, 'comparaison_modeles.csv'), index=False)
    print(f"\n  ✓ Tableau sauvegardé : comparaison_modeles.csv")

    # Format LaTeX pour le rapport
    latex = df_display.to_latex(index=False, float_format="%.4f")
    with open(os.path.join(CHEMIN_TABLES, 'comparaison_modeles.tex'), 'w') as f:
        f.write(latex)
    print(f"  ✓ Tableau LaTeX sauvegardé : comparaison_modeles.tex")

    return df_display


# ==============================================================
#  ÉTAPE 5 — VISUALISATION COMPARATIVE
# ==============================================================
def visualisation_comparative(df_results):
    """
    Crée les graphiques de comparaison des modèles.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Visualisation comparative")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── 5.1 Barplot des métriques principales
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    x = np.arange(len(df_results))
    width = 0.2

    for i, metric in enumerate(metrics):
        axes[0, 0].bar(x + i * width, df_results[metric], width,
                       label=metric, alpha=0.8, edgecolor='black')

    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(df_results['Modèle'], rotation=15, ha='right')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Comparaison des métriques par modèle')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].set_ylim(0, 1.1)

    # ── 5.2 Radar chart (simplifié en barplot groupé)
    df_results.set_index('Modèle')[['Accuracy', 'F1-score', 'AUC-ROC']].plot(
        kind='bar', ax=axes[0, 1], color=['#3498db', '#2ecc71', '#e74c3c'],
        edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Accuracy vs F1-score vs AUC-ROC')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].set_ylim(0, 1.1)

    # ── 5.3 Temps d'entraînement
    axes[1, 0].bar(df_results['Modèle'], df_results['Temps (s)'],
                   color='coral', edgecolor='black', alpha=0.8)
    axes[1, 0].set_title('Temps d\'entraînement (secondes)')
    axes[1, 0].set_ylabel('Temps (s)')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # ── 5.4 Ranking global (score moyen normalisé)
    df_results['Score_moyen'] = df_results[['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC']].mean(axis=1)
    ranking = df_results.sort_values('Score_moyen', ascending=True)

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(ranking)))
    axes[1, 1].barh(ranking['Modèle'], ranking['Score_moyen'], color=colors, edgecolor='black')
    axes[1, 1].set_xlabel('Score moyen')
    axes[1, 1].set_title('Ranking global des modèles')
    axes[1, 1].set_xlim(0, 1)

    for i, (idx, row) in enumerate(ranking.iterrows()):
        axes[1, 1].text(row['Score_moyen'] + 0.01, i, f"{row['Score_moyen']:.3f}",
                        va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'comparaison_modeles.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : comparaison_modeles.png")

    return ranking


# ==============================================================
#  ÉTAPE 6 — MATRICES DE CONFUSION
# ==============================================================
def matrices_confusion(modeles, y_test):
    """
    Affiche les matrices de confusion pour chaque modèle.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Matrices de confusion")
    print("=" * 60)

    n_modeles = len(modeles)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    class_names = ['inappropriate', 'appropriate', 'not_applicable']

    for idx, (nom, info) in enumerate(modeles.items()):
        y_pred = info['y_pred']
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names[:cm.shape[1]],
                    yticklabels=class_names[:cm.shape[0]])
        axes[idx].set_title(f'Matrice de confusion – {nom}')
        axes[idx].set_ylabel('Réel')
        axes[idx].set_xlabel('Prédit')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'matrices_confusion.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : matrices_confusion.png")


# ==============================================================
#  ÉTAPE 7 — COURBES ROC (MULTICLASSE)
# ==============================================================
def courbes_roc(modeles, y_test):
    """
    Trace les courbes ROC pour chaque modèle (one-vs-rest).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : Courbes ROC (One-vs-Rest)")
    print("=" * 60)

    from sklearn.preprocessing import label_binarize

    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (nom, info) in enumerate(modeles.items()):
        y_proba = info['y_proba']

        # Calcul ROC pour chaque classe
        from sklearn.metrics import roc_curve, auc

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            if y_proba.shape[1] > i:
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                axes[idx].plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

        axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('Taux de Faux Positifs')
        axes[idx].set_ylabel('Taux de Vrais Positifs')
        axes[idx].set_title(f'ROC – {nom}')
        axes[idx].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'courbes_roc.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : courbes_roc.png")


# ==============================================================
#  ÉTAPE 8 — RECOMMANDATION FINALE
# ==============================================================
def recommandation_finale(df_results, ranking):
    """
    Produit la recommandation finale pour le rapport.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 8 : Recommandation finale")
    print("=" * 60)

    meilleur = ranking.iloc[-1]  # Dernier = meilleur score
    deuxieme = ranking.iloc[-2] if len(ranking) > 1 else None

    print(f"\n  🏆 MEILLEUR MODÈLE : {meilleur['Modèle']}")
    print(f"     Score moyen : {meilleur['Score_moyen']:.4f}")
    print(f"     Accuracy    : {meilleur['Accuracy']:.4f}")
    print(f"     F1-score    : {meilleur['F1-score']:.4f}")
    print(f"     AUC-ROC     : {meilleur['AUC-ROC']:.4f}")

    print(f"\n  📋 RECOMMANDATION POUR LE SYSTÈME D'AIDE AU DIAGNOSTIC :")

    if meilleur['Modèle'] == 'Random Forest':
        print("""
    Le Random Forest est recommandé comme modèle principal :
    - Meilleure performance globale (accuracy, F1-score)
    - Feature importance : identifie les variables à améliorer
    - Robustesse au bruit et aux outliers

    Le Decision Tree est recommandé comme modèle explicatif :
    - Règles de décision directement exploitables par les médecins
    - Visualisation intuitive pour la formation médicale
        """)
    elif meilleur['Modèle'] == 'Decision Tree':
        print("""
    Le Decision Tree offre le meilleur équilibre performance/interprétabilité :
    - Règles cliniques exploitables immédiatement
    - Performance proche du Random Forest
    - Idéal pour un système d'aide à la décision transparent
        """)

    # Sauvegarde texte
    with open(os.path.join(CHEMIN_TABLES, 'recommandation_finale.txt'), 'w', encoding='utf-8') as f:
        f.write("RECOMMANDATION FINALE – SYSTÈME D'AIDE AU DIAGNOSTIC ORL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Meilleur modèle : {meilleur['Modèle']}\n")
        f.write(f"Score moyen : {meilleur['Score_moyen']:.4f}\n\n")
        f.write("Tableau comparatif complet :\n")
        f.write(df_results.to_string(index=False))

    print(f"\n  ✓ Recommandation sauvegardée : recommandation_finale.txt")

    return meilleur['Modèle']


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_comparison():
    """
    Exécute le pipeline complet de comparaison.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║     COMPARAISON FINALE – Évaluation des Modèles ORL     ║")
    print("╚" + "═" * 58 + "╝")

    X_train_bal, X_test, y_train_bal, y_test, noms_features = charger_donnees()
    modeles = entrainer_modeles(X_train_bal, y_train_bal, X_test, y_test)
    df_results = calculer_metriques(modeles, y_test)
    df_display = tableau_comparatif(df_results)
    ranking = visualisation_comparative(df_results)
    matrices_confusion(modeles, y_test)
    courbes_roc(modeles, y_test)
    meilleur = recommandation_finale(df_results, ranking)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  COMPARAISON TERMINÉ – Meilleur modèle identifié        ║")
    print("╚" + "═" * 58 + "╝\n")

    return df_results, meilleur


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    results, meilleur_modele = pipeline_comparison()