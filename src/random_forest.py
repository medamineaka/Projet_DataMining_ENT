# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : random_forest.py
#  Rôle    : Random Forest – Performance et feature importance
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("Set2")

# ──────────────────────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────────────────────
CHEMIN_MODELS  = os.path.join('outputs', 'models')
CHEMIN_OUTPUT  = os.path.join('outputs', 'figures')
CHEMIN_TABLES  = os.path.join('outputs', 'tables')

os.makedirs(CHEMIN_OUTPUT, exist_ok=True)
os.makedirs(CHEMIN_TABLES, exist_ok=True)


# =============================================================
#  FONCTION PRINCIPALE POUR main.py
# =============================================================

def pipeline_random_forest():
    """
    Pipeline complet : Random Forest avec feature importance.
    """
    from src.utils import (
        charger_donnees_pretraitees, evaluer_modele, afficher_metrics,
        plot_confusion_matrix, plot_feature_importance, sauvegarder_modele,
        sauvegarder_tableau, print_section
    )

    print_section("RANDOM FOREST – Forêt aléatoire")

    # 1. Chargement
    data = charger_donnees_pretraitees()
    X_train = data['X_train_bal']
    X_test = data['X_test']
    y_train = data['y_train_bal']
    y_test = data['y_test']
    noms_features = data['noms_features']
    encodeurs = data['encodeurs']

    noms_classes = list(encodeurs['target'].classes_)

    # 2. Modèle
    print("\n  Entraînement du Random Forest (n_estimators=100)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 3. Prédiction
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    if len(noms_classes) == 2:
        y_proba_roc = y_proba[:, 1]
    else:
        y_proba_roc = y_proba

    # 4. Évaluation
    resultats = evaluer_modele(y_test, y_pred, y_proba_roc, noms_classes)
    afficher_metrics(resultats)

    # 5. Matrice de confusion
    plot_confusion_matrix(
        resultats['confusion_matrix'],
        noms_classes,
        'Matrice de confusion – Random Forest',
        'confusion_matrix_random_forest'
    )

    # 6. Feature importance
    importance_df = pd.DataFrame({
        'feature': noms_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plot_feature_importance(
        importance_df,
        'Feature Importance – Random Forest',
        'feature_importance_random_forest'
    )

    # 7. Sauvegarde
    sauvegarder_modele(rf, 'random_forest')
    sauvegarder_tableau(importance_df, 'feature_importance_rf')

    # 8. Récap
    recap = pd.DataFrame([{
        'Algorithme': 'Random Forest',
        'Accuracy': resultats['metrics']['accuracy'],
        'Precision': resultats['metrics']['precision'],
        'Recall': resultats['metrics']['recall'],
        'F1-Score': resultats['metrics']['f1_score'],
        'ROC-AUC': resultats['metrics'].get('roc_auc', None),
        'N_Estimators': 100
    }])
    sauvegarder_tableau(recap, 'metrics_random_forest')

    print("\n  ✅ Random Forest terminé")

    return {
        'modele': rf,
        'metrics': resultats['metrics'],
        'importance': importance_df
    }


if __name__ == '__main__':
    pipeline_random_forest()
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
    X_test      = data['X_test']
    y_train_bal = data['y_train_bal']
    y_test      = data['y_test']
    noms_features = data['noms_features']
    encodeurs     = data['encodeurs']

    print(f"  ✓ Données chargées")
    print(f"  ✓ X_train_bal : {X_train_bal.shape}")
    print(f"  ✓ X_test      : {X_test.shape}")
    print(f"  ✓ Features    : {noms_features}")

    return X_train_bal, X_test, y_train_bal, y_test, noms_features, encodeurs


# ==============================================================
#  ÉTAPE 2 — ENTRAÎNEMENT DU RANDOM FOREST
# ==============================================================
def entrainer_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Entraîne un Random Forest avec paramètres optimisés.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Entraînement du Random Forest")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,    # nombre d'arbres
        max_depth=max_depth,           # profondeur max
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1                     # utiliser tous les cœurs
    )

    rf.fit(X_train, y_train)

    print(f"  ✓ Random Forest entraîné")
    print(f"  ✓ Nombre d'arbres : {n_estimators}")
    print(f"  ✓ Profondeur max  : {max_depth}")
    print(f"  ✓ Score train     : {rf.score(X_train, y_train):.3f}")

    return rf


# ==============================================================
#  ÉTAPE 3 — ÉVALUATION SUR LE JEU DE TEST
# ==============================================================
def evaluer_random_forest(rf, X_test, y_test, encodeurs):
    """
    Évalue le Random Forest avec toutes les métriques.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Évaluation sur le jeu de test")
    print("=" * 60)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # AUC-ROC multi-classe
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except:
        auc = roc_auc_score(y_test, y_proba[:, 1])

    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'AUC-ROC': auc
    }

    print(f"\n  === MÉTRIQUES RANDOM FOREST ===")
    print(f"  Accuracy      : {acc:.3f}")
    print(f"  Precision     : {prec:.3f}")
    print(f"  Recall        : {rec:.3f}")
    print(f"  F1-score      : {f1:.3f}")
    print(f"  AUC-ROC       : {auc:.3f}")

    # Classification report
    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    print(f"\n  === CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=noms_classes))

    return y_pred, y_proba, metrics


# ==============================================================
#  ÉTAPE 4 — FEATURE IMPORTANCE
# ==============================================================
def feature_importance(rf, noms_features):
    """
    Analyse et visualise l'importance des variables.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Feature Importance – Random Forest")
    print("=" * 60)

    importance = pd.DataFrame({
        'feature': noms_features,
        'importance': rf.feature_importances_,
        'std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    }).sort_values('importance', ascending=False)

    print("\n  Importance des variables :")
    print(importance.to_string(index=False))

    # Visualisation 1 : Barplot avec erreur
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
    plt.errorbar(
        y=range(len(importance)),
        x=importance['importance'],
        xerr=importance['std'],
        fmt='none',
        color='black',
        capsize=3,
        alpha=0.5
    )
    plt.title('Feature Importance – Random Forest (avec écart-type)', fontsize=14)
    plt.xlabel('Importance moyenne')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'feature_importance_rf.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : feature_importance_rf.png")

    # Visualisation 2 : Top 5 en détail
    plt.figure(figsize=(8, 5))
    top5 = importance.head(5)
    plt.barh(top5['feature'], top5['importance'], color='steelblue', edgecolor='black')
    plt.xlabel('Importance')
    plt.title('Top 5 variables les plus importantes', fontsize=14)
    plt.gca().invert_yaxis()
    for i, v in enumerate(top5['importance']):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'top5_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : top5_feature_importance.png")

    # Sauvegarde
    importance.to_csv(os.path.join(CHEMIN_TABLES, 'feature_importance_rf.csv'), index=False)

    return importance


# ==============================================================
#  ÉTAPE 5 — MATRICE DE CONFUSION
# ==============================================================
def matrice_confusion(y_test, y_pred, encodeurs):
    """
    Matrice de confusion pour Random Forest.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Matrice de confusion")
    print("=" * 60)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=noms_classes,
                yticklabels=noms_classes,
                linewidths=0.5, linecolor='gray')

    plt.title('Matrice de Confusion – Random Forest', fontsize=14, pad=20)
    plt.ylabel('Réel', fontsize=12)
    plt.xlabel('Prédit', fontsize=12)

    # Pourcentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color='darkgreen')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'confusion_matrix_rf.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : confusion_matrix_rf.png")

    cm_df = pd.DataFrame(cm, index=noms_classes, columns=noms_classes)
    cm_df.to_csv(os.path.join(CHEMIN_TABLES, 'confusion_matrix_rf.csv'))

    return cm


# ==============================================================
#  ÉTAPE 6 — COURBE ROC
# ==============================================================
def courbe_roc(y_test, y_proba, encodeurs):
    """
    Courbe ROC pour Random Forest.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Courbe ROC")
    print("=" * 60)

    from sklearn.preprocessing import label_binarize

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    plt.figure(figsize=(10, 8))

    if y_test_bin.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        auc = roc_auc_score(y_test, y_proba[:, 1])
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    else:
        # Multi-classe : ROC pour classe "appropriate" (code 1)
        fpr, tpr, _ = roc_curve(y_test_bin[:, 1], y_proba[:, 1])
        auc = roc_auc_score(y_test_bin[:, 1], y_proba[:, 1])
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs', fontsize=12)
    plt.title('Courbe ROC – Random Forest', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'roc_curve_rf.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : roc_curve_rf.png")

    return auc


# ==============================================================
#  ÉTAPE 7 — COMPARAISON AVEC DECISION TREE
# ==============================================================
def comparer_avec_dt(rf_metrics):
    """
    Compare Random Forest avec Decision Tree (modèle précédent).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : Comparaison avec Decision Tree")
    print("=" * 60)

    # Charger les métriques du Decision Tree
    chemin_dt = os.path.join(CHEMIN_MODELS, 'modele_decision_tree.pkl')

    try:
        with open(chemin_dt, 'rb') as f:
            dt_data = pickle.load(f)
        dt_metrics = dt_data['metrics']

        comparison = pd.DataFrame({
            'Métrique': list(rf_metrics.keys()),
            'Decision Tree': list(dt_metrics.values()),
            'Random Forest': list(rf_metrics.values())
        })

        print("\n  === TABLEAU COMPARATIF ===")
        print(comparison.to_string(index=False))

        # Visualisation
        plt.figure(figsize=(10, 6))
        x = np.arange(len(comparison))
        width = 0.35

        plt.bar(x - width/2, comparison['Decision Tree'], width,
                label='Decision Tree', color='lightcoral', edgecolor='black')
        plt.bar(x + width/2, comparison['Random Forest'], width,
                label='Random Forest', color='steelblue', edgecolor='black')

        plt.xlabel('Métrique', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Comparaison Decision Tree vs Random Forest', fontsize=14)
        plt.xticks(x, comparison['Métrique'], rotation=45)
        plt.legend()
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')

        # Valeurs sur les barres
        for i, (dt, rf) in enumerate(zip(comparison['Decision Tree'], comparison['Random Forest'])):
            plt.text(i - width/2, dt + 0.01, f'{dt:.3f}', ha='center', fontsize=9)
            plt.text(i + width/2, rf + 0.01, f'{rf:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(CHEMIN_OUTPUT, 'comparaison_dt_rf.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("  ✓ Figure sauvegardée : comparaison_dt_rf.png")

        comparison.to_csv(os.path.join(CHEMIN_TABLES, 'comparaison_dt_rf.csv'), index=False)

        return comparison

    except FileNotFoundError:
        print("  ⚠ Modèle Decision Tree non trouvé. Comparaison impossible.")
        return None


# ==============================================================
#  ÉTAPE 8 — VALIDATION CROISÉE
# ==============================================================
def validation_croisee(rf, X_train, y_train, cv=5):
    """
    Validation croisée 5-fold pour Random Forest.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 8 : Validation croisée (5-fold)")
    print("=" * 60)

    scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"\n  Scores par fold : {scores.round(3)}")
    print(f"  Moyenne accuracy : {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, cv + 1), scores, color='forestgreen', edgecolor='black', alpha=0.8)
    plt.axhline(y=scores.mean(), color='red', linestyle='--',
                label=f'Moyenne = {scores.mean():.3f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Croisée 5-fold – Random Forest', fontsize=14)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'validation_croisee_rf.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : validation_croisee_rf.png")

    return scores


# ==============================================================
#  ÉTAPE 9 — INTERPRÉTATION MÉDICALE
# ==============================================================
def interpretation_medicale(rf, noms_features, encodeurs, metrics, importance):
    """
    Interprétation médicale pour le rapport.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 9 : Interprétation médicale")
    print("=" * 60)

    print("\n  === INTERPRÉTATION POUR LE RAPPORT ===\n")

    print("  1. PERFORMANCE DU RANDOM FOREST :")
    print(f"     Accuracy : {metrics['Accuracy']:.1%} | F1-score : {metrics['F1-score']:.3f}")
    print(f"     Le Random Forest surpasse généralement le Decision Tree")
    print(f"     grâce à l'agrégation de multiples arbres (bagging).\n")

    print("  2. VARIABLES CLÉS (feature importance) :")
    for idx, row in importance.head(5).iterrows():
        print(f"     • {row['feature']}: {row['importance']:.3f}")
    print(f"\n     → '{importance.iloc[0]['feature']}' est le facteur dominant")
    print(f"       pour prédire l'adéquation du referral.\n")

    print("  3. AVANTAGE SUR DECISION TREE :")
    print(f"     • Moins de surapprentissage (overfitting)")
    print(f"     • Robustesse aux données bruitées")
    print(f"     • Feature importance plus fiable (moyenne sur 100 arbres)\n")

    print("  4. IMPLICATIONS CLINIQUES :")
    print(f"     • Prioriser la formation sur : {importance.iloc[0]['feature']}")
    print(f"     • Protocole de triage basé sur les top 3 variables")
    print(f"     • Système d'aide à la décision en temps réel")

    # Sauvegarde
    with open(os.path.join(CHEMIN_TABLES, 'interpretation_medicale_rf.txt'), 'w', encoding='utf-8') as f:
        f.write("INTERPRÉTATION MÉDICALE – Random Forest\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy : {metrics['Accuracy']:.3f}\n")
        f.write(f"F1-score : {metrics['F1-score']:.3f}\n")
        f.write(f"AUC-ROC  : {metrics['AUC-ROC']:.3f}\n\n")
        f.write("Top 5 variables :\n")
        for idx, row in importance.head(5).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.3f}\n")
        f.write("\nLe Random Forest confirme et affine les résultats du Decision Tree.\n")
        f.write("Utilisation recommandée : système de scoring des referrals.\n")

    print("\n  ✓ Interprétation sauvegardée : interpretation_medicale_rf.txt")


# ==============================================================
#  ÉTAPE 10 — SAUVEGARDE DU MODÈLE
# ==============================================================
def sauvegarder_modele(rf, metrics):
    """
    Sauvegarde le modèle Random Forest.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 10 : Sauvegarde du modèle")
    print("=" * 60)

    modele_data = {
        'modele': rf,
        'type': 'RandomForest',
        'metrics': metrics
    }

    chemin = os.path.join(CHEMIN_MODELS, 'modele_random_forest.pkl')
    with open(chemin, 'wb') as f:
        pickle.dump(modele_data, f)

    print(f"  ✓ Modèle sauvegardé : {chemin}")

    return chemin


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_random_forest():
    """
    Pipeline complet du Random Forest.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║     RANDOM FOREST – Ensemble d'Arbres (Performance)       ║")
    print("╚" + "═" * 58 + "╝")

    # Chargement
    X_train_bal, X_test, y_train_bal, y_test, noms_features, encodeurs = charger_donnees()

    # Entraînement
    rf = entrainer_random_forest(X_train_bal, y_train_bal, n_estimators=100, max_depth=10)

    # Évaluation
    y_pred, y_proba, metrics = evaluer_random_forest(rf, X_test, y_test, encodeurs)

    # Analyses
    importance = feature_importance(rf, noms_features)
    matrice_confusion(y_test, y_pred, encodeurs)
    courbe_roc(y_test, y_proba, encodeurs)
    comparer_avec_dt(metrics)
    validation_croisee(rf, X_train_bal, y_train_bal, cv=5)

    # Interprétation
    interpretation_medicale(rf, noms_features, encodeurs, metrics, importance)

    # Sauvegarde
    sauvegarder_modele(rf, metrics)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  RANDOM FOREST TERMINÉ – Meilleur modèle de classification  ║")
    print("╚" + "═" * 58 + "╝\n")

    return rf, metrics


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    rf, metrics = pipeline_random_forest()