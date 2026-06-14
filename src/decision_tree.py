# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : decision_tree.py
#  Rôle    : Arbre de décision (CART) – Modèle central du projet
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (20, 12)
sns.set_palette("Set2")

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
    print("=" * 60)
    print("ÉTAPE 1 : Chargement des données prétraitées")
    print("=" * 60)

    chemin_pkl = os.path.join(CHEMIN_MODELS, 'donnees_preprocessees.pkl')

    with open(chemin_pkl, 'rb') as f:
        data = pickle.load(f)

    X_train_bal   = data['X_train_bal']
    X_test        = data['X_test']
    y_train_bal   = data['y_train_bal']
    y_test        = data['y_test']
    noms_features = data['noms_features']
    encodeurs     = data['encodeurs']

    print(f"  ✓ Données chargées")
    print(f"  ✓ X_train_bal : {X_train_bal.shape}")
    print(f"  ✓ X_test      : {X_test.shape}")
    print(f"  ✓ Features    : {noms_features}")

    return X_train_bal, X_test, y_train_bal, y_test, noms_features, encodeurs


# ==============================================================
#  ÉTAPE 2 — ENTRAÎNEMENT DU DECISION TREE
# ==============================================================
def entrainer_arbre(X_train, y_train, max_depth=4, min_samples_split=20):
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Entraînement du Decision Tree")
    print("=" * 60)

    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )

    dt.fit(X_train, y_train)

    print(f"  ✓ Arbre entraîné")
    print(f"  ✓ Profondeur max : {max_depth}")
    print(f"  ✓ Nombre de feuilles : {dt.get_n_leaves()}")
    print(f"  ✓ Score train : {dt.score(X_train, y_train):.3f}")

    return dt


# ==============================================================
#  ÉTAPE 3 — ÉVALUATION SUR LE JEU DE TEST
# ==============================================================
def evaluer_arbre(dt, X_test, y_test, noms_features, encodeurs):
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Évaluation sur le jeu de test")
    print("=" * 60)

    y_pred  = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)   # shape (n, 3) – multiclasse

    # Métriques de base
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')

    # AUC-ROC multiclasse (OVR)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        auc = None

    print(f"\n  === MÉTRIQUES ===")
    print(f"  Accuracy      : {acc:.3f}")
    print(f"  Precision     : {prec:.3f}")
    print(f"  Recall        : {rec:.3f}")
    print(f"  F1-score      : {f1:.3f}")
    print(f"  AUC-ROC       : {auc:.3f}" if auc is not None else "  AUC-ROC       : N/A")

    le_cible    = encodeurs['target']
    noms_classes = le_cible.classes_

    print(f"\n  === CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=noms_classes))

    metrics = {
        'Accuracy'  : acc,
        'Precision' : prec,
        'Recall'    : rec,
        'F1-score'  : f1,
        'AUC-ROC'   : auc
    }

    return y_pred, y_proba, metrics


# ==============================================================
#  ÉTAPE 4 — MATRICE DE CONFUSION
# ==============================================================
def matrice_confusion(y_test, y_pred, encodeurs):
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Matrice de confusion")
    print("=" * 60)

    le_cible     = encodeurs['target']
    noms_classes = le_cible.classes_

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=noms_classes,
                yticklabels=noms_classes,
                linewidths=0.5, linecolor='gray')

    plt.title('Matrice de Confusion – Decision Tree', fontsize=14, pad=20)
    plt.ylabel('Réel', fontsize=12)
    plt.xlabel('Prédit', fontsize=12)

    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                     ha='center', va='center', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'confusion_matrix_dt.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : confusion_matrix_dt.png")

    cm_df = pd.DataFrame(cm, index=noms_classes, columns=noms_classes)
    cm_df.to_csv(os.path.join(CHEMIN_TABLES, 'confusion_matrix_dt.csv'))

    return cm


# ==============================================================
#  ÉTAPE 5 — COURBE ROC (One-vs-Rest multiclasse)
# ==============================================================
def courbe_roc(y_test, y_proba, encodeurs):
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Courbe ROC")
    print("=" * 60)

    from sklearn.preprocessing import label_binarize

    le_cible     = encodeurs['target']
    noms_classes = le_cible.classes_
    classes_idx  = np.unique(y_test)

    y_test_bin = label_binarize(y_test, classes=classes_idx)
    n_classes  = y_test_bin.shape[1]

    plt.figure(figsize=(10, 8))
    colors = ['steelblue', 'darkorange', 'green']

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        label = noms_classes[classes_idx[i]] if i < len(noms_classes) else f'Classe {i}'
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 linewidth=2, label=f'{label} (AUC = {auc_i:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
    plt.title('Courbe ROC – Decision Tree (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'roc_curve_dt.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : roc_curve_dt.png")


# ==============================================================
#  ÉTAPE 6 — VISUALISATION DE L'ARBRE
# ==============================================================
def visualiser_arbre(dt, noms_features, encodeurs):
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Visualisation de l'arbre de décision")
    print("=" * 60)

    le_cible     = encodeurs['target']
    noms_classes = list(le_cible.classes_)

    plt.figure(figsize=(28, 18))
    plot_tree(dt, feature_names=noms_features, class_names=noms_classes,
              filled=True, rounded=True, fontsize=10, max_depth=4,
              proportion=True, precision=2)
    plt.title('Arbre de Décision – Prédiction du Referral ORL\n(profondeur max=4)',
              fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'arbre_decision_complet.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : arbre_decision_complet.png")

    plt.figure(figsize=(24, 14))
    plot_tree(dt, feature_names=noms_features, class_names=noms_classes,
              filled=True, rounded=True, fontsize=11, max_depth=3,
              proportion=True, precision=2)
    plt.title('Arbre de Décision – Vue Simplifiée (profondeur=3)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'arbre_decision_simplifie.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : arbre_decision_simplifie.png")


# ==============================================================
#  ÉTAPE 7 — EXTRACTION DES RÈGLES DE DÉCISION
# ==============================================================
def extraire_regles(dt, noms_features, encodeurs):
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : Extraction des règles de décision")
    print("=" * 60)

    le_cible     = encodeurs['target']
    noms_classes = le_cible.classes_

    regles = export_text(dt, feature_names=noms_features, max_depth=4)

    print("\n  === RÈGLES DE DÉCISION (texte) ===")
    print(regles)

    with open(os.path.join(CHEMIN_TABLES, 'regles_decision_tree.txt'), 'w', encoding='utf-8') as f:
        f.write("RÈGLES DE DÉCISION – Decision Tree\n")
        f.write("=" * 60 + "\n\n")
        f.write(regles)

    print("  ✓ Règles sauvegardées : regles_decision_tree.txt")

    tree = dt.tree_

    def parcourir_arbre(node_id=0, chemin="", profondeur=0, max_prof=4):
        if profondeur > max_prof:
            return []
        if tree.feature[node_id] == -2:
            classe = noms_classes[np.argmax(tree.value[node_id])]
            proba  = np.max(tree.value[node_id]) / np.sum(tree.value[node_id])
            return [{'chemin': chemin, 'classe': classe,
                     'probabilite': proba,
                     'echantillons': int(tree.n_node_samples[node_id])}]
        feature = noms_features[tree.feature[node_id]]
        seuil   = tree.threshold[node_id]
        return (
            parcourir_arbre(tree.children_left[node_id],
                            chemin + f"{feature} <= {seuil:.2f} → ",
                            profondeur + 1, max_prof) +
            parcourir_arbre(tree.children_right[node_id],
                            chemin + f"{feature} > {seuil:.2f} → ",
                            profondeur + 1, max_prof)
        )

    regles_list = parcourir_arbre()
    regles_df   = pd.DataFrame(regles_list)
    regles_df   = regles_df.sort_values('probabilite', ascending=False).head(10)

    print("\n  === TOP 10 RÈGLES ===")
    for idx, row in regles_df.iterrows():
        print(f"\n  Règle {idx+1}:")
        print(f"    Chemin      : {row['chemin'][:100]}...")
        print(f"    Classe      : {row['classe']}")
        print(f"    Probabilité : {row['probabilite']:.3f}")
        print(f"    Échantillons: {row['echantillons']}")

    regles_df.to_csv(os.path.join(CHEMIN_TABLES, 'top_regles_decision_tree.csv'), index=False)

    return regles_df


# ==============================================================
#  ÉTAPE 8 — VALIDATION CROISÉE
# ==============================================================
def validation_croisee(dt, X_train, y_train, cv=5):
    print("\n" + "=" * 60)
    print("ÉTAPE 8 : Validation croisée (5-fold)")
    print("=" * 60)

    scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"\n  Scores par fold : {scores.round(3)}")
    print(f"  Moyenne accuracy : {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, cv + 1), scores, color='steelblue', edgecolor='black', alpha=0.8)
    plt.axhline(y=scores.mean(), color='red', linestyle='--',
                label=f'Moyenne = {scores.mean():.3f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Croisée 5-fold – Decision Tree', fontsize=14)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'validation_croisee_dt.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : validation_croisee_dt.png")

    return scores


# ==============================================================
#  ÉTAPE 9 — INTERPRÉTATION MÉDICALE
# ==============================================================
def interpretation_medicale(dt, noms_features, encodeurs, metrics):
    print("\n" + "=" * 60)
    print("ÉTAPE 9 : Interprétation médicale")
    print("=" * 60)

    importance = pd.DataFrame({
        'feature'   : noms_features,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  === INTERPRÉTATION POUR LE RAPPORT ===\n")
    print(f"  Accuracy : {metrics['Accuracy']:.1%}")
    print(f"  F1-score : {metrics['F1-score']:.3f}")
    auc_val = metrics['AUC-ROC']
    print(f"  AUC-ROC  : {auc_val:.3f}" if auc_val else "  AUC-ROC  : N/A")

    print("\n  Variables clés :")
    for _, row in importance.head(5).iterrows():
        print(f"    • {row['feature']}: {row['importance']:.3f}")

    with open(os.path.join(CHEMIN_TABLES, 'interpretation_medicale_dt.txt'), 'w', encoding='utf-8') as f:
        f.write("INTERPRÉTATION MÉDICALE – Decision Tree\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy : {metrics['Accuracy']:.3f}\n")
        f.write(f"F1-score : {metrics['F1-score']:.3f}\n")
        f.write(f"AUC-ROC  : {metrics['AUC-ROC']}\n\n")
        f.write("Variables clés :\n")
        for _, row in importance.head(5).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.3f}\n")

    print("\n  ✓ Interprétation sauvegardée : interpretation_medicale_dt.txt")


# ==============================================================
#  ÉTAPE 10 — SAUVEGARDE DU MODÈLE
# ==============================================================
def sauvegarder_modele(dt, metrics):
    print("\n" + "=" * 60)
    print("ÉTAPE 10 : Sauvegarde du modèle")
    print("=" * 60)

    chemin = os.path.join(CHEMIN_MODELS, 'modele_decision_tree.pkl')
    with open(chemin, 'wb') as f:
        pickle.dump({'modele': dt, 'type': 'DecisionTree', 'metrics': metrics}, f)

    print(f"  ✓ Modèle sauvegardé : {chemin}")
    return chemin


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_decision_tree():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║    DECISION TREE – Arbre de Décision CART (Modèle Central)  ║")
    print("╚" + "═" * 58 + "╝")

    X_train_bal, X_test, y_train_bal, y_test, noms_features, encodeurs = charger_donnees()
    dt = entrainer_arbre(X_train_bal, y_train_bal, max_depth=4, min_samples_split=20)
    y_pred, y_proba, metrics = evaluer_arbre(dt, X_test, y_test, noms_features, encodeurs)

    matrice_confusion(y_test, y_pred, encodeurs)
    courbe_roc(y_test, y_proba, encodeurs)
    visualiser_arbre(dt, noms_features, encodeurs)
    extraire_regles(dt, noms_features, encodeurs)
    validation_croisee(dt, X_train_bal, y_train_bal, cv=5)
    interpretation_medicale(dt, noms_features, encodeurs, metrics)
    sauvegarder_modele(dt, metrics)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  DECISION TREE TERMINÉ – Modèle central prêt pour rapport   ║")
    print("╚" + "═" * 58 + "╝\n")

    return dt, metrics


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    dt, metrics = pipeline_decision_tree()