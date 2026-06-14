# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : naive_bayes.py
#  Rôle    : Classification Bayésienne Naïve (GaussianNB)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

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

def pipeline_naive_bayes():
    """
    Pipeline complet : Naive Bayes (GaussianNB).
    """
    from src.utils import (
        charger_donnees_pretraitees, evaluer_modele, afficher_metrics,
        plot_confusion_matrix, plot_roc_curve, sauvegarder_modele,
        sauvegarder_tableau, print_section
    )

    print_section("NAIVE BAYES – Classification bayésienne naïve")

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
    print("\n  Entraînement du Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # 3. Prédiction + probabilités
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)

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
        'Matrice de confusion – Naive Bayes',
        'confusion_matrix_naive_bayes'
    )

    # 6. ROC curve
    if len(noms_classes) == 2:
        plot_roc_curve(
            y_test, y_proba_roc,
            'Courbe ROC – Naive Bayes',
            'roc_curve_naive_bayes'
        )

    # 7. Exemples de probabilités
    print("\n  === PROBABILITÉS DE CLASSE (10 premiers patients) ===")
    for i in range(min(10, len(y_test))):
        vrai = noms_classes[y_test[i]]
        pred = noms_classes[y_pred[i]]
        proba_str = " | ".join([f"{noms_classes[j]}={y_proba[i][j]:.3f}"
                                for j in range(len(noms_classes))])
        print(f"  Patient {i}: [{vrai}] → prédit [{pred}] | {proba_str}")

    # 8. Sauvegarde
    sauvegarder_modele(nb, 'naive_bayes')

    # 9. Récap
    recap = pd.DataFrame([{
        'Algorithme': 'Naive Bayes',
        'Accuracy': resultats['metrics']['accuracy'],
        'Precision': resultats['metrics']['precision'],
        'Recall': resultats['metrics']['recall'],
        'F1-Score': resultats['metrics']['f1_score'],
        'ROC-AUC': resultats['metrics'].get('roc_auc', None),
        'Variante': 'GaussianNB'
    }])
    sauvegarder_tableau(recap, 'metrics_naive_bayes')

    print("\n  ✅ Naive Bayes terminé")

    return {
        'modele': nb,
        'metrics': resultats['metrics'],
        'proba': y_proba
    }


if __name__ == '__main__':
    pipeline_naive_bayes()

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
#  ÉTAPE 2 — ENTRAÎNEMENT DU MODÈLE NAIVE BAYES
# ==============================================================
def entrainer_naive_bayes(X_train, y_train):
    """
    Entraîne un GaussianNB sur les données équilibrées (SMOTE).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Entraînement Naive Bayes (GaussianNB)")
    print("=" * 60)

    # GaussianNB est adapté aux features numériques continues
    # Les variables catégorielles encodées sont traitées comme continues
    model = GaussianNB()
    model.fit(X_train, y_train)

    print("  ✓ Modèle entraîné")
    print(f"  ✓ Classes apprises : {model.classes_}")
    print(f"  ✓ P(class) a priori  : {model.class_prior_.round(4)}")

    # Sauvegarde du modèle
    chemin_model = os.path.join(CHEMIN_MODELS, 'naive_bayes_model.pkl')
    with open(chemin_model, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Modèle sauvegardé → {chemin_model}")

    return model


# ==============================================================
#  ÉTAPE 3 — PRÉDICTION ET PROBABILITÉS
# ==============================================================
def predire(model, X_test, encodeurs):
    """
    Effectue les prédictions et extrait les probabilités.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Prédictions et probabilités")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    print(f"  ✓ Prédictions effectuées : {len(y_pred)} patients")
    print(f"\n  Distribution des prédictions :")
    unique, counts = np.unique(y_pred, return_counts=True)
    for u, c in zip(unique, counts):
        nom = noms_classes[u] if u < len(noms_classes) else f"classe_{u}"
        print(f"    {nom} : {c} ({c/len(y_pred)*100:.1f}%)")

    # Exemple de probabilités pour 5 patients
    print(f"\n  Exemples de probabilités (5 premiers patients) :")
    print(f"  {'Patient':<<10} {'Inappropriate':<<15} {'Appropriate':<<15} {'Prédiction'}")
    print("  " + "-" * 55)
    for i in range(min(5, len(y_proba))):
        proba_inap = y_proba[i][0]
        proba_app = y_proba[i][1]
        pred = noms_classes[y_pred[i]] if y_pred[i] < len(noms_classes) else str(y_pred[i])
        print(f"  {i+1:<10} {proba_inap:.3f}          {proba_app:.3f}          {pred}")

    return y_pred, y_proba


# ==============================================================
#  ÉTAPE 4 — ÉVALUATION DU MODÈLE
# ==============================================================
def evaluer_model(y_test, y_pred, y_proba, encodeurs):
    """
    Calcule toutes les métriques d'évaluation.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Évaluation du modèle")
    print("=" * 60)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    # Métriques globales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # AUC-ROC (one-vs-rest pour multiclass)
    try:
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        if y_proba.shape[1] == y_test_binarized.shape[1]:
            auc = roc_auc_score(y_test_binarized, y_proba, multi_class='ovr', average='weighted')
        else:
            auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba.shape[1] == 2 else 0.5
    except Exception as e:
        print(f"  ⚠ AUC calcul avec erreur : {e}")
        auc = 0.5

    print(f"\n  === MÉTRIQUES GLOBALES ===")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-score   : {f1:.4f}")
    print(f"  AUC-ROC    : {auc:.4f}")

    # Classification report détaillé
    print(f"\n  === RAPPORT PAR CLASSE ===")
    print(classification_report(y_test, y_pred, target_names=noms_classes))

    # Sauvegarde des métriques
    metrics = {
        'Algorithme': 'Naive Bayes',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'AUC-ROC': auc
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(CHEMIN_TABLES, 'metrics_naive_bayes.csv'), index=False)

    return metrics


# ==============================================================
#  ÉTAPE 5 — MATRICE DE CONFUSION
# ==============================================================
def matrice_confusion(y_test, y_pred, encodeurs):
    """
    Visualise la matrice de confusion.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Matrice de confusion")
    print("=" * 60)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=noms_classes,
                yticklabels=noms_classes,
                linewidths=0.5, linecolor='gray')
    plt.title('Matrice de Confusion – Naive Bayes', fontsize=14, pad=20)
    plt.ylabel('Réel', fontsize=12)
    plt.xlabel('Prédit', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'confusion_matrix_naive_bayes.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : confusion_matrix_naive_bayes.png")

    # Pourcentages par ligne
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print(f"\n  Pourcentages par classe réelle :")
    for i, nom in enumerate(noms_classes):
        print(f"    {nom} : ", end="")
        for j, nom_pred in enumerate(noms_classes):
            print(f"{cm_pct[i,j]:.1f}% prédit {nom_pred}  ", end="")
        print()

    return cm


# ==============================================================
#  ÉTAPE 6 — COURBE ROC
# ==============================================================
def courbe_roc(y_test, y_proba, encodeurs):
    """
    Trace la courbe ROC pour chaque classe (One-vs-Rest).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Courbe ROC")
    print("=" * 60)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    # Binariser y_test
    classes_uniques = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes_uniques)

    plt.figure(figsize=(10, 8))

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']

    for i, cls in enumerate(classes_uniques):
        if i < y_proba.shape[1] and i < len(noms_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc_cls = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
            nom = noms_classes[cls] if cls < len(noms_classes) else f"classe_{cls}"

            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f'{nom} (AUC = {auc_cls:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
    plt.title('Courbe ROC – Naive Bayes (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'roc_curve_naive_bayes.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : roc_curve_naive_bayes.png")


# ==============================================================
#  ÉTAPE 7 — INTERPRÉTATION MÉDICALE DES PROBABILITÉS
# ==============================================================
def interpretation_medicale(model, X_test, y_proba, noms_features, encodeurs):
    """
    Interprète les probabilités du modèle Bayésien dans un contexte médical.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : Interprétation médicale")
    print("=" * 60)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    # Trouver des exemples typiques
    print("\n  --- Cas typiques ---")

    # Patient avec forte probabilité d'être "inappropriate"
    idx_inap = np.argmax(y_proba[:, 0])  # classe 0 = inappropriate
    print(f"\n  CAS 1 – Referral probablement INAPPROPRIÉ (confiance = {y_proba[idx_inap, 0]:.1%})")
    print(f"    Caractéristiques :")
    for i, feat in enumerate(noms_features):
        val = X_test.iloc[idx_inap, i] if hasattr(X_test, 'iloc') else X_test[idx_inap, i]
        # Décoder si possible
        if feat in encodeurs and feat != 'target':
            try:
                val_decode = encodeurs[feat].inverse_transform([int(val)])[0]
                print(f"      {feat:25s} : {val_decode}")
            except:
                print(f"      {feat:25s} : {val}")
        else:
            print(f"      {feat:25s} : {val}")

    # Patient avec forte probabilité d'être "appropriate"
    idx_app = np.argmax(y_proba[:, 1]) if y_proba.shape[1] > 1 else 0
    print(f"\n  CAS 2 – Referral probablement APPROPRIÉ (confiance = {y_proba[idx_app, 1]:.1%})")
    print(f"    Caractéristiques :")
    for i, feat in enumerate(noms_features):
        val = X_test.iloc[idx_app, i] if hasattr(X_test, 'iloc') else X_test[idx_app, i]
        if feat in encodeurs and feat != 'target':
            try:
                val_decode = encodeurs[feat].inverse_transform([int(val)])[0]
                print(f"      {feat:25s} : {val_decode}")
            except:
                print(f"      {feat:25s} : {val}")
        else:
            print(f"      {feat:25s} : {val}")

    # Interprétation générale
    print(f"\n  --- INTERPRÉTATION CLINIQUE ---")
    print(f"  Le modèle bayésien calcule P(adéquation | âge, niveau, diagnostic, raison, spécialité)")
    print(f"  grâce au théorème de Bayes :")
    print(f"")
    print(f"    P(Y|X) = P(X|Y) × P(Y) / P(X)")
    print(f"")
    print(f"  Où :")
    print(f"    • P(Y)      = fréquence a priori de chaque classe dans les données SMOTE")
    print(f"    • P(X|Y)    = vraisemblance (loi gaussienne pour chaque feature)")
    print(f"    • P(Y|X)    = probabilité a posteriori du referral étant approprié ou non")
    print(f"")
    print(f"  APPLICATION MÉDICALE :")
    print(f"    Un médecin généraliste peut utiliser ces probabilités comme")
    print(f"    'score de risque' avant de référer un patient à l'ORL :")
    print(f"    • Score > 70% appropriate  → referral justifié, envoyer le patient")
    print(f"    • Score < 30% appropriate  → traitement local probablement suffisant")
    print(f"    • Zone grise 30-70%       → avis téléphonique de l'ORL recommandé")


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_naive_bayes():
    """
    Exécute l'ensemble du pipeline Naive Bayes.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║    NAIVE BAYES – Classification Bayésienne Naïve ORL    ║")
    print("╚" + "═" * 58 + "╝")

    X_train_bal, X_test, y_train_bal, y_test, noms_features, encodeurs = charger_donnees()

    model = entrainer_naive_bayes(X_train_bal, y_train_bal)
    y_pred, y_proba = predire(model, X_test, encodeurs)
    metrics = evaluer_model(y_test, y_pred, y_proba, encodeurs)
    matrice_confusion(y_test, y_pred, encodeurs)
    courbe_roc(y_test, y_proba, encodeurs)
    interpretation_medicale(model, X_test, y_proba, noms_features, encodeurs)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  NAIVE BAYES TERMINÉ – Modèle probabiliste prêt        ║")
    print("╚" + "═" * 58 + "╝\n")

    return model, metrics


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    model, metrics = pipeline_naive_bayes()