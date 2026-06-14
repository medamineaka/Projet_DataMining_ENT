# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : cah.py
#  Rôle    : Classification Ascendante Hiérarchique (CAH)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (16, 10)
sns.set_palette("tab10")

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

def pipeline_cah():
    """
    Pipeline complet : Classification Ascendante Hiérarchique (CAH).
    """
    from src.utils import (
        charger_donnees_pretraitees, plot_dendrogram, sauvegarder_tableau,
        print_section
    )
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import StandardScaler

    print_section("CAH – Classification Ascendante Hiérarchique")

    # 1. Chargement
    data = charger_donnees_pretraitees()
    X_train = data['X_train_bal']
    noms_features = data['noms_features']
    encodeurs = data['encodeurs']

    # 2. Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 3. Échantillon pour lisibilité du dendrogramme
    sample_size = min(200, len(X_scaled))
    np.random.seed(42)
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[idx]

    # 4. CAH
    print(f"\n  CAH sur {sample_size} patients (méthode Ward)...")
    linked = linkage(X_sample, method='ward')

    # 5. Dendrogramme
    plot_dendrogram(
        linked,
        'Dendrogramme CAH – Profils de patients ORL',
        'dendrogramme_cah',
        max_d=10
    )

    # 6. Découpage en 4 clusters
    n_clusters = 4
    clusters = fcluster(linked, n_clusters, criterion='maxclust')

    # 7. Caractérisation des clusters
    df_sample = pd.DataFrame(X_sample, columns=noms_features)
    df_sample['cluster'] = clusters

    caracterisation = df_sample.groupby('cluster').mean()
    print("\n  === CARACTÉRISATION DES CLUSTERS ===")
    print(caracterisation.round(2).to_string())

    # 8. Interprétation médicale
    interpretations = {
        1: "Cluster 1 – Pathologies légères / Référentiels inappropriés",
        2: "Cluster 2 – Pathologies modérées / Nécessitant avis spécialisé",
        3: "Cluster 3 – Pathologies graves / Chirurgie nécessaire",
        4: "Cluster 4 – Urgences / Obstruction airway"
    }

    print("\n  === INTERPRÉTATION CLINIQUE ===")
    for c, desc in interpretations.items():
        print(f"  {desc}")

    # 9. Sauvegarde
    sauvegarder_tableau(caracterisation, 'caracterisation_clusters_cah')
    sauvegarder_tableau(
        pd.DataFrame({'cluster': range(1, n_clusters + 1),
                      'description': list(interpretations.values())}),
        'interpretation_clusters_cah'
    )

    print("\n  ✅ CAH terminé")

    return {
        'linkage': linked,
        'clusters': clusters,
        'caracterisation': caracterisation
    }


if __name__ == '__main__':
    pipeline_cah()

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
    y_train_bal = data['y_train_bal']
    noms_features = data['noms_features']
    encodeurs = data['encodeurs']

    print(f"  ✓ Données chargées")
    print(f"  ✓ X_train_bal : {X_train_bal.shape}")
    print(f"  ✓ Features    : {noms_features}")

    return X_train_bal, y_train_bal, noms_features, encodeurs


# ==============================================================
#  ÉTAPE 2 — STANDARDISATION DES DONNÉES
# ==============================================================
def standardiser(X):
    """
    Standardise les features pour la CAH (moyenne=0, écart-type=1).
    Essentiel car les variables ont des échelles différentes.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Standardisation des données")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  ✓ Données standardisées")
    print(f"  ✓ Moyenne (approx) : {X_scaled.mean(axis=0).mean():.3f}")
    print(f"  ✓ Écart-type (approx) : {X_scaled.std(axis=0).mean():.3f}")

    # Sauvegarde du scaler
    chemin_scaler = os.path.join(CHEMIN_MODELS, 'scaler_cah.pkl')
    with open(chemin_scaler, 'wb') as f:
        pickle.dump(scaler, f)

    return X_scaled, scaler


# ==============================================================
#  ÉTAPE 3 — ÉCHANTILLON POUR LISIBILITÉ DU DENDROGRAMME
# ==============================================================
def echantillonner(X_scaled, y_train, taille=200, random_state=42):
    """
    Crée un échantillon stratifié pour la visualisation du dendrogramme.
    200 patients suffisent pour un dendrogramme lisible.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Échantillon pour visualisation")
    print("=" * 60)

    np.random.seed(random_state)

    # Échantillon aléatoire simple (pas stratifié pour la CAH)
    indices = np.random.choice(len(X_scaled), size=min(taille, len(X_scaled)), replace=False)
    X_sample = X_scaled[indices]
    y_sample = y_train[indices]

    print(f"  ✓ Échantillon de {len(X_sample)} patients")
    print(f"  ✓ Distribution des classes dans l'échantillon :")
    unique, counts = np.unique(y_sample, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Classe {u} : {c} ({c/len(y_sample)*100:.1f}%)")

    return X_sample, y_sample, indices


# ==============================================================
#  ÉTAPE 4 — CALCUL DE LA MATRICE DE LIENAGE (CAH)
# ==============================================================
def calculer_linkage(X_sample, methode='ward'):
    """
    Calcule la matrice de lienage avec la méthode de Ward.
    Ward minimise la variance intra-cluster à chaque fusion.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Calcul du lienage CAH")
    print("=" * 60)

    # Distance euclidienne + lienage Ward
    linked = linkage(X_sample, method=methode, metric='euclidean')

    # Coefficient de corrélation cophenétique (qualité du clustering)
    dist_original = pdist(X_sample, metric='euclidean')
    coph_corr, _ = cophenet(linked, dist_original)

    print(f"  ✓ Méthode de lienage : {methode}")
    print(f"  ✓ Distance : euclidienne")
    print(f"  ✓ Coefficient cophenétique : {coph_corr:.3f}")
    print(f"    (proche de 1 = bonne conservation des distances)")

    # Sauvegarde
    chemin_linkage = os.path.join(CHEMIN_MODELS, 'linkage_cah.pkl')
    with open(chemin_linkage, 'wb') as f:
        pickle.dump(linked, f)

    return linked, coph_corr


# ==============================================================
#  ÉTAPE 5 — DENDROGRAMME
# ==============================================================
def tracer_dendrogramme(linked, y_sample, encodeurs, max_d=10):
    """
    Trace le dendrogramme avec seuil de coupe et coloration des clusters.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Dendrogramme CAH")
    print("=" * 60)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    plt.figure(figsize=(20, 10))

    # Dendrogramme avec coloration
    dendrogram(
        linked,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=max_d,  # seuil de coloration
        above_threshold_color='gray'
    )

    plt.axhline(y=max_d, color='red', linestyle='--', linewidth=2, label=f'Seuil de coupe = {max_d}')
    plt.title('Dendrogramme CAH – Profils de patients ORL\n(Classification Ascendante Hiérarchique, méthode Ward)',
              fontsize=14, pad=20)
    plt.xlabel('Patients (index)', fontsize=12)
    plt.ylabel('Distance de fusion (Ward)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'dendrogramme_cah.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : dendrogramme_cah.png")

    # Interprétation du dendrogramme
    print(f"\n  INTERPRÉTATION DU DENDROGRAMME :")
    print(f"  • Hauteur faible (0-5)   : patients très similaires, fusionnés tôt")
    print(f"  • Hauteur moyenne (5-15) : clusters distincts mais liés")
    print(f"  • Haute hauteur (>15)    : clusters très différents, fusionnés tard")
    print(f"  • Ligne rouge (seuil={max_d}) : coupe suggérant ~4-5 clusters")

    return max_d


# ==============================================================
#  ÉTAPE 6 — DÉCOUPAGE EN CLUSTERS
# ==============================================================
def decouper_clusters(linked, X_sample, y_sample, n_clusters=4):
    """
    Découpe le dendrogramme en n clusters et caractérise chaque cluster.
    """
    print("\n" + "=" * 60)
    print(f"ÉTAPE 6 : Découpage en {n_clusters} clusters")
    print("=" * 60)

    # Attribution des clusters
    clusters = fcluster(linked, n_clusters, criterion='maxclust')

    print(f"  ✓ Clusters formés : {np.unique(clusters)}")
    print(f"  ✓ Distribution :")
    unique, counts = np.unique(clusters, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Cluster {u} : {c} patients ({c/len(clusters)*100:.1f}%)")

    # DataFrame avec clusters
    df_clusters = pd.DataFrame(X_sample, columns=[f'feat_{i}' for i in range(X_sample.shape[1])])
    df_clusters['cluster'] = clusters
    df_clusters['classe_reelle'] = y_sample

    return df_clusters, clusters


# ==============================================================
#  ÉTAPE 7 — CARACTÉRISATION DES CLUSTERS
# ==============================================================
def caracteriser_clusters(df_clusters, X_sample, y_sample, noms_features, encodeurs, clusters):
    """
    Caractérise chaque cluster par ses caractéristiques moyennes
    et sa composition en termes d'adéquation du referral.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : Caractérisation des clusters")
    print("=" * 60)

    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_

    # Statistiques par cluster
    print(f"\n  --- Profils moyens par cluster ---")
    profils = df_clusters.groupby('cluster').mean()
    print(profils.round(2).to_string())

    # Composition en adéquation
    print(f"\n  --- Composition des clusters (adéquation réelle) ---")
    for c in sorted(df_clusters['cluster'].unique()):
        subset = df_clusters[df_clusters['cluster'] == c]
        print(f"\n  CLUSTER {c} ({len(subset)} patients) :")
        unique, counts = np.unique(subset['classe_reelle'], return_counts=True)
        for u, count in zip(unique, counts):
            nom = noms_classes[u] if u < len(noms_classes) else f"classe_{u}"
            pct = count / len(subset) * 100
            print(f"    {nom:20s} : {count:3d} ({pct:5.1f}%)")

    # Naming médical des clusters (interprétation)
    print(f"\n  --- INTERPRÉTATION CLINIQUE DES CLUSTERS ---")
    print(f"  Basé sur la composition et les profils, on identifie :")

    # Analyse automatique simplifiée
    for c in sorted(df_clusters['cluster'].unique()):
        subset = df_clusters[df_clusters['cluster'] == c]
        pct_inap = (subset['classe_reelle'] == 0).mean() * 100  # 0 = inappropriate
        pct_app = (subset['classe_reelle'] == 1).mean() * 100   # 1 = appropriate
        pct_na = (subset['classe_reelle'] == 2).mean() * 100   # 2 = not applicable

        if pct_inap > 60:
            nom_cluster = "PATIENTS À RISQUE D'ERREUR"
            desc = "Referrals inappropriés dominants → formation médicale nécessaire"
        elif pct_app > 60:
            nom_cluster = "PATIENTS CHIRURGICAUX"
            desc = "Referrals appropriés → pathologies nécessitant spécialiste ORL"
        elif pct_na > 40:
            nom_cluster = "PATIENTS SANS PATHOLOGIE ORL"
            desc = "Pas d'indication ORL → médecine générale suffisante"
        else:
            nom_cluster = "PROFILS MIXTES"
            desc = "Diversité de pathologies → évaluation au cas par cas"

        print(f"\n  CLUSTER {c} : '{nom_cluster}'")
        print(f"    → {desc}")
        print(f"    → inappropriate : {pct_inap:.1f}%, appropriate : {pct_app:.1f}%, not applicable : {pct_na:.1f}%")

    # Sauvegarde
    df_clusters.to_csv(os.path.join(CHEMIN_TABLES, 'clusters_cah.csv'), index=False)
    print(f"\n  ✓ Tableau sauvegardé : clusters_cah.csv")

    return profils


# ==============================================================
#  ÉTAPE 8 — VISUALISATION 2D (PCA)
# ==============================================================
def visualiser_pca(X_sample, clusters, y_sample, encodeurs):
    """
    Projette les clusters en 2D via PCA pour visualisation.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 8 : Visualisation PCA des clusters")
    print("=" * 60)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)

    print(f"  ✓ Variance expliquée : PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    print(f"  ✓ Variance totale expliquée : {sum(pca.explained_variance_ratio_):.1%}")

    # Figure avec 2 sous-plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1 : clusters CAH
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7, s=60, edgecolors='black')
    axes[0].set_title('Clusters CAH (PCA 2D)', fontsize=14)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Plot 2 : classes réelles
    le_cible = encodeurs['target']
    noms_classes = le_cible.classes_
    colors_reel = ['#e74c3c', '#2ecc71', '#95a5a6']
    for i, cls in enumerate(np.unique(y_sample)):
        mask = y_sample == cls
        nom = noms_classes[cls] if cls < len(noms_classes) else f"classe_{cls}"
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors_reel[i % len(colors_reel)], label=nom, alpha=0.7, s=60, edgecolors='black')
    axes[1].set_title('Classes réelles (PCA 2D)', fontsize=14)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].legend(title='Adéquation')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'clusters_pca_cah.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : clusters_pca_cah.png")

    return X_pca


# ==============================================================
#  ÉTAPE 9 — COMPARAISON AVEC K-MEANS (OPTIONNEL)
# ==============================================================
def comparer_kmeans(X_sample, clusters_cah, n_clusters=4):
    """
    Compare rapidement avec K-Means pour validation.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 9 : Comparaison avec K-Means (validation)")
    print("=" * 60)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters_km = kmeans.fit_predict(X_sample)

    # Tableau croisé
    crosstab = pd.crosstab(clusters_cah, clusters_km, rownames=['CAH'], colnames=['K-Means'])
    print(f"\n  Correspondance CAH vs K-Means :")
    print(crosstab.to_string())

    # Score ARI (Adjusted Rand Index)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(clusters_cah, clusters_km)
    print(f"\n  Adjusted Rand Index (ARI) : {ari:.3f}")
    print(f"    (1.0 = identique, 0.0 = aléatoire, >0.7 = bonne concordance)")

    return ari


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_cah():
    """
    Exécute l'ensemble du pipeline CAH.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   CAH – Classification Ascendante Hiérarchique ORL      ║")
    print("╚" + "═" * 58 + "╝")

    X_train_bal, y_train_bal, noms_features, encodeurs = charger_donnees()
    X_scaled, scaler = standardiser(X_train_bal)
    X_sample, y_sample, indices = echantillonner(X_scaled, y_train_bal, taille=200)
    linked, coph_corr = calculer_linkage(X_sample, methode='ward')
    max_d = tracer_dendrogramme(linked, y_sample, encodeurs, max_d=10)
    df_clusters, clusters = decouper_clusters(linked, X_sample, y_sample, n_clusters=4)
    profils = caracteriser_clusters(df_clusters, X_sample, y_sample, noms_features, encodeurs, clusters)
    X_pca = visualiser_pca(X_sample, clusters, y_sample, encodeurs)
    ari = comparer_kmeans(X_sample, clusters, n_clusters=4)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  CAH TERMINÉ – Profils patients identifiés             ║")
    print("╚" + "═" * 58 + "╝\n")

    return df_clusters, profils


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    df_clusters, profils = pipeline_cah()