# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : eda.py
#  Rôle    : Analyse exploratoire des données (EDA)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

# ──────────────────────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────────────────────
CHEMIN_DATA    = os.path.join('data', 'Diagnostic_error_of_ENT_diseases.xlsx')
CHEMIN_OUTPUT  = os.path.join('outputs', 'figures')
CHEMIN_TABLES  = os.path.join('outputs', 'tables')

# Créer les dossiers si inexistants
os.makedirs(CHEMIN_OUTPUT, exist_ok=True)
os.makedirs(CHEMIN_TABLES, exist_ok=True)


# ==============================================================
#  ÉTAPE 1 — CHARGEMENT DES DONNÉES BRUTES (avant preprocessing)
# ==============================================================
def charger_donnees_brutes(chemin=CHEMIN_DATA):
    """
    Charge le dataset brut pour l'analyse exploratoire.
    """
    print("=" * 60)
    print("ÉTAPE 1 : Chargement des données brutes")
    print("=" * 60)

    df = pd.read_excel(chemin)
    # Supprimer la ligne d'en-tête dupliquée
    df = df[df['age'] != 'age'].reset_index(drop=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['facility ref level'] = pd.to_numeric(df['facility ref level'], errors='coerce')

    print(f"  ✓ Dataset chargé : {df.shape[0]} patients, {df.shape[1]} variables")
    return df


# ==============================================================
#  ÉTAPE 2 — STATISTIQUES DESCRIPTIVES UNIVARIÉES
# ==============================================================
def stats_univariees(df):
    """
    Analyse univariée : effectifs, centralité, dispersion.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Statistiques descriptives univariées")
    print("=" * 60)

    # ── 2.1 Variable numérique : âge
    print("\n--- Variable numérique : âge ---")
    print(df['age'].describe().round(2).to_string())

    # ── 2.2 Variables catégorielles principales
    vars_cat = ['referral appropriateness', 'specialtyentdx1',
                'reason for referral', 'attendance', 'UTH referral?']

    for col in vars_cat:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(df[col].value_counts().head(10).to_string())

    # Sauvegarde des stats
    stats_age = df['age'].describe().round(2)
    stats_age.to_csv(os.path.join(CHEMIN_TABLES, 'stats_age.csv'))

    return df['age'].describe()


# ==============================================================
#  ÉTAPE 3 — VISUALISATIONS UNIVARIÉES
# ==============================================================
def visualisations_univariees(df):
    """
    Graphiques univariés : distributions, barplots.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Visualisations univariées")
    print("=" * 60)

    # ── 3.1 Distribution de l'âge
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogramme âge
    axes[0, 0].hist(df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution de l\'âge des patients')
    axes[0, 0].set_xlabel('Âge (années)')
    axes[0, 0].set_ylabel('Effectif')
    axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', label=f'Moyenne = {df["age"].mean():.1f}')
    axes[0, 0].axvline(df['age'].median(), color='green', linestyle='--', label=f'Médiane = {df["age"].median():.1f}')
    axes[0, 0].legend()

    # Boxplot âge
    axes[0, 1].boxplot(df['age'].dropna(), vert=True)
    axes[0, 1].set_title('Boxplot de l\'âge')
    axes[0, 1].set_ylabel('Âge (années)')

    # Barplot : adéquation du referral
    adequation = df['referral appropriateness'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    axes[1, 0].bar(adequation.index, adequation.values, color=colors, edgecolor='black')
    axes[1, 0].set_title('Répartition de l\'adéquation du referral')
    axes[1, 0].set_ylabel('Nombre de patients')
    for i, v in enumerate(adequation.values):
        axes[1, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')

    # Barplot : spécialité ORL
    specialty = df['specialtyentdx1'].value_counts().head(8)
    axes[1, 1].barh(specialty.index, specialty.values, color='coral', edgecolor='black')
    axes[1, 1].set_title('Top 8 spécialités ORL diagnostiquées')
    axes[1, 1].set_xlabel('Nombre de patients')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'eda_univarie.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : eda_univarie.png")


# ==============================================================
#  ÉTAPE 4 — ANALYSE BIVARIÉE
# ==============================================================
def analyse_bivariee(df):
    """
    Analyse bivariée : relations entre variables.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Analyse bivariée")
    print("=" * 60)

    # ── 4.1 Âge vs adéquation du referral
    print("\n--- Âge moyen par adéquation ---")
    age_par_adequation = df.groupby('referral appropriateness')['age'].agg(['mean', 'std', 'count'])
    print(age_par_adequation.round(2).to_string())

    # ── 4.2 Tableau croisé : niveau de soins vs adéquation
    print("\n--- Niveau de soins vs adéquation ---")
    crosstab = pd.crosstab(df['facility ref level'], df['referral appropriateness'],
                           margins=True, normalize='index') * 100
    print(crosstab.round(1).to_string())

    # ── 4.3 Spécialité vs adéquation
    print("\n--- Spécialité vs adéquation ---")
    specialty_adequation = pd.crosstab(df['specialtyentdx1'], df['referral appropriateness'],
                                       normalize='index') * 100
    print(specialty_adequation.round(1).to_string())

    # Sauvegarde
    age_par_adequation.to_csv(os.path.join(CHEMIN_TABLES, 'age_par_adequation.csv'))
    crosstab.to_csv(os.path.join(CHEMIN_TABLES, 'niveau_vs_adequation.csv'))

    return age_par_adequation, crosstab


# ==============================================================
#  ÉTAPE 5 — VISUALISATIONS BIVARIÉES
# ==============================================================
def visualisations_bivariees(df):
    """
    Graphiques bivariés : relations entre variables clés.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Visualisations bivariées")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── 5.1 Boxplot âge par adéquation
    df.boxplot(column='age', by='referral appropriateness', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution de l\'âge selon l\'adéquation du referral')
    axes[0, 0].set_xlabel('Adéquation du referral')
    axes[0, 0].set_ylabel('Âge (années)')

    # ── 5.2 Heatmap : niveau de soins vs adéquation
    crosstab_counts = pd.crosstab(df['facility ref level'], df['referral appropriateness'])
    sns.heatmap(crosstab_counts, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 1])
    axes[0, 1].set_title('Niveau de soins vs Adéquation (effectifs)')
    axes[0, 1].set_xlabel('Adéquation')
    axes[0, 1].set_ylabel('Niveau de soins')

    # ── 5.3 Stacked bar : spécialité vs adéquation
    specialty_counts = pd.crosstab(df['specialtyentdx1'], df['referral appropriateness'])
    specialty_counts.plot(kind='barh', stacked=True, ax=axes[1, 0],
                         color=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[1, 0].set_title('Spécialité ORL vs Adéquation')
    axes[1, 0].set_xlabel('Nombre de patients')
    axes[1, 0].legend(title='Adéquation', bbox_to_anchor=(1.05, 1), loc='upper left')

    # ── 5.4 Raison du referral vs adéquation
    reason_counts = pd.crosstab(df['reason for referral'], df['referral appropriateness'])
    reason_counts.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[1, 1].set_title('Raison du referral vs Adéquation')
    axes[1, 1].set_ylabel('Nombre de patients')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Adéquation')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'eda_bivarie.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : eda_bivarie.png")


# ==============================================================
#  ÉTAPE 6 — HEATMAP DES CORRÉLATIONS (données encodées)
# ==============================================================
def heatmap_correlations(df):
    """
    Heatmap des corrélations sur variables encodées.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Heatmap des corrélations")
    print("=" * 60)

    # Encodage temporaire pour la corrélation
    df_encoded = df.copy()
    colonnes_cat = df_encoded.select_dtypes(include=['object']).columns

    le_temp = {}
    for col in colonnes_cat:
        le = pd.factorize(df_encoded[col])[0]
        df_encoded[col] = le

    # Matrice de corrélation
    corr = df_encoded.corr()

    # Figure
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('Matrice de corrélation des variables (encodées)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'heatmap_correlations.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : heatmap_correlations.png")

    # Sauvegarde des corrélations avec la cible
    cible_corr = corr['referral appropriateness'].abs().sort_values(ascending=False)
    cible_corr.to_csv(os.path.join(CHEMIN_TABLES, 'correlations_avec_cible.csv'))
    print("\n  Top corrélations avec la cible :")
    print(cible_corr.head(10).to_string())


# ==============================================================
#  ÉTAPE 7 — TOP DIAGNOSTICS (initial vs définitif)
# ==============================================================
def analyse_diagnostics(df):
    """
    Analyse des diagnostics : concordance et discordance.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : Analyse des diagnostics")
    print("=" * 60)

    # Top 10 diagnostics initiaux
    print("\n--- Top 10 diagnostics initiaux (non-ORL) ---")
    top_initial = df['refdiag1'].value_counts().head(10)
    print(top_initial.to_string())

    # Top 10 diagnostics définitifs (ORL)
    print("\n--- Top 10 diagnostics définitifs (ORL) ---")
    top_definitif = df['entdiag1'].value_counts().head(10)
    print(top_definitif.to_string())

    # Taux de concordance (approximatif)
    df['concordance'] = (df['refdiag1'].str.strip().str.lower() ==
                         df['entdiag1'].str.strip().str.lower())
    taux_concordance = df['concordance'].mean() * 100
    print(f"\n--- Taux de concordance approximatif : {taux_concordance:.1f}% ---")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_initial.plot(kind='barh', ax=axes[0], color='lightblue', edgecolor='black')
    axes[0].set_title('Top 10 diagnostics initiaux (non-ORL)')
    axes[0].set_xlabel('Nombre de patients')

    top_definitif.plot(kind='barh', ax=axes[1], color='lightcoral', edgecolor='black')
    axes[1].set_title('Top 10 diagnostics définitifs (ORL)')
    axes[1].set_xlabel('Nombre de patients')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'top_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : top_diagnostics.png")

    return taux_concordance


# ==============================================================
#  PIPELINE PRINCIPALE EDA
# ==============================================================
def pipeline_eda():
    """
    Exécute toute l'analyse exploratoire.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║         EDA – Analyse Exploratoire des Données ORL      ║")
    print("╚" + "═" * 58 + "╝")

    df = charger_donnees_brutes()
    stats_univariees(df)
    visualisations_univariees(df)
    analyse_bivariee(df)
    visualisations_bivariees(df)
    heatmap_correlations(df)
    analyse_diagnostics(df)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  EDA TERMINÉ – Toutes les figures et tables sont prêtes  ║")
    print("╚" + "═" * 58 + "╝\n")

    return df


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    df_eda = pipeline_eda()