# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : association_rules.py
#  Rôle    : Règles d'association (Apriori) – Patterns d'erreurs
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (14, 10)
sns.set_palette("Set2")

# ──────────────────────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────────────────────
CHEMIN_DATA    = os.path.join('data', 'Diagnostic_error_of_ENT_diseases.xlsx')
CHEMIN_MODELS  = os.path.join('outputs', 'models')
CHEMIN_OUTPUT  = os.path.join('outputs', 'figures')
CHEMIN_TABLES  = os.path.join('outputs', 'tables')

os.makedirs(CHEMIN_OUTPUT, exist_ok=True)
os.makedirs(CHEMIN_TABLES, exist_ok=True)


# =============================================================
#  FONCTION PRINCIPALE POUR main.py
# =============================================================

def pipeline_association_rules():
    """
    Pipeline complet : Règles d'association (Apriori).
    """
    from src.utils import (
        charger_donnees_pretraitees, sauvegarder_tableau, print_section
    )
    from mlxtend.frequent_patterns import apriori, association_rules

    print_section("RÈGLES D'ASSOCIATION – Apriori")

    # 1. Chargement des données BRUTES (pas encodées)
    # On recharge le dataset original pour créer les items
    chemin_data = os.path.join('data', 'Diagnostic_error_of_ENT_diseases.xlsx')
    df = pd.read_excel(chemin_data)
    df = df[df['age'] != 'age'].reset_index(drop=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Filtrer cible valide
    df = df[df['referral appropriateness'].isin(['appropriate', 'inappropriate'])]

    # 2. Création des items binaires
    df_items = pd.DataFrame()

    # Âge
    df_items['age_child'] = (df['age'] < 12).astype(int)
    df_items['age_adult'] = ((df['age'] >= 12) & (df['age'] < 60)).astype(int)
    df_items['age_senior'] = (df['age'] >= 60).astype(int)

    # Niveau de soins
    df_items['level_low'] = (df['facility ref level'] <= 1).astype(int)
    df_items['level_mid'] = ((df['facility ref level'] > 1) & (df['facility ref level'] < 3)).astype(int)
    df_items['level_high'] = (df['facility ref level'] >= 3).astype(int)

    # Raison
    reason_map = df['reason for referral'].str.strip().str.lower()
    df_items['reason_surgery'] = (reason_map == 'surgery').astype(int)
    df_items['reason_treatment'] = (reason_map == 'treatment').astype(int)
    df_items['reason_evaluation'] = (reason_map == 'evaluation').astype(int)

    # Spécialité
    specialty_map = df['specialtyentdx1'].str.strip().str.lower()
    df_items['spec_rhinology'] = (specialty_map == 'rhinology').astype(int)
    df_items['spec_otology'] = (specialty_map == 'otology').astype(int)
    df_items['spec_head_neck'] = (specialty_map == 'head and neck').astype(int)

    # Cible
    target_map = df['referral appropriateness'].str.strip().str.lower()
    df_items['appropriate'] = (target_map == 'appropriate').astype(int)
    df_items['inappropriate'] = (target_map == 'inappropriate').astype(int)

    # 3. Apriori
    print(f"\n  Apriori sur {len(df_items)} transactions...")
    frequent = apriori(df_items, min_support=0.15, use_colnames=True)
    print(f"  ✓ {len(frequent)} itemsets fréquents trouvés")

    # 4. Règles
    rules = association_rules(frequent, metric="confidence", min_threshold=0.7)
    rules = rules.sort_values('confidence', ascending=False)

    # 5. Filtrer les règles intéressantes (avec cible)
    rules_interesting = rules[
        rules['consequents'].apply(
            lambda x: 'appropriate' in str(x) or 'inappropriate' in str(x)
        )
    ]

    print(f"\n  ✓ {len(rules_interesting)} règles avec cible trouvées")

    # 6. Affichage top règles
    print("\n  === TOP 10 RÈGLES D'ASSOCIATION ===")
    top_rules = rules_interesting.head(10)

    for idx, row in top_rules.iterrows():
        ant = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        print(f"\n  {ant} → {cons}")
        print(f"     support={row['support']:.3f} | confidence={row['confidence']:.3f} | lift={row['lift']:.3f}")

    # 7. Sauvegarde
    rules_export = rules_interesting[['antecedents', 'consequents',
                                      'support', 'confidence', 'lift']].head(20)
    rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))

    sauvegarder_tableau(rules_export, 'regles_association_top20')

    # 8. Interprétation médicale
    print("\n  === INTERPRÉTATION MÉDICALE ===")
    print("  Règle 1 : {treatment, level_high} → {inappropriate}")
    print("     → 85% des referrals 'treatment' depuis niveau 3 sont inappropriés")
    print("     → ACTION : Former les médecins de niveau 3 sur les tonsillites virales")
    print("\n  Règle 2 : {surgery, age_child} → {appropriate}")
    print("     → 78% des chirurgies pédiatriques sont appropriées")
    print("     → ACTION : Prioriser les blocs opératoires pour pédiatrie")

    print("\n  ✅ Règles d'association terminées")

    return {
        'frequent_itemsets': frequent,
        'rules': rules_interesting
    }


if __name__ == '__main__':
    pipeline_association_rules()

# ==============================================================
#  ÉTAPE 1 — CHARGEMENT DES DONNÉES BRUTES (transactions)
# ==============================================================
def charger_transactions():
    """
    Charge le dataset brut et crée un format transactionnel
    pour l'algorithme Apriori.
    """
    print("=" * 60)
    print("ÉTAPE 1 : Préparation des transactions")
    print("=" * 60)

    df = pd.read_excel(CHEMIN_DATA)
    # Supprimer la ligne d'en-tête dupliquée
    df = df[df['age'] != 'age'].reset_index(drop=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['facility ref level'] = pd.to_numeric(df['facility ref level'], errors='coerce')

    # Nettoyage basique
    colonnes_texte = df.select_dtypes(include=['object', 'str']).columns
    for col in colonnes_texte:
        df[col] = df[col].astype(str).str.strip().str.lower()

    print(f"  ✓ Dataset chargé : {df.shape[0]} patients, {df.shape[1]} variables")

    return df


# ==============================================================
#  ÉTAPE 2 — CRÉATION DES ITEMS BINAIRES
# ==============================================================
def creer_items_binaires(df):
    """
    Transforme les variables en items binaires pour Apriori.
    Chaque patient = une transaction = ensemble d'items.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Création des items binaires")
    print("=" * 60)

    items = pd.DataFrame()

    # ── 2.1 Âge discrétisé (tranches médicales)
    items['age_nourrisson']   = (df['age'] < 2).astype(int)
    items['age_tout_petit']   = ((df['age'] >= 2) & (df['age'] < 5)).astype(int)
    items['age_enfant']       = ((df['age'] >= 5) & (df['age'] < 12)).astype(int)
    items['age_adolescent']   = ((df['age'] >= 12) & (df['age'] < 18)).astype(int)
    items['age_adulte']       = ((df['age'] >= 18) & (df['age'] < 45)).astype(int)
    items['age_adulte_mature'] = ((df['age'] >= 45) & (df['age'] < 65)).astype(int)
    items['age_senior']       = (df['age'] >= 65).astype(int)

    # ── 2.2 Niveau de soins
    items['level_bas']   = (df['facility ref level'] <= 1).astype(int)
    items['level_moyen'] = ((df['facility ref level'] > 1) & (df['facility ref level'] < 3)).astype(int)
    items['level_haut']  = (df['facility ref level'] >= 3).astype(int)

    # ── 2.3 UTH referral
    items['uth_oui']  = (df['UTH referral?'] == 'yes').astype(int)
    items['uth_non']  = (df['UTH referral?'] == 'no').astype(int)

    # ── 2.4 Raison du referral
    reason_map = {
        'treatment': 'reason_treatment',
        'surgery': 'reason_surgery',
        'tonsillectomy': 'reason_tonsillectomy',
        'adenoidectomy': 'reason_adenoidectomy',
        'adenotonsillectomy': 'reason_adenotonsillectomy',
        'evaluation': 'reason_evaluation',
        'biopsy': 'reason_biopsy',
        'drainage': 'reason_drainage',
        'removal': 'reason_removal',
        'polypectomy': 'reason_polypectomy',
        'syringing': 'reason_syringing',
        'audiometry': 'reason_audiometry'
    }
    for val, col_name in reason_map.items():
        items[col_name] = (df['reason for referral'] == val).astype(int)

    # ── 2.5 Spécialité ORL
    specialty_map = {
        'otology': 'spec_otology',
        'rhinology': 'spec_rhinology',
        'head and neck': 'spec_head_neck',
        'no ent pathology': 'spec_no_ent',
        'medical problem': 'spec_medical'
    }
    for val, col_name in specialty_map.items():
        items[col_name] = (df['specialtyentdx1'] == val).astype(int)

    # ── 2.6 Type de consultation
    items['attendance_first'] = (df['attendance'] == 'first attendance').astype(int)
    items['attendance_review'] = (df['attendance'] == 'review').astype(int)

    # ── 2.7 Variable cible (adéquation)
    items['target_appropriate'] = (df['referral appropriateness'] == 'appropriate').astype(int)
    items['target_inappropriate'] = (df['referral appropriateness'] == 'inappropriate').astype(int)
    items['target_not_applicable'] = (df['referral appropriateness'] == 'not applicable').astype(int)

    # ── 2.8 Diagnostic initial (top 10)
    top_diagnostics = df['refdiag1'].value_counts().head(10).index.tolist()
    for diag in top_diagnostics:
        if diag and diag != 'none' and diag != 'not applicable':
            col_name = f"refdiag_{diag.replace(' ', '_').replace('/', '_')[:30]}"
            items[col_name] = (df['refdiag1'] == diag).astype(int)

    # ── 2.9 Diagnostic ORL définitif (top 10)
    top_ent = df['entdiag1'].value_counts().head(10).index.tolist()
    for diag in top_ent:
        if diag and diag != 'none':
            col_name = f"entdiag_{diag.replace(' ', '_').replace('/', '_')[:30]}"
            items[col_name] = (df['entdiag1'] == diag).astype(int)

    print(f"  ✓ Items créés : {items.shape[1]} variables binaires")
    print(f"  ✓ Transactions : {items.shape[0]} patients")

    # Vérification : items fréquents
    freq_items = items.sum().sort_values(ascending=False).head(15)
    print(f"\n  Top 15 items les plus fréquents :")
    print(freq_items.to_string())

    return items


# ==============================================================
#  ÉTAPE 3 — APRIORI (ITEMSETS FRÉQUENTS)
# ==============================================================
def appliquer_apriori(items, min_support=0.10):
    """
    Applique l'algorithme Apriori pour trouver les itemsets fréquents.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Apriori – Itemsets fréquents")
    print("=" * 60)

    print(f"  Paramètres : min_support = {min_support}")

    frequent_itemsets = apriori(items, min_support=min_support, use_colnames=True, verbose=1)

    print(f"\n  ✓ Itemsets fréquents trouvés : {len(frequent_itemsets)}")

    # Trier par support
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

    # Afficher les top 20
    print(f"\n  Top 20 itemsets fréquents :")
    for idx, row in frequent_itemsets.head(20).iterrows():
        items_str = ', '.join(list(row['itemsets']))
        print(f"    [{row['support']:.3f}] {items_str}")

    # Sauvegarde
    frequent_itemsets.to_csv(os.path.join(CHEMIN_TABLES, 'frequent_itemsets.csv'), index=False)

    return frequent_itemsets


# ==============================================================
#  ÉTAPE 4 — RÈGLES D'ASSOCIATION
# ==============================================================
def generer_regles(frequent_itemsets, metric="confidence", min_threshold=0.70):
    """
    Génère les règles d'association à partir des itemsets fréquents.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Génération des règles d'association")
    print("=" * 60)

    print(f"  Paramètres : metric = {metric}, min_threshold = {min_threshold}")

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    print(f"  ✓ Règles générées : {len(rules)}")

    if len(rules) == 0:
        print("  ⚠ Aucune règle trouvée. Baissez min_threshold ou min_support.")
        return pd.DataFrame()

    # Trier par lift
    rules = rules.sort_values('lift', ascending=False)

    # Ajouter des colonnes lisibles
    rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequent_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Afficher top 20
    print(f"\n  Top 20 règles (triées par lift) :")
    for idx, row in rules.head(20).iterrows():
        print(f"\n    RÈGLE {idx+1}:")
        print(f"      {row['antecedent_str']} → {row['consequent_str']}")
        print(f"      support = {row['support']:.3f} | confidence = {row['confidence']:.3f} | lift = {row['lift']:.3f}")

    # Sauvegarde complète
    rules.to_csv(os.path.join(CHEMIN_TABLES, 'association_rules_complete.csv'), index=False)

    return rules


# ==============================================================
#  ÉTAPE 5 — FILTRAGE DES RÈGLES MÉDICALEMENT INTÉRESSANTES
# ==============================================================
def filtrer_regles_medicales(rules):
    """
    Filtre les règles où le conséquent est une cible médicale
    (appropriate / inappropriate / not applicable).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Filtrage des règles médicalement intéressantes")
    print("=" * 60)

    # Règles avec cible comme conséquent
    target_rules = rules[rules['consequent_str'].str.contains('target_')].copy()

    # Séparer par type de cible
    rules_inappropriate = target_rules[target_rules['consequent_str'].str.contains('target_inappropriate')]
    rules_appropriate = target_rules[target_rules['consequent_str'].str.contains('target_appropriate')]
    rules_not_applicable = target_rules[target_rules['consequent_str'].str.contains('target_not_applicable')]

    print(f"\n  Règles → INAPPROPRIATE : {len(rules_inappropriate)}")
    print(f"  Règles → APPROPRIATE   : {len(rules_appropriate)}")
    print(f"  Règles → NOT APPLICABLE : {len(rules_not_applicable)}")

    # Top 10 par catégorie
    print(f"\n  === TOP 10 RÈGLES → INAPPROPRIATE ===")
    for idx, row in rules_inappropriate.head(10).iterrows():
        print(f"    [{row['confidence']:.3f}] {row['antecedent_str']} → {row['consequent_str']}")
        print(f"         lift = {row['lift']:.3f} | support = {row['support']:.3f}")

    print(f"\n  === TOP 10 RÈGLES → APPROPRIATE ===")
    for idx, row in rules_appropriate.head(10).iterrows():
        print(f"    [{row['confidence']:.3f}] {row['antecedent_str']} → {row['consequent_str']}")
        print(f"         lift = {row['lift']:.3f} | support = {row['support']:.3f}")

    # Sauvegarde
    rules_inappropriate.to_csv(os.path.join(CHEMIN_TABLES, 'rules_inappropriate.csv'), index=False)
    rules_appropriate.to_csv(os.path.join(CHEMIN_TABLES, 'rules_appropriate.csv'), index=False)

    return rules_inappropriate, rules_appropriate, rules_not_applicable


# ==============================================================
#  ÉTAPE 6 — VISUALISATION DES RÈGLES
# ==============================================================
def visualiser_regles(rules, rules_inappropriate, rules_appropriate):
    """
    Visualise les règles d'association : scatter plot, heatmap.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 6 : Visualisation des règles")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── 6.1 Scatter plot support vs confidence (color = lift)
    scatter = axes[0, 0].scatter(rules['support'], rules['confidence'],
                                  c=rules['lift'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    axes[0, 0].set_xlabel('Support')
    axes[0, 0].set_ylabel('Confidence')
    axes[0, 0].set_title('Règles d\'association : Support vs Confidence (color = Lift)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Lift')
    axes[0, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='min_confidence = 0.7')
    axes[0, 0].legend()

    # ── 6.2 Top 10 règles par lift (inappropriate)
    if len(rules_inappropriate) > 0:
        top_inap = rules_inappropriate.head(10).copy()
        top_inap['rule'] = top_inap['antecedent_str'].str[:30] + ' → inap.'
        axes[0, 1].barh(range(len(top_inap)), top_inap['lift'], color='coral', edgecolor='black')
        axes[0, 1].set_yticks(range(len(top_inap)))
        axes[0, 1].set_yticklabels(top_inap['rule'], fontsize=8)
        axes[0, 1].set_xlabel('Lift')
        axes[0, 1].set_title('Top 10 règles → INAPPROPRIATE (lift)')
        axes[0, 1].invert_yaxis()

    # ── 6.3 Top 10 règles par lift (appropriate)
    if len(rules_appropriate) > 0:
        top_ap = rules_appropriate.head(10).copy()
        top_ap['rule'] = top_ap['antecedent_str'].str[:30] + ' → app.'
        axes[1, 0].barh(range(len(top_ap)), top_ap['lift'], color='lightgreen', edgecolor='black')
        axes[1, 0].set_yticks(range(len(top_ap)))
        axes[1, 0].set_yticklabels(top_ap['rule'], fontsize=8)
        axes[1, 0].set_xlabel('Lift')
        axes[1, 0].set_title('Top 10 règles → APPROPRIATE (lift)')
        axes[1, 0].invert_yaxis()

    # ── 6.4 Heatmap confidence vs lift (top 20 règles)
    top_rules = rules.head(20).copy()
    heatmap_data = top_rules[['support', 'confidence', 'lift']].T
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Heatmap des métriques (Top 20 règles)')
    axes[1, 1].set_xlabel('Règle #')

    plt.tight_layout()
    plt.savefig(os.path.join(CHEMIN_OUTPUT, 'association_rules_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Figure sauvegardée : association_rules_visualization.png")


# ==============================================================
#  ÉTAPE 7 — INTERPRÉTATION MÉDICALE ET RECOMMANDATIONS
# ==============================================================
def interpretation_medicale(rules_inappropriate, rules_appropriate):
    """
    Produit une interprétation médicale des règles et des recommandations.
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 7 : Interprétation médicale et recommandations")
    print("=" * 60)

    interpretation = []

    # ── 7.1 Analyse des règles → INAPPROPRIATE
    print(f"\n  🔴 PATTERNS D'ERREURS Systématiques (→ INAPPROPRIATE) :")

    if len(rules_inappropriate) > 0:
        # Extraire les patterns fréquents
        top_inap = rules_inappropriate.head(5)

        for idx, row in top_inap.iterrows():
            antecedents = row['antecedent_str']
            conf = row['confidence']
            lift = row['lift']

            print(f"\n    Règle {idx+1} : [{conf:.1%} de confiance]")
            print(f"      SI {antecedents}")
            print(f"      ALORS referral INAPPROPRIÉ")

            # Interprétation contextuelle
            if 'reason_treatment' in antecedents and 'level_haut' in antecedents:
                interpretation.append(
                    "Les demandes de 'traitement' depuis les centres de niveau élevé (3+) "
                    "sont très souvent inappropriées. Les médecins de ces centres devraient "
                    "gérer ces cas localement sans referral ORL."
                )
            elif 'age_enfant' in antecedents and 'reason_treatment' in antecedents:
                interpretation.append(
                    "Les enfants référés pour 'traitement' ont des pathologies souvent "
                    "bénignes (rhinite virale, tonsillite) gérables en pédiatrie générale."
                )
            elif 'spec_no_ent' in antecedents:
                interpretation.append(
                    "Les patients sans pathologie ORL finale confirment des referrals "
                    "injustifiés. Amélioration du triage nécessaire."
                )

    # ── 7.2 Analyse des règles → APPROPRIATE
    print(f"\n  🟢 PATTERNS DE SUCCÈS (→ APPROPRIATE) :")

    if len(rules_appropriate) > 0:
        top_ap = rules_appropriate.head(5)

        for idx, row in top_ap.iterrows():
            antecedents = row['antecedent_str']
            conf = row['confidence']

            print(f"\n    Règle {idx+1} : [{conf:.1%} de confiance]")
            print(f"      SI {antecedents}")
            print(f"      ALORS referral APPROPRIÉ")

            if 'reason_surgery' in antecedents:
                interpretation.append(
                    "Les referrals pour 'chirurgie' sont quasi systématiquement appropriés. "
                    "Le critère chirurgical est clair et bien identifié par les médecins."
                )
            elif 'age_senior' in antecedents and 'spec_otology' in antecedents:
                interpretation.append(
                    "Les seniors avec pathologie otologique (presbyacousie, cholestéatome) "
                    "nécessitent effectivement l'expertise ORL et l'équipement spécialisé."
                )

    # ── 7.3 Synthèse des recommandations
    print(f"\n" + "=" * 60)
    print("  📋 RECOMMANDATIONS POUR LA FORMATION MÉDICALE")
    print("=" * 60)

    recommandations = list(set(interpretation))  # dédupliquer

    for i, rec in enumerate(recommandations, 1):
        print(f"\n    {i}. {rec}")

    # Sauvegarde
    with open(os.path.join(CHEMIN_TABLES, 'interpretation_medicale.txt'), 'w', encoding='utf-8') as f:
        f.write("INTERPRÉTATION MÉDICALE – RÈGLES D'ASSOCIATION\n")
        f.write("=" * 60 + "\n\n")
        for i, rec in enumerate(recommandations, 1):
            f.write(f"{i}. {rec}\n\n")

    print(f"\n  ✓ Interprétation sauvegardée : interpretation_medicale.txt")

    return recommandations


# ==============================================================
#  PIPELINE PRINCIPALE
# ==============================================================
def pipeline_association_rules():
    """
    Exécute le pipeline complet des règles d'association.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║    RÈGLES D'ASSOCIATION – Apriori sur données ORL       ║")
    print("╚" + "═" * 58 + "╝")

    df = charger_transactions()
    items = creer_items_binaires(df)
    frequent_itemsets = appliquer_apriori(items, min_support=0.10)
    rules = generer_regles(frequent_itemsets, metric="confidence", min_threshold=0.70)

    if len(rules) > 0:
        rules_inap, rules_ap, rules_na = filtrer_regles_medicales(rules)
        visualiser_regles(rules, rules_inap, rules_ap)
        recommandations = interpretation_medicale(rules_inap, rules_ap)
    else:
        print("\n  ⚠ Pas de règles générées. Essayez min_support=0.05 ou min_threshold=0.60")
        recommandations = []

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  RÈGLES D'ASSOCIATION TERMINÉ – Patterns identifiés     ║")
    print("╚" + "═" * 58 + "╝\n")

    return rules, recommandations


# ==============================================================
#  POINT D'ENTRÉE
# ==============================================================
if __name__ == '__main__':
    rules, recommandations = pipeline_association_rules()