# 🏥 Projet Data Mining — Erreurs de Diagnostic des Maladies ORL

> **Master IASD — Module Data Mining 2025/2026**  
> Université Mohammed Premier — Faculté des Sciences d'Oujda  
> Encadrant : Prof. A. Kerkour Elmiad

---

## 📋 Description

Ce projet applique les techniques de **Data Mining** sur un dataset médical réel portant sur les **erreurs de diagnostic des maladies ORL** (Oto-Rhino-Laryngologie).

L'objectif est d'analyser les divergences entre les diagnostics posés par des **médecins non-ORL** dans les établissements référents et ceux confirmés par les **spécialistes ORL** à l'hôpital universitaire (UTH), afin d'extraire des connaissances cachées et d'identifier les facteurs influençant les erreurs de référencement.

**Source des données :**  
Lukama et al. (2022) — *Données sur les erreurs de diagnostic des maladies ORL*  
[DOI : 10.17632/6tgz2db7yn.1](https://data.mendeley.com/datasets/6tgz2db7yn/1)

---

## 📊 Dataset

| Propriété | Valeur |
|-----------|--------|
| Nombre de patients | **1 543** |
| Nombre de variables | **12** |
| Variable cible | `referral appropriateness` |
| Classes | `appropriate` / `inappropriate` / `not applicable` |
| Valeurs manquantes | **0** |
| Déséquilibre classes | Oui → traité par **SMOTE** |

### Variables principales

| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numérique | Âge du patient (0.1 – 87 ans) |
| `UTH referral?` | Catégoriel | Référé via UTH (yes/no) |
| `facility ref level` | Ordinal | Niveau établissement référent (0–3, 9) |
| `refdiag1` | Catégoriel | Diagnostic initial (médecin non-ORL) |
| `entdiag1` | Catégoriel | Diagnostic définitif (spécialiste ORL) |
| `specialtyentdx1` | Catégoriel | Spécialité ORL (rhinologie, otologie…) |
| `attendance` | Catégoriel | Type de visite (1ère visite / suivi) |
| `referral appropriateness` | **Cible** | Statut de la référence |

---

## 🧠 Algorithmes Implémentés

| # | Algorithme | Bibliothèque | Objectif |
|---|------------|-------------|----------|
| 1 | **Decision Tree** (CART) | `sklearn` | Classification + interprétation médicale |
| 2 | **Random Forest** | `sklearn` | Meilleure précision + feature importance |
| 3 | **Naive Bayes** | `sklearn` | Classification probabiliste bayésienne |
| 4 | **CAH** (Hiérarchique) | `scipy` | Clustering patients + dendrogramme |
| 5 | **Règles d'Association** | `mlxtend` | Patterns diagnostics → inappropriatesse |

---

## 📁 Structure du Projet

```
Projet_DataMining_ENT/
│
├── 📄 README.md                    # Ce fichier
├── 📄 requirements.txt             # Dépendances Python
├── 📄 main.py                      # Script principal (exécute tout)
│
├── 📂 data/
│   └── Diagnostic_error_of_ENT_diseases.xlsx
│
├── 📂 src/                         # Scripts Python modulaires
│   ├── preprocessing.py            # Nettoyage, encodage, SMOTE
│   ├── eda.py                      # Analyse exploratoire + visualisations
│   ├── feature_selection.py        # Heatmap corrélation + sélection
│   ├── decision_tree.py            # Algorithme 1 : Arbre de décision
│   ├── random_forest.py            # Algorithme 2 : Forêt aléatoire
│   ├── naive_bayes.py              # Algorithme 3 : Naïve Bayes
│   ├── cah.py                      # Algorithme 4 : Clustering hiérarchique
│   ├── association_rules.py        # Algorithme 5 : Règles d'association
│   └── comparison.py               # Tableau comparatif final
│
├── 📂 notebooks/
│   └── exploration.ipynb           # Notebook d'exploration interactif
│
└── 📂 outputs/
    ├── figures/                    # Graphiques générés (.png)
    └── tables/                     # Résultats CSV + métriques
```

---

## ⚙️ Installation

### Prérequis
- Python 3.9+
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/medamineaka/Projet_DataMining_ENT.git
cd Projet_DataMining_ENT

# 2. Créer un environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer le dataset
# → Copier le fichier Excel dans le dossier data/
```

---

## 🚀 Utilisation

### Exécuter le pipeline complet

```bash
python main.py
```

### Exécuter un script spécifique

```bash
# Preprocessing uniquement
python src/preprocessing.py

# Analyse exploratoire
python src/eda.py

# Arbre de décision
python src/decision_tree.py

# Toute la comparaison finale
python src/comparison.py
```

---

## 📈 Résultats Principaux

### Distribution de la variable cible

| Classe | Avant SMOTE | Après SMOTE |
|--------|-------------|-------------|
| `inappropriate` | 777 (50.4%) | 582 |
| `appropriate` | 597 (38.7%) | 582 |
| `not applicable` | 169 (10.9%) | 582 |

### Découverte clé
> **80.1% des patients** présentent une discordance entre le diagnostic initial et le diagnostic ORL définitif — ce qui confirme l'importance critique de ce problème de santé publique.

---

## 📦 Dépendances

```
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
scipy>=1.12.0
mlxtend>=0.23.0
openpyxl>=3.1.2
xlrd==2.0.1
joblib>=1.3.2
```

---

## 👥 Équipe

| Rôle | Responsabilité |
|------|---------------|
| **Mohammed Amine** | Implémentation Python — preprocessing, algorithmes, visualisations |
| **Binôme** | Workflow Orange — pipeline visuel, comparaison graphique |

---

## 📚 Référence Dataset

```bibtex
@dataset{lukama2022ent,
  author    = {Lukama, Lufunda and Aldous, Colleen and Michelo, Charles and Kalinda, Chester},
  title     = {Diagnostic error of ENT diseases},
  year      = {2022},
  publisher = {Mendeley Data},
  version   = {1},
  doi       = {10.17632/6tgz2db7yn.1}
}
```

---

## 📝 Licence

Projet académique — Master IASD, FSO Université Mohammed Premier, Oujda, Maroc.
