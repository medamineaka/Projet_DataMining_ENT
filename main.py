# =============================================================
#  PROJET DATA MINING – Module IASD 2025/2026
#  Sujet : Erreurs de diagnostic des maladies ORL
#  Script  : main.py
#  Rôle    : Orchestrateur principal – exécute tous les scripts
# =============================================================
import mtplotlib
matplotlib.use('Agg')
import sys
import os
import argparse
import time

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ──────────────────────────────────────────────────────────────
#  IMPORTS DES SCRIPTS DU PROJET
# ──────────────────────────────────────────────────────────────
try:
    import preprocessing as prep
    import eda
    import feature_selection as fs
    import decision_tree as dt
    import random_forest as rf
    import naive_bayes as nb
    import cah
    import association_rules as ar
    import comparison as comp

    MODULES_OK = True
except ImportError as e:
    print(f"⚠ Erreur d'import : {e}")
    print("Certains modules peuvent être manquants.")
    MODULES_OK = False

# ==============================================================
#  CONFIGURATION
# ==============================================================

ETAPES = {
    '1': {
        'nom': 'Preprocessing',
        'description': 'Chargement, nettoyage, encodage, SMOTE',
        'module': 'preprocessing',
        'fonction': 'pipeline_preprocessing',
        'obligatoire': True
    },
    '2': {
        'nom': 'EDA',
        'description': 'Analyse exploratoire des données',
        'module': 'eda',
        'fonction': 'pipeline_eda',
        'obligatoire': True
    },
    '3': {
        'nom': 'Feature Selection',
        'description': 'Sélection de variables',
        'module': 'feature_selection',
        'fonction': 'pipeline_feature_selection',
        'obligatoire': False
    },
    '4': {
        'nom': 'Decision Tree',
        'description': 'Arbre de décision CART',
        'module': 'decision_tree',
        'fonction': 'pipeline_decision_tree',
        'obligatoire': True
    },
    '5': {
        'nom': 'Random Forest',
        'description': 'Forêt aléatoire',
        'module': 'random_forest',
        'fonction': 'pipeline_random_forest',
        'obligatoire': True
    },
    '6': {
        'nom': 'Naive Bayes',
        'description': 'Classification bayésienne naïve',
        'module': 'naive_bayes',
        'fonction': 'pipeline_naive_bayes',
        'obligatoire': True
    },
    '7': {
        'nom': 'CAH',
        'description': 'Classification ascendante hiérarchique',
        'module': 'cah',
        'fonction': 'pipeline_cah',
        'obligatoire': True
    },
    '8': {
        'nom': 'Règles d\'association',
        'description': 'Apriori – patterns d\'erreurs',
        'module': 'association_rules',
        'fonction': 'pipeline_association_rules',
        'obligatoire': True
    },
    '9': {
        'nom': 'Comparaison finale',
        'description': 'Tableau comparatif de tous les modèles',
        'module': 'comparison',
        'fonction': 'pipeline_comparison',
        'obligatoire': True
    }
}


# ==============================================================
#  FONCTIONS D'AFFICHAGE
# ==============================================================

def afficher_banniere():
    """Affiche la bannière du projet"""
    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " PROJET DATA MINING – Erreurs de diagnostic ORL ".center(70) + "║")
    print("║" + " Master IASD 2025/2026 ".center(70) + "║")
    print("╠" + "═" * 70 + "╣")
    print("║" + " Python (vous) + Orange Workflow (votre amie) ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")


def afficher_menu():
    """Affiche le menu des étapes disponibles"""
    print("\n" + "─" * 70)
    print("  ÉTAPES DISPONIBLES")
    print("─" * 70)

    for num, etape in ETAPES.items():
        statut = "⭐" if etape['obligatoire'] else "  "
        print(f"  [{num}] {statut} {etape['nom']:25s} – {etape['description']}")

    print("\n  [all]  Exécuter toutes les étapes obligatoires (1,2,4,5,6,7,8,9)")
    print("  [full] Exécuter TOUTES les étapes (1-9)")
    print("  [0]    Quitter")
    print("─" * 70)


def afficher_recap(resultats):
    """Affiche le récapitulatif des exécutions"""
    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " RÉCAPITULATIF D'EXÉCUTION ".center(70) + "║")
    print("╠" + "═" * 70 + "╣")

    total = len(resultats)
    reussis = sum(1 for r in resultats.values() if r['succes'])
    echecs = total - reussis

    for num, res in resultats.items():
        etape = ETAPES[num]['nom']
        statut = "✅ OK" if res['succes'] else "❌ ÉCHEC"
        temps = f"{res['temps']:.2f}s" if res['temps'] else "N/A"
        print(f"  {statut}  Étape {num} – {etape:25s} ({temps})")
        if not res['succes'] and res['erreur']:
            print(f"      → {res['erreur'][:60]}")

    print("╠" + "═" * 70 + "╣")
    print(f"  Total : {reussis}/{total} étapes réussies")
    print("╚" + "═" * 70 + "╝")


# ==============================================================
#  EXÉCUTION DES ÉTAPES
# ==============================================================

def executer_etape(num_etape):
    """
    Exécute une étape spécifique par son numéro
    """
    if num_etape not in ETAPES:
        print(f"❌ Étape {num_etape} inconnue")
        return False

    etape = ETAPES[num_etape]
    print(f"\n{'=' * 70}")
    print(f"  ÉTAPE {num_etape} : {etape['nom']}")
    print(f"  {etape['description']}")
    print(f"{'=' * 70}")

    # Vérifier que le module existe
    # Vérifier que le module existe
    try:
        module = globals()[etape['module']]
    except KeyError:
        try:
            module = __import__(etape['module'])
        except ImportError as e:
            print(f"❌ Module {etape['module']} non trouvé : {e}")
            return False, 0, str(e)

    # Vérifier que la fonction existe
    fonction_nom = etape['fonction']
    if not hasattr(module, fonction_nom):
        print(f"❌ Fonction {fonction_nom} non trouvée dans {etape['module']}")
        return False, 0, f"Fonction {fonction_nom} introuvable"

    # Exécution
    fonction = getattr(module, fonction_nom)
    debut = time.time()

    try:
        resultat = fonction()
        temps = time.time() - debut
        print(f"\n  ✅ Étape {num_etape} terminée en {temps:.2f}s")
        return True, temps, None
    except Exception as e:
        temps = time.time() - debut
        print(f"\n  ❌ Étape {num_etape} échouée : {str(e)}")
        return False, temps, str(e)


def executer_plusieurs_etapes(liste_etapes):
    """
    Exécute une liste d'étapes et retourne les résultats
    """
    resultats = {}

    for num in liste_etapes:
        succes, temps, erreur = executer_etape(num)
        resultats[num] = {
            'succes': succes,
            'temps': temps,
            'erreur': erreur
        }

        # Pause entre étapes
        if num != liste_etapes[-1]:
            print(f"\n  ⏳ Pause... (appuyez sur Entrée pour continuer)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n  ⚠ Exécution interrompue par l'utilisateur")
                break

    return resultats


def executer_toutes_etapes(obligatoires_only=False):
    """
    Exécute toutes les étapes (ou seulement les obligatoires)
    """
    if obligatoires_only:
        etapes = [k for k, v in ETAPES.items() if v['obligatoire']]
    else:
        etapes = list(ETAPES.keys())

    print(f"\n  📋 Exécution de {len(etapes)} étape(s)")
    if obligatoires_only:
        print("  (mode : étapes obligatoires uniquement)")

    return executer_plusieurs_etapes(etapes)


# ==============================================================
#  INTERFACE EN LIGNE DE COMMANDE
# ==============================================================

def parse_arguments():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Orchestrateur du projet Data Mining ORL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py              → Menu interactif
  python main.py --all        → Exécute toutes les étapes obligatoires
  python main.py --full       → Exécute TOUTES les étapes
  python main.py -e 1 4 5     → Exécute les étapes 1, 4 et 5
  python main.py --step 4     → Exécute uniquement l'étape 4
        """
    )

    parser.add_argument(
        '-a', '--all',
        action='store_true',
        help='Exécuter toutes les étapes obligatoires'
    )

    parser.add_argument(
        '-f', '--full',
        action='store_true',
        help='Exécuter TOUTES les étapes (1-9)'
    )

    parser.add_argument(
        '-e', '--etapes',
        nargs='+',
        help='Liste d\'étapes à exécuter (ex: 1 4 5)'
    )

    parser.add_argument(
        '-s', '--step',
        type=str,
        help='Exécuter une seule étape'
    )

    parser.add_argument(
        '--no-pause',
        action='store_true',
        help='Ne pas faire de pause entre les étapes'
    )

    return parser.parse_args()


def mode_interactif():
    """Mode interactif avec menu"""
    while True:
        afficher_menu()

        choix = input("\n  Votre choix : ").strip().lower()

        if choix == '0' or choix == 'q':
            print("\n  👋 Au revoir !")
            break

        elif choix == 'all':
            resultats = executer_toutes_etapes(obligatoires_only=True)
            afficher_recap(resultats)

        elif choix == 'full':
            resultats = executer_toutes_etapes(obligatoires_only=False)
            afficher_recap(resultats)

        elif choix in ETAPES:
            succes, temps, erreur = executer_etape(choix)
            if succes:
                print(f"\n  ✅ Terminé en {temps:.2f}s")
            else:
                print(f"\n  ❌ Échec : {erreur}")

        else:
            print(f"\n  ⚠ Choix invalide : '{choix}'")


# ==============================================================
#  POINT D'ENTRÉE PRINCIPAL
# ==============================================================

def main():
    """Fonction principale"""
    afficher_banniere()

    args = parse_arguments()

    # Mode ligne de commande
    if args.all:
        print("\n  🚀 Mode : Toutes les étapes obligatoires")
        resultats = executer_toutes_etapes(obligatoires_only=True)
        afficher_recap(resultats)

    elif args.full:
        print("\n  🚀 Mode : TOUTES les étapes")
        resultats = executer_toutes_etapes(obligatoires_only=False)
        afficher_recap(resultats)

    elif args.etapes:
        etapes = [e for e in args.etapes if e in ETAPES]
        print(f"\n  🚀 Mode : Étapes {', '.join(etapes)}")
        resultats = executer_plusieurs_etapes(etapes)
        afficher_recap(resultats)

    elif args.step:
        if args.step in ETAPES:
            succes, temps, erreur = executer_etape(args.step)
            if succes:
                print(f"\n  ✅ Terminé en {temps:.2f}s")
        else:
            print(f"❌ Étape {args.step} inconnue")

    else:
        # Mode interactif par défaut
        mode_interactif()


if __name__ == '__main__':
    main()