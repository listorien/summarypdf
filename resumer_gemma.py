import os
import time
from tqdm import tqdm
import threading
from pypdf import PdfReader
from llama_cpp import Llama

# === CONFIGURATION ===
BASE_DIR        = os.path.dirname(__file__)
DOSSIER_DE_BASE = BASE_DIR
GGUF_MODEL_PATH = os.path.join(BASE_DIR, "gemma-3-4b-it-q4_0_s.gguf")
FICHIER_SORTIE  = os.path.join(BASE_DIR, "summary_pdf.txt")
MAX_PAGES       = 30
CTX_TOKENS      = 131072
MAX_OUTPUT_TOKENS = 128  # tokens générés par le modèle
THREADS         = 8
GPU_LAYERS      = -1
N_BATCH         = 64
REFRESH_INTERVAL = 2  # secondes entre les mises à jour de la barre

# === État global pour moyenne temporelle et statistiques ===
avg_time_per_token = None  # en secondes/token, mis à jour en cours d'exécution
total_tokens = 0
debut_programme = time.time()

# === Lire fichiers déjà traités ===
deja_traite = set()
if os.path.exists(FICHIER_SORTIE):
    with open(FICHIER_SORTIE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Chemin :"):
                deja_traite.add(line[len("Chemin :"):].strip())

# === Chargement du modèle ===
print("🔄 Chargement du modèle Gemma-3 (GPU)…")
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=CTX_TOKENS,
    n_threads=THREADS,
    n_batch=N_BATCH,   # traite 256 nouveaux tokens en parallèle
    n_gpu_layers=GPU_LAYERS,
    verbose=False
)

# Préparation du prompt de base
PROMPT_PREFIX = """### Instruction:
Ce document est un fichier PDF. Donne un titre pertinent et un résumé clair en français.

### Document :
"""

PROMPT_SUFFIX = """
### Réponse attendue :
Titre :"""

PROMPT_TOKENS = len(llm.tokenize((PROMPT_PREFIX + PROMPT_SUFFIX).encode("utf-8")))

# === Fonctions utilitaires ===

def trouver_pdfs(dossier):
    for racine, _, fichiers in os.walk(dossier):
        for fichier in fichiers:
            if fichier.lower().endswith(".pdf"):
                yield os.path.join(racine, fichier)

def extraire_texte(pdf_path, max_pages=MAX_PAGES):
    try:
        reader = PdfReader(pdf_path)
        pages = reader.pages[:max_pages]
        return "".join(page.extract_text() or "" for page in pages)
    except Exception as e:
        print(f"❌ Erreur lecture PDF {pdf_path}: {e}")
        return ""

def compter_tokens(texte):
    try:
        return len(llm.tokenize(texte.encode("utf-8")))
    except Exception as e:
        print(f"❌ Erreur tokenisation : {e}")
        return 0

def tronquer_texte_si_necessaire(texte, max_tokens):
    tokens = llm.tokenize(texte.encode("utf-8"))
    if len(tokens) > max_tokens:
        print(f"⚠️ Texte tronqué à {max_tokens} tokens")
        tokens = tokens[:max_tokens]
        return llm.detokenize(tokens).decode("utf-8", errors="ignore")
    return texte

# === Traitement principal ===

print(f"📂 Analyse du dossier : {DOSSIER_DE_BASE}")
pdfs = list(trouver_pdfs(DOSSIER_DE_BASE))
print(f"📄 {len(pdfs)} PDF(s) détectés.")

for chemin_pdf in tqdm(pdfs):
    if chemin_pdf in deja_traite:
        print(f"⏭️  Déjà traité : {chemin_pdf}")
        continue

    # Affichage de l'heure actuelle avant traitement
    heure_actuelle = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕒 Heure actuelle : {heure_actuelle}")

    texte = extraire_texte(chemin_pdf)
    nb_caract = len(texte)
    print(f"ℹ️  {chemin_pdf} — {nb_caract} caractères extraits")

    if nb_caract < 100:
        print("⚠️ Document vide ou illisible, ignoré.")
        continue

    max_input_tokens = CTX_TOKENS - PROMPT_TOKENS - MAX_OUTPUT_TOKENS
    texte = tronquer_texte_si_necessaire(texte, max_input_tokens)
    nb_tokens = compter_tokens(texte)
    print(f"🔢 {nb_tokens} tokens à envoyer au modèle")

    # Estimation avant appel
    if avg_time_per_token is not None:
        est = nb_tokens * avg_time_per_token
        print(f"⏳ Estimation du temps d'inférence : {est:.2f}s")
    else:
        print("⏳ Pas encore assez de données pour estimer le temps")

    # Appel au modèle et mesure
    prompt = f"{PROMPT_PREFIX}{texte}{PROMPT_SUFFIX}"

    def appeler_modele():
        return llm(prompt, max_tokens=MAX_OUTPUT_TOKENS, temperature=0.7, stop=["###"])

    start = time.time()
    if avg_time_per_token is not None:
        estimation = nb_tokens * avg_time_per_token
        pbar = tqdm(total=estimation, unit="s", mininterval=REFRESH_INTERVAL)
        resultat = {}

        def tache():
            resultat["out"] = appeler_modele()

        th = threading.Thread(target=tache)
        th.start()
        while th.is_alive():
            time.sleep(REFRESH_INTERVAL)
            incr = min(REFRESH_INTERVAL, max(0, estimation - pbar.n))
            pbar.update(incr)
        th.join()
        elapsed = time.time() - start
        pbar.update(max(0, estimation - pbar.n))
        pbar.close()
        out = resultat["out"]
    else:
        out = appeler_modele()
        elapsed = time.time() - start

    # Mise à jour de la moyenne temporelle (exponentielle)
    time_per_token = elapsed / nb_tokens if nb_tokens > 0 else 0
    if avg_time_per_token is None:
        avg_time_per_token = time_per_token
    else:
        alpha = 0.1
        avg_time_per_token = alpha * time_per_token + (1 - alpha) * avg_time_per_token

    # Affichage performance réelle
    tps = nb_tokens / elapsed if elapsed > 0 else 0
    print(f"⏱️  Temps réel : {elapsed:.2f}s — ⚡ {tps:.1f} tok/s")

    # Écriture du résumé
    resume_final = out["choices"][0]["text"].strip()
    nb_tokens_out = compter_tokens(resume_final)
    nb_tokens_input = PROMPT_TOKENS + nb_tokens
    total_tokens += nb_tokens_input + nb_tokens_out
    duree = time.time() - debut_programme
    moyenne_globale = total_tokens / duree if duree > 0 else 0
    print(f"📊 Total cumulé : {total_tokens} tokens — Moyenne {moyenne_globale:.1f} tok/s")

    bloc = f"Chemin : {chemin_pdf}\n{resume_final}\n\n{'-'*80}\n"
    try:
        with open(FICHIER_SORTIE, "a", encoding="utf-8") as f:
            f.write(bloc)
            f.flush()
        print(f"✅ Résumé écrit dans {FICHIER_SORTIE}")
        deja_traite.add(chemin_pdf)
    except Exception as e:
        print(f"❌ Erreur écriture fichier : {e}")
