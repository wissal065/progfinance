# progfinance

## Lancer l'application dans VS Code

### 1) Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

Sous Windows (PowerShell) :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3) Démarrer Streamlit

```bash
python -m streamlit run app.py
```

## Lancement direct via VS Code (Run and Debug)

Le dépôt contient une configuration `.vscode/launch.json`.
Dans VS Code :

1. Ouvre l'onglet **Run and Debug**
2. Sélectionne **Streamlit: app.py**
3. Clique sur **Start Debugging**
