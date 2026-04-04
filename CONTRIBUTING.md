# Contributing & Local Development

This guide explains how to work on the project — either in a GitHub Codespace or in a local clone.

---

## Opening in GitHub Codespaces

GitHub Codespaces provides a browser-based VS Code environment with all dependencies installed automatically.

1. On the main repository page, click the green **Code** button.
2. Select the **Codespaces** tab.
3. Click **Create codespace on main** (or choose a branch).

The `.devcontainer/devcontainer.json` configuration in this repository will:
- Spin up a Python 3.11 container (matching the local minimum requirement of Python 3.10+).
- Run `pip install -r requirements.txt` automatically.
- Install the Jupyter and Python VS Code extensions.
- Forward port **5000** for the Flask API server.

Once the Codespace is ready you can run any notebook, train the model, or start the API — all from within the browser.

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/thmolena/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization.git
cd Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization

# 2. Create and activate a virtual environment (Python 3.10+)
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the project

| Task | Command |
|---|---|
| Train the GNN model | `python -m src.train` |
| Start the Flask API server | `python -m src.server` |
| Open notebooks | `jupyter lab notebooks/` |
| Export notebooks to HTML | `python scripts/export_notebook_html.py` |

---

## Reviewing changes without committing

If you want to inspect changes proposed by a collaborator (or a Copilot coding session) before committing them yourself:

1. Open the pull request on GitHub.
2. Review the diff directly in the browser, or check out the branch locally:
   ```bash
   gh pr checkout <number>
   # or without the GitHub CLI:
   git fetch origin pull/<number>/head:pr-<number>
   git checkout pr-<number>
   ```
3. Review the changes in GitHub Desktop or your editor of choice.

This gives you full control over what lands in the project history under your name.
