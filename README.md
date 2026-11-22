# MDST City Generator

Generate a procedural city with terrain, buildings, roads, and parks. The
Gradio interface shows a 2D map and a rotatable 3D model and exports a binary
STL for slicing and 3D printing. Choose the settings in Manual mode or describe
a city in Prompt mode with a locally trained model.

## Run locally

Use Python 3.10 or newer. From the repository root on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Manual mode works as soon
as the app starts. Results are saved in `data/gallery/`. Press Ctrl+C in
the terminal to stop the server.

## Enable Prompt mode

Run these commands from the repository root using the same virtual
environment's Python. On Windows PowerShell, replace `python` with
`.\.venv\Scripts\python.exe` if the environment is not activated.

```bash
python -m pip install -r text2city/requirements.txt
python text2city/generate_dataset.py
python text2city/train.py
python app.py
```

The default training command downloads the sentence embedding model on first
use. To train without that download, use
`python text2city/train.py --embedder tfidf`. On Intel Macs, see the dependency
pins in [text2city/requirements.txt](text2city/requirements.txt).

## Command line tools

```bash
python scripts/generate_city.py --terrain hills --size 128 --seed 7 --out data/city.npz
python scripts/visualize_2d.py data/city.npz
python text2city/infer.py "a dense city in the hills" --seed 7 --preview
python scripts/check_stl.py path/to/city.stl
```

The app exports meshes without Blender. To create an editable Blender scene
from a saved city, install Blender and run:

```bash
blender --background --python scripts/export_to_blender.py -- data/city.npz
```

The repository contains source code and setup files. Generated cities, training
data, and model checkpoints are created locally.
