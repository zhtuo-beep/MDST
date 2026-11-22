#!/usr/bin/env python3
"""
City Generator: the visual interface.

    python app.py                 # opens http://localhost:7860 in your browser
    python app.py --share         # also prints a temporary public link
    python app.py --port 8000     # a different port

Hosted (e.g. a Hugging Face Space, detected by its SPACE_ID variable, or
CITYGEN_HOSTED=1): each visitor gets their own gallery, and galleries older
than a few hours are deleted so the disk never fills up.

Two modes:
  Prompt mode   describe a city in plain English; your text2city model picks
                the 10 parameters and citygen builds the city
  Manual mode   set the 10 parameters yourself

Every city is saved to data/gallery/ as
    .npz  the grid             .png  2D preview
    .stl  printable model      .glb  coloured model for the 3D viewer
No Blender needed: the meshes come from citygen/mesh.py.
"""
import argparse
import html
import json
import os
import random
import re
import shutil
import time
import sys
import threading
from pathlib import Path

import warnings

import matplotlib

# Gradio 5 warns that theme/css move to launch() in Gradio 6; this file handles both.
warnings.filterwarnings("ignore", message=".*(parameter in the Blocks constructor|default value of .allow_tags.).*")
matplotlib.use("Agg")                     # draw to files, never open a window
from matplotlib.figure import Figure      # noqa: E402  (thread-safe, unlike pyplot)

import gradio as gr                       # noqa: E402
import numpy as np                        # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from citygen import (CITY_ARCHETYPES, TERRAIN_ARCHETYPES, LAND_BUILDING,  # noqa: E402
                     LAND_EMPTY, LAND_GREEN, LAND_ROAD)
from citygen.mesh import write_glb, write_stl                            # noqa: E402
from generate_city import build_city                                     # noqa: E402
from text2city.infer import kwargs_to_cli_args                           # noqa: E402
from text2city.schema import PARAM_BY_NAME, to_generator_kwargs          # noqa: E402

GALLERY = ROOT / "data" / "gallery"
CHECKPOINT = ROOT / "text2city" / "checkpoint"
DATASET = ROOT / "data" / "text2city_dataset.csv"
SIZES = [64, 96, 128, 160]
HOSTED = bool(os.environ.get("SPACE_ID") or os.environ.get("CITYGEN_HOSTED"))
HOSTED_KEEP_HOURS = 6       # hosted: delete a visitor's cities this long after their last one
CAMERA = (-120, 55, None)   # 3D view: (rotation, tilt from straight down, distance) in degrees
BACKGROUND = (0.94, 0.95, 0.93, 1.0)

EXAMPLES = [
    "A harbour city with high mountains on a straight line",
    "A hillside city with lots of green space",
    "A dense polycentric city with four business hubs",
    "A sparse little town on flat land with patchy roads",
    "A long linear city stretched along one road",
]

# Gradio 6 moved a few options around; support both 5.x and 6.x.
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
CHAT_KWARGS = {} if GRADIO_6 else {"type": "messages"}
if "allow_tags" in __import__("inspect").signature(gr.Chatbot.__init__).parameters:
    CHAT_KWARGS["allow_tags"] = False   # plain HTML only; also silences a Gradio 5 warning


# ---------------------------------------------------------------------------
# the text model: loaded ONCE at startup, shared by every request
# ---------------------------------------------------------------------------
class TextModel:
    def __init__(self):
        self.model = self.embedder = self._predict = None
        self.error = None
        self.lock = threading.Lock()

    def load(self, allow_stale=False):
        try:
            import torch
            from text2city.evaluate import load_checkpoint, predict
            from text2city.freshness import is_stale

            ckpt = CHECKPOINT / "model.pt"
            if not ckpt.exists():
                self.error = ("There's no trained model yet, so prompt mode is off. Train one:\n\n"
                              "    python text2city/generate_dataset.py\n    python text2city/train.py\n\n"
                              "then restart the app. Manual mode works without it.")
                return
            fingerprint = torch.load(ckpt, map_location="cpu", weights_only=False).get("fingerprint")
            if is_stale(fingerprint, str(DATASET)) and not allow_stale:
                self.error = ("The trained model is STALE: your dataset, text2city code or terrain list "
                              "changed after it was trained. Retrain:\n\n"
                              "    python text2city/generate_dataset.py\n    python text2city/train.py\n\n"
                              "then restart the app (or start it with --allow-stale).")
                return
            self.model, self.embedder = load_checkpoint(str(CHECKPOINT), str(DATASET))
            self._predict = predict
        except Exception as e:  # never stop the app: manual mode still works
            self.error = f"Couldn't load the text model ({type(e).__name__}: {e}). Manual mode still works."

    def params(self, text: str) -> dict:
        with self.lock:
            p = self._predict(self.model, self.embedder, [text])[0]
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()}


MODEL = TextModel()


# ---------------------------------------------------------------------------
# building and saving one city
# ---------------------------------------------------------------------------
_name_lock = threading.Lock()


def gallery_dir(request: gr.Request = None) -> Path:
    """
    Where this visitor's cities go. Locally there's one user, so everything
    lands in data/gallery/. Hosted, each browser session gets its own
    subfolder, so visitors never see (or download) each other's cities.
    """
    if HOSTED and request is not None and getattr(request, "session_hash", None):
        return GALLERY / re.sub(r"[^A-Za-z0-9_-]", "", request.session_hash)
    return GALLERY


def _cleanup_old_sessions() -> None:
    """Hosted only: remove session folders nobody has added to for HOSTED_KEEP_HOURS."""
    if not HOSTED or not GALLERY.exists():
        return
    cutoff = time.time() - HOSTED_KEEP_HOURS * 3600
    for folder in GALLERY.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)


def _next_stem(label: str, folder: Path) -> Path:
    """<gallery>/city_007_a-harbour-city -- numbered, so the list stays in order."""
    with _name_lock:
        folder.mkdir(parents=True, exist_ok=True)
        numbers = [int(m.group(1)) for p in folder.glob("city_*.npz")
                   if (m := re.match(r"city_(\d+)", p.name))]
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40].strip("-") or "city"
        stem = folder / f"city_{max(numbers, default=0) + 1:03d}_{slug}"
        Path(f"{stem}.npz").touch()          # claim the number before releasing the lock
        return stem


LAND_USE_COLORS = {LAND_EMPTY: (0.85, 0.85, 0.8), LAND_BUILDING: (0.75, 0.2, 0.2),
                   LAND_ROAD: (0.3, 0.3, 0.3), LAND_GREEN: (0.25, 0.6, 0.25)}


def render_preview(city, path: str, title: str) -> None:
    """The same four panels as scripts/visualize_2d.py, drawn straight to a PNG."""
    fig = Figure(figsize=(16, 4.3))
    axes = fig.subplots(1, 4)
    im0 = axes[0].imshow(np.where(city.is_water, np.nan, city.terrain_height), cmap="terrain")
    axes[0].set_title("Terrain height")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[1].imshow(city.is_water, cmap="Blues")
    axes[1].set_title("Water")
    im2 = axes[2].imshow(city.building_height, cmap="magma")
    axes[2].set_title("Building height")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    rgb = np.zeros((*city.land_use.shape, 3), dtype=np.float32)
    for code, color in LAND_USE_COLORS.items():
        rgb[city.land_use == code] = color
    axes[3].imshow(rgb)
    axes[3].set_title("Land use (red=building, gray=road, green=park)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def make_city(kwargs: dict, seed: int, size: int, label: str, prompt: str = None,
              folder: Path = GALLERY) -> dict:
    """Build one city and save its files (.npz .png .stl .glb .json). Returns their paths."""
    _cleanup_old_sessions()
    city = build_city(size=int(size), seed=int(seed), **kwargs)
    stem = _next_stem(label, folder)
    paths = {ext: f"{stem}.{ext}" for ext in ("npz", "png", "stl", "glb", "json")}
    city.save(paths["npz"])
    Path(paths["json"]).write_text(json.dumps(
        {"prompt": prompt, "kwargs": kwargs, "seed": int(seed), "size": int(size)}, indent=2))
    render_preview(city, paths["png"], f"{Path(stem).name}   ({city.size}x{city.size}, seed {seed})")
    write_stl(city, paths["stl"])
    write_glb(city, paths["glb"])
    return paths


def _file_url(path: Path) -> str:
    return "gradio_api/file=" + str(path.resolve())


def gallery_html(request: gr.Request = None) -> str:
    """The sidebar list: this visitor's saved cities, newest first, with download links."""
    return gallery_html_for_folder(gallery_dir(request))


def gallery_html_for_folder(folder: Path) -> str:
    items = []
    for stl in sorted(folder.glob("city_*.stl"), reverse=True):
        stem = stl.with_suffix("")
        number = stem.name.split("_")[1]
        info = {}
        if stem.with_suffix(".json").exists():
            try:
                info = json.loads(stem.with_suffix(".json").read_text())
            except ValueError:
                pass
        title = info.get("prompt") or ("Manual" if stem.name.endswith("_manual") else stem.name)
        k = info.get("kwargs")
        sub = f"{_fmt(k['terrain_archetype'])} · {_fmt(k['city_archetype'])} · seed {info['seed']}" if k else ""
        links = [f'<a href="{_file_url(stl)}" download>STL</a>']
        for ext, label, attr in ((".glb", "GLB", "download"), (".png", "PNG", 'target="_blank"')):
            if stem.with_suffix(ext).exists():
                links.append(f'<a href="{_file_url(stem.with_suffix(ext))}" {attr}>{label}</a>')
        items.append(f'<div class="city-item"><div class="city-title">City {int(number)}: '
                     f'{html.escape(title)}</div><div class="city-sub">{html.escape(sub)}</div>'
                     f'<div class="city-links">{" · ".join(links)}</div></div>')
    if not items:
        return '<div class="city-sub">No cities yet. Describe one, or use manual mode.</div>'
    return '<div class="city-list">' + "".join(items) + "</div>"


# ---------------------------------------------------------------------------
# what we show about a city
# ---------------------------------------------------------------------------
NICE_NAMES = [("terrain_archetype", "Terrain"), ("city_archetype", "City type"), ("clusters", "Clusters"),
              ("density", "Density"), ("cluster_tightness", "Cluster tightness"), ("relief", "Relief"),
              ("water_level", "Water level"), ("green_amount", "Green amount"),
              ("green_override", "Green override"), ("road_continuity", "Road continuity")]


def _fmt(value) -> str:
    if value is None:
        return "auto (terrain default)"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("_", " ")


def command_for(kwargs: dict, seed: int, size: int, npz_path: str) -> str:
    """The generate_city.py command that rebuilds exactly this city (Week 6's receipt)."""
    out = Path(npz_path).relative_to(ROOT)
    args = kwargs_to_cli_args(kwargs) + ["--size", str(size), "--seed", str(seed), "--out", str(out)]
    return "python scripts/generate_city.py " + " ".join(args)


def details_markdown(kwargs: dict, seed: int, size: int, paths: dict) -> str:
    rows = "\n".join(f"| {nice} | {_fmt(kwargs[key])} |" for key, nice in NICE_NAMES)
    return (f"| Parameter | Value |\n|---|---|\n{rows}\n| Seed | {seed} |\n\n"
            f"Rebuild it from a terminal:\n\n```\n{command_for(kwargs, seed, size, paths['npz'])}\n```")


def headline(kwargs: dict) -> str:
    n = kwargs["clusters"]
    # a linear city always strings at least 3 centres along its line, so a count would mislead
    count = "" if kwargs["city_archetype"] == "linear" else f", {n} cluster{'s' if n != 1 else ''}"
    return (f"**{_fmt(kwargs['terrain_archetype']).capitalize()} terrain, "
            f"{_fmt(kwargs['city_archetype'])} city{count}**")


def reply_messages(kwargs: dict, seed: int, size: int, paths: dict, intro: str = "") -> list:
    text = (f"{intro}{headline(kwargs)} · seed {seed}\n\n"
            f"<details><summary>Parameters used</summary>\n\n{details_markdown(kwargs, seed, size, paths)}\n\n</details>")
    return [{"role": "assistant", "content": text},
            {"role": "assistant", "content": {"path": paths["png"]}},
            {"role": "assistant", "content": {"path": paths["stl"]}}]


# ---------------------------------------------------------------------------
# prompt mode
# ---------------------------------------------------------------------------
def add_user_message(text, history):
    text = (text or "").strip()
    if not text:
        return gr.update(), history
    return "", history + [{"role": "user", "content": text}]


def _text_of(content) -> str:
    """A chat message's text. Gradio 5 sends a string; Gradio 6 a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, (list, tuple)):
        return " ".join(_text_of(c) for c in content if not isinstance(c, dict) or c.get("type", "text") == "text")
    return ""


def answer_prompt(history, size, last, request: gr.Request = None):
    if not history or history[-1]["role"] != "user":
        return history, gr.update(), last, gr.update()
    text = _text_of(history[-1]["content"]).strip()
    if MODEL.error:
        return history + [{"role": "assistant", "content": MODEL.error}], gr.update(), last, gr.update()
    kwargs = to_generator_kwargs(MODEL.params(text))
    seed = random.randrange(10_000)
    paths = make_city(kwargs, seed, size, text, prompt=text, folder=gallery_dir(request))
    last = {"text": text, "kwargs": kwargs, "seed": seed}
    return history + reply_messages(kwargs, seed, size, paths), paths["glb"], last, gallery_html(request)


def new_variation(history, size, last, request: gr.Request = None):
    if not last:
        gr.Info("Describe a city first.")
        return history, gr.update(), last, gr.update()
    seed = random.randrange(10_000)
    paths = make_city(last["kwargs"], seed, size, last["text"], prompt=last["text"], folder=gallery_dir(request))
    last = {**last, "seed": seed}
    intro = f"New variation of *“{last['text']}”*: same parameters, different seed.\n\n"
    return (history + reply_messages(last["kwargs"], seed, size, paths, intro), paths["glb"], last,
            gallery_html(request))


# ---------------------------------------------------------------------------
# manual mode
# ---------------------------------------------------------------------------
def manual_generate(terrain, city_type, clusters, density, tightness, relief,
                    water_auto, water, green, override, roads, seed, size, request: gr.Request = None):
    kwargs = {"terrain_archetype": terrain, "city_archetype": city_type, "clusters": int(clusters),
              "density": float(density), "cluster_tightness": float(tightness), "relief": float(relief),
              "water_level": None if water_auto else float(water), "green_amount": float(green),
              "green_override": bool(override), "road_continuity": float(roads)}
    seed = int(seed) if seed is not None else 0
    paths = make_city(kwargs, seed, size, "manual", folder=gallery_dir(request))
    summary = (f"{headline(kwargs)} · seed {seed}\n\nRebuild it from a terminal:\n\n"
               f"```\n{command_for(kwargs, seed, size, paths['npz'])}\n```")
    return paths["png"], paths["glb"], paths["stl"], summary, gallery_html(request)


def send_to_manual(last):
    """'Edit in manual mode': copy the last prompt-mode city's values into the sliders."""
    if not last:
        gr.Info("Generate a city in prompt mode first.")
        return [gr.update()] * 13
    k = last["kwargs"]
    water_auto = k["water_level"] is None
    return [gr.update(value=k["terrain_archetype"]), gr.update(value=k["city_archetype"]),
            gr.update(value=k["clusters"]), gr.update(value=k["density"]),
            gr.update(value=k["cluster_tightness"]), gr.update(value=k["relief"]),
            gr.update(value=water_auto),
            gr.update(value=PARAM_BY_NAME["water_level"].default if water_auto else k["water_level"],
                      interactive=not water_auto),
            gr.update(value=k["green_amount"]), gr.update(value=k["green_override"]),
            gr.update(value=k["road_continuity"]), gr.update(value=last["seed"]),
            gr.Tabs(selected="manual")]


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
CSS = """
footer { display: none !important; }
.city-list { max-height: 62vh; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
.city-item { padding: 8px 10px; border-radius: 8px; background: var(--block-background-fill);
             border: 1px solid var(--border-color-primary); }
.city-title { font-weight: 600; font-size: 0.92em; line-height: 1.3; }
.city-sub { font-size: 0.8em; opacity: 0.7; margin: 2px 0 4px; }
.city-links a { font-size: 0.85em; }
"""


def build_ui() -> gr.Blocks:
    spec = PARAM_BY_NAME
    blocks_kwargs = {} if GRADIO_6 else {"theme": gr.themes.Soft(primary_hue="green"), "css": CSS}
    with gr.Blocks(title="City Generator", **blocks_kwargs) as demo:
        last = gr.State(None)

        with gr.Sidebar(open=True, width=330):
            gr.Markdown("### Your cities")
            gallery = gr.HTML("" if HOSTED else gallery_html())
            size = gr.Radio(SIZES, value=128, label="Grid size",
                            info="Bigger = more detail, slower and larger files")
            if HOSTED:
                gr.Markdown(f"Your cities are private to this browser tab and kept for about "
                            f"{HOSTED_KEEP_HOURS} hours. Download the `.stl` to keep or 3D-print one.")
            else:
                gr.Markdown("Every city is saved in `data/gallery/`: the 2D preview, the grid, "
                            "a printable `.stl` and a coloured `.glb`.")

        gr.Markdown("# City Generator")
        with gr.Tabs() as tabs:
            # ---------------- prompt mode
            with gr.Tab("Prompt mode", id="prompt"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(elem_id="chat", height=560, show_label=False,
                                          placeholder="Describe a city and press **Enter**.<br>"
                                                      "Each message is a new city: the model reads one "
                                                      "sentence at a time, so say everything at once.",
                                          **CHAT_KWARGS)
                        prompt = gr.Textbox(placeholder="e.g. a dense city in the hills with three business hubs",
                                            show_label=False, submit_btn=True, autofocus=True)
                        gr.Examples(EXAMPLES, inputs=prompt, label="Try one")
                    with gr.Column(scale=2):
                        prompt_view = gr.Model3D(label="3D view of the latest city (drag to rotate)",
                                                 height=430, clear_color=BACKGROUND, camera_position=CAMERA)
                        with gr.Row():
                            variation_btn = gr.Button("New variation (new seed)")
                            edit_btn = gr.Button("Edit in manual mode", variant="primary")

            # ---------------- manual mode
            with gr.Tab("Manual mode", id="manual"):
                with gr.Row(equal_height=False):
                    with gr.Column():
                        gr.Markdown("#### Environment")
                        terrain = gr.Dropdown(list(TERRAIN_ARCHETYPES), value=spec["terrain"].default,
                                              label="Terrain type")
                        relief = gr.Slider(*spec["relief"].value_range, value=spec["relief"].default,
                                           step=0.05, label="Relief", info="how hilly")
                        water_auto = gr.Checkbox(True, label="Water level: auto (terrain default)")
                        water = gr.Slider(*spec["water_level"].value_range, value=spec["water_level"].default,
                                          step=0.01, label="Water level", interactive=False)
                        green = gr.Slider(*spec["green_amount"].value_range, value=spec["green_amount"].default,
                                          step=0.05, label="Green amount")
                        override = gr.Checkbox(spec["green_override"].default, label="Green override",
                                               info="nature takes over: green space replaces buildings")
                        roads = gr.Slider(*spec["road_continuity"].value_range,
                                          value=spec["road_continuity"].default, step=0.05,
                                          label="Road continuity", info="0 = patchy, 1 = complete grid")
                    with gr.Column():
                        gr.Markdown("#### City")
                        city_type = gr.Dropdown(list(CITY_ARCHETYPES), value=spec["city_archetype"].default,
                                                label="City type")
                        lo, hi = spec["num_clusters"].value_range
                        clusters = gr.Slider(int(lo), int(hi), value=int(spec["num_clusters"].default), step=1,
                                             label="Clusters", info="polycentric uses at least 2")
                        density = gr.Slider(*spec["density"].value_range, value=spec["density"].default,
                                            step=0.05, label="Density")
                        tightness = gr.Slider(*spec["cluster_tightness"].value_range,
                                              value=spec["cluster_tightness"].default, step=0.05,
                                              label="Cluster tightness")
                        with gr.Row(equal_height=True):
                            seed = gr.Number(42, precision=0, label="Seed", scale=3)
                            dice = gr.Button("Random seed", scale=1)
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        manual_png = gr.Image(label="2D preview", interactive=False)
                        manual_info = gr.Markdown()
                    with gr.Column(scale=2):
                        manual_view = gr.Model3D(label="3D view (drag to rotate)", height=380,
                                                 clear_color=BACKGROUND, camera_position=CAMERA)
                        manual_stl = gr.File(label="Printable model (.stl)", interactive=False)

        # ---------------- wiring
        busy = dict(show_progress="minimal")
        prompt.submit(add_user_message, [prompt, chat], [prompt, chat], queue=False).then(
            answer_prompt, [chat, size, last], [chat, prompt_view, last, gallery], **busy)
        variation_btn.click(new_variation, [chat, size, last], [chat, prompt_view, last, gallery], **busy)
        manual_inputs = [terrain, city_type, clusters, density, tightness, relief,
                         water_auto, water, green, override, roads, seed]
        edit_btn.click(send_to_manual, last, manual_inputs + [tabs])
        water_auto.change(lambda a: gr.update(interactive=not a), water_auto, water, queue=False)
        dice.click(lambda: random.randrange(10_000), None, seed, queue=False)
        generate_btn.click(manual_generate, manual_inputs + [size],
                           [manual_png, manual_view, manual_stl, manual_info, gallery], **busy)
        demo.load(gallery_html, None, gallery)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="also create a temporary public link")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("--allow-stale", action="store_true", help="use the text model even if it's stale")
    args = parser.parse_args()

    print("loading the text model ...")
    MODEL.load(allow_stale=args.allow_stale)
    print("  " + (MODEL.error.splitlines()[0] if MODEL.error else "ready"))
    GALLERY.mkdir(parents=True, exist_ok=True)

    launch_kwargs = {"theme": gr.themes.Soft(primary_hue="green"), "css": CSS} if GRADIO_6 else {}
    if HOSTED:
        print(f"hosted mode: per-visitor galleries, kept {HOSTED_KEEP_HOURS} h")
    build_ui().queue(default_concurrency_limit=2).launch(
        server_port=args.port, share=args.share and not HOSTED, inbrowser=not (args.no_browser or HOSTED),
        allowed_paths=[str(GALLERY)], **launch_kwargs)


if __name__ == "__main__":
    main()
