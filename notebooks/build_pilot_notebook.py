"""Build the clean Colab tutorial; pin to a commit containing pilot_support.py."""
from pathlib import Path
import argparse
import json
import re
import textwrap

parser=argparse.ArgumentParser()
parser.add_argument('--revision',required=True)
args=parser.parse_args()
assert re.fullmatch('[0-9a-f]{40}',args.revision)
cells=[]
def add(kind,source,tags=()):
    cell={'cell_type':kind,'id':f'pilot-{len(cells):02d}',
          'metadata':{'tags':list(tags)} if tags else {},
          'source':textwrap.dedent(source).strip()+'\n'}
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)
def md(s):add('markdown',s)
def code(s,tags=()):add('code',s,tags)

md('''
# Moral relation readouts: an end-to-end tutorial
### Can an internal readout change while the model's answer stays the same?

This notebook reproduces the **completed direct-journal calibration pilot** from
*Context-Conditioned Moral Relation Readouts in Language Models*. It walks through
inputs, controls, exact checkpoints, numerical validation, activation capture,
Jeff's frozen linear probes, and interpretation.

**Start with `replay` and Runtime → Run all.** It recomputes the published contrasts
from saved per-call scores on CPU. Change to `live` in a fresh GPU runtime to
collect new activations and run the complete experiment for one or both models.
Replay does not re-extract activations: the original raw activation archives are
not distributed in the repository. Live mode creates a new complete archive.

| Path | What you run | What you need |
|---|---|---|
| `replay` | Original per-call CSVs → independent aggregation → interactive plots → export | CPU; no credentials or model downloads |
| `live` | Same battery → pinned model → numerical gate → 576 captures/model → frozen scoring → plots → export | 40 GB-class BF16 GPU, such as an A100; approved Hugging Face model access |

The original run used an **RTX A6000 48 GB**, one checkpoint at a time. Live mode
retains the Colab CUDA/PyTorch runtime, records it, and pins the other inference
libraries. This is a rerun of the methodology, not a promise of bitwise equality
across GPUs and software. A T4 cannot run this tutorial's unquantized BF16 path.
GPU availability varies; see [Colab's resource FAQ](https://research.google.com/colaboratory/faq.html).

**Study boundary:** no agents interact in this completed experiment. A change in
this generic Supports/Opposes readout does not establish moral adoption,
persistence, personal value alignment, or contagion.
''')
code('''
#@title 1. Choose your learning path
MODE = "replay" #@param ["replay", "live"]
MODEL = "both" #@param ["gemma", "llama", "both"]
INSTALL_DEPENDENCIES = True #@param {type:"boolean"}

from pathlib import Path
import sys, os, json, subprocess, tempfile, importlib.util
from datetime import datetime, timezone

assert MODE in {"replay", "live"} and MODEL in {"gemma", "llama", "both"}
SELECTED = ["gemma", "llama"] if MODEL == "both" else [MODEL]
WORK = (Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())) / "moral-pilot-tutorial"
WORK.mkdir(parents=True, exist_ok=True)
SESSION = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RESULTS = WORK / "results" / SESSION
(RESULTS / "analysis").mkdir(parents=True)
print(f"Mode: {MODE}; models: {SELECTED}; fresh results: {RESULTS}")
''',('configuration',))
md('''
## 2. Reproducible setup

The tutorial and scientific inputs come from a pinned project commit. The model
and tokenizer revisions, probe-file hashes, authored battery, and original scorer
are also frozen. Existing modified checkouts are rejected rather than overwritten.
Dependencies are installed before importing the analysis or inference libraries.

In live mode, model files are cached **outside** the result directory. Run one
model at a time; `both` unloads the first checkpoint before loading the second.
Allow roughly 30 GB free disk for one model, or 55 GB for both, plus outputs.
Model downloads and loading dominate this small experiment's setup time; the
original roughly 42-second calibration times exclude both and are not a Colab
runtime estimate.
''')
code('''
#@title Fetch pinned code and install dependencies
PROJECT_REVISION = "__REVISION__"
JEFF_REVISION = "ff3588c1d15ac89c6a3edd5da41c6fa121ad4157"
REPO = WORK / ("multiagent-" + PROJECT_REVISION[:12])
JEFF = WORK / ("geometry-" + JEFF_REVISION[:12])

def checkout(url, revision, target):
    if not target.exists():
        subprocess.run(["git", "init", "-q", str(target)], check=True)
        subprocess.run(["git", "remote", "add", "origin", url], cwd=target, check=True)
        subprocess.run(["git", "fetch", "--quiet", "--depth", "1", "origin", revision], cwd=target, check=True)
        subprocess.run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=target, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=target)
    if head != revision or dirty:
        raise RuntimeError(f"Use a clean pinned checkout: {target}")
    return target

checkout("https://github.com/stevenokada/multiagent.git", PROJECT_REVISION, REPO)
if MODE == "live":
    checkout("https://github.com/JeffVallyath/geometry-of-endorsement.git", JEFF_REVISION, JEFF)

if INSTALL_DEPENDENCIES:
    imports = {"numpy":"numpy>=1.26", "pandas":"pandas>=2", "matplotlib":"matplotlib>=3.8",
               "ipywidgets":"ipywidgets>=8,<9", "yaml":"PyYAML>=6"}
    missing = [package for module, package in imports.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    if MODE == "live":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "-r", str(REPO / "requirements-runpod.txt"),
                        "scikit-learn==1.7.1"], check=True)

# HF_HOME is set before Hugging Face libraries are imported; never export it.
os.environ["HF_HOME"] = str(WORK / "hf-cache")
sys.path.insert(0, str(REPO))
from notebooks import pilot_support as pilot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", "inline")
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML, SVG
pd.set_option("display.max_colwidth", 95)
print("Source revision:", PROJECT_REVISION)
display(pd.DataFrame(pilot.MODELS).T[["name", "revision", "hf_layer", "width", "batch_size"]])
'''.replace('__REVISION__',args.revision),('setup',))
md('''
## 3. Motivation: two measurements of the same forward pass

A categorical answer can stay fixed while a continuous readout moves. We therefore
measure both **native answer probabilities** and **frozen activation-probe scores**.
Neither alone is a validated measurement of an agent's personal moral commitment.

Each item asks whether a **named consideration provides a reason for or against an
action**. For example, honesty opposes lying about a defect even if someone changes
how much they prioritize honesty. This is why the starter battery is a calibration
anchor, not a complete moral-reasoning or alignment instrument.

The visual below explains the completed experiment. The gold future-study section
is a proposal, not another phase that this notebook automatically executes.
''')
code('''
#@title Explore the setup
visual = REPO / "docs/learning/methodology.svg"
if visual.exists():
    display(SVG(filename=str(visual)))
else:
    print("Flow: persona + journal + item → rendered prompt → model → native A/B probability + hidden vector → frozen probe → paired contrasts")
''',('visual',))
md('''
## 4. Inspect the battery and controls before running a model

Per checkpoint: **3 personas × 3 repeats × 8 items × 4 journal conditions = 288
observations**. Each observation has standard and reversed A/B answer mappings,
so there are **576 captured vectors**. Repeats are repeated measurements, not
independent social simulations. The eight anchors were authored for this pilot;
they are not held-out ValuePrism rows and their training overlap is unverified.

- **Neutral:** an ordinary introductory journal.
- **Repeat:** exactly the same neutral journal; a repeatability check.
- **Seeded:** an honesty-absolutist conviction written into the journal.
- **Mentioned/quoted:** the same conviction recorded as someone else's view,
  explicitly without adopting it; an exposure/quotation control.

Four Honesty targets and four Fairness/Safety controls are balanced for reference
Supports/Opposes labels. Those reference labels are never included in the prompt.
''')
code('''
#@title Load verified original evidence and inspect inputs
references = {key: pilot.load_reference(REPO, key) for key in SELECTED}
first = references[SELECTED[0]]
battery_table = pd.DataFrame(first["battery"]["items"])
display(battery_table[["id", "situation", "consideration", "reference_label", "on_target"]])
display(pd.DataFrame(first["design"]["personas"]))
for name, text in first["design"]["condition_journals"].items():
    display(Markdown(f"**{name}**\\n\\n{text}"))
''')
code('''
#@title Build a prompt: change the persona, condition, item, or answer mapping
from mindvirus.personas import PERSONAS
from mindvirus.probes import Battery
from mindvirus.relations import _relation_requests
battery = Battery.load(REPO / "config/hand-relation-battery.json")

def preview_prompt(persona, condition, item_id, mapping):
    p = next(p for p in PERSONAS if p.name == persona)
    item = next(i for i in battery.items if i.id == item_id)
    reqs = _relation_requests(p, first["design"]["condition_journals"][condition], item, 0)
    req = next(r for r in reqs if r["answer_mapping"] == mapping)
    print("SYSTEM / PERSONA AND JOURNAL:\\n" + req["system"])
    print("\\nUSER / QUESTION:\\n" + req["messages"][0]["content"])
    print("\\nThe HF tokenizer adds its pinned chat template and generation prompt.")
    print("Gemma folds the system instructions into the first user message; Llama retains the system role.")

prompt_controls = widgets.interactive(preview_prompt,
    persona=[p.name for p in PERSONAS[:3]],
    condition=list(first["design"]["condition_journals"]),
    item_id=[i.id for i in battery.items], mapping=["standard", "reversed"])
display(prompt_controls)
''',('interactive',))
md('''
**Pause and predict.** If quotation alone moves the internal score, what would
that tell us about interpreting a seeded-context shift?

<details><summary>Reveal a suggested answer</summary>
It shows sensitivity to changed context without demonstrated adoption. The next
experiment needs better controls for exposure, lexical overlap, attribution, and
persistence before we interpret a shift as a change in moral reasoning.
</details>

## 5. Live-mode access and hardware check

Replay skips this section automatically. For live mode, request access to the
[Llama checkpoint](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) and/or
[Gemma checkpoint](https://huggingface.co/google/gemma-2-9b-it) first. Set a Colab
secret named **HF_TOKEN** and enable notebook access, or enter a read token at the
hidden prompt. A token alone does not grant access to a gated model.

The token stays in the process environment; it is not placed in configuration,
notebook output, Git, or the result archive. No Drive mount or API judge key is
needed. See the [Hugging Face authentication guide](https://huggingface.co/docs/huggingface_hub/en/quick-start).
''')
code('''
#@title Verify live prerequisites (replay skips)
if MODE == "live":
    from importlib.metadata import version
    import shutil
    expected = {"transformers":"4.56.2", "tokenizers":"0.22.0",
                "huggingface-hub":"0.34.4", "accelerate":"1.10.1", "scikit-learn":"1.7.1"}
    assert all(version(name) == pinned for name, pinned in expected.items()), "Restart the runtime after package installation."
    display(pilot.gpu_preflight())
    free_gib = shutil.disk_usage(WORK).free / 1024**3
    print(f"Free disk: {free_gib:.1f} GiB. Models are cached outside results.")
    if free_gib < 25:
        raise RuntimeError("Less than 25 GiB free disk; free space before downloading checkpoints.")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
        except Exception:
            hf_token = None
    if not hf_token:
        from getpass import getpass
        hf_token = getpass("Hugging Face read token (hidden): ").strip()
    if not hf_token:
        raise RuntimeError("No Hugging Face token supplied; no model was loaded.")
    os.environ["HF_TOKEN"] = hf_token
    del hf_token
    from huggingface_hub import hf_hub_download
    for key in SELECTED:
        spec = pilot.MODELS[key]
        # Small gated file; verify checkpoint access before downloading its weight shards.
        hf_hub_download(spec["model"], "model.safetensors.index.json", revision=spec["revision"])
        probe_path = pilot.check_probe(JEFF, key)
        print(key, "access and frozen probe hash verified; HF layer", spec["hf_layer"])
else:
    print("Replay uses the checked-in evidence. No GPU, model download, or authentication requested.")
''',('gpu',))
md('''
## 6. Numerical gate → complete calibration → frozen scoring

Live mode reuses the original numerical driver, calibration implementation, and
scorer. It compares 48 mixed-length prompts serially and in batches of two, with
both answer mappings. The predeclared limits are maximum probability difference
≤0.01, relative vector L2 ≤0.02, cosine ≥0.999, and zero changed A/B decisions.

**Original outcome:** Gemma passed; Llama failed and its final calibration used
batch size one. This tutorial retains each comparison and uses serial calibration
if a completed comparison fails. Llama stays serial even if a new GPU passes.
An exception or an unexpected later batch fallback stops the run with partial
artifacts retained; it is not silently counted as a clean batched measurement.

The full calibration then captures the final real prompt-token hidden state:
Gemma HF index **28** (Jeff block 27), Llama **20** (block 19). Model inference is
BF16; vectors are saved as FP16 and scored offline in FP32. The native probability
comes from the same prompt forward pass. No answer is appended to agent memory.
No probe weights, intercepts, or scalers are fitted on this experiment.

The live cell below runs the complete procedure for each selected model. Its
implementation is in `notebooks/pilot_support.py`; the original measurement code
remains in `mindvirus/calibrate.py` and `audit/score_calibration_probes.py`.
''')
code('''
#@title Run the end-to-end experiment (or select saved results)
bundles = {}
if MODE == "live":
    for key in SELECTED:
        print(f"\\n=== {pilot.MODELS[key]['name']}: numerical gate → calibration → frozen scoring ===")
        output = pilot.run_live_model(REPO, JEFF, key, RESULTS / "live" / key)
        bundles[key] = pilot.load_live(output)
        display(json.loads((output / "run-manifest.json").read_text()))
else:
    bundles = references
    replay = RESULTS / "replay"
    replay.mkdir()
    for key, bundle in references.items():
        bundle["scores"].to_csv(replay / f"{key}-per_call_scores.csv", index=False)
        pilot.write_json(replay / f"{key}-native-report.json", bundle["native_report"])
        pilot.write_json(replay / f"{key}-probe-summary.json", bundle["probe_summary"])
    print("Using saved original measurements; no new checkpoint inference occurred.")
''',('experiment',))
md(r'''
## 7. What do the two scores mean?

The native answer score is the model's conditional probability of the token
mapped to Supports, restricted to A and B:

$$p_S=\frac{e^{\ell_S}}{e^{\ell_A}+e^{\ell_B}}.$$

Jeff's frozen probes instead read the captured vector $h$:

$$s_D(h)=h^\top d-b_D,\qquad s_L(h)=w^\top h+b_L,\qquad
z_k(h)=\frac{s_k(h)-\mu_k}{\sigma_k}.$$

A positive **raw** probe score means Supports under the original scoring
contract. Standardized zero is not the raw classification threshold. The frozen
standardization uses Jeff's selection-score mean and SD, not this battery's SD.
The resulting $z$ is neither a probability nor a new-study standardized effect size.

We average the repeated calls and the two answer mappings for each
persona–item–condition. **We do not negate a frozen probe score when the answer
symbols reverse.** A symbol swap does not reverse the semantic Supports direction.
Cosine similarity would also change the original measurement by normalizing the
activation norm and omitting the midpoint unless separately handled.
''')
code('''
#@title Verify full design and independently recompute contrasts
metrics, mapping_gaps, averages = {}, {}, {}
for key, bundle in bundles.items():
    metrics[key], mapping_gaps[key], averages[key] = pilot.summarize_scores(bundle["scores"])
    metrics[key].to_csv(RESULTS / "analysis" / f"{key}-paired-contrasts.csv", index=False)
    mapping_gaps[key].to_csv(RESULTS / "analysis" / f"{key}-mapping-gaps.csv", index=False)
    if MODE == "replay":
        for row in metrics[key].to_dict("records"):
            original = bundle["probe_summary"]["changes"][row["comparison"]][row["group"]][row["probe"]]
            assert np.isclose(row["mean_absolute_change"], original["mean_absolute_z_change"], atol=1e-12, rtol=0)
    print(key, len(bundle["scores"]), "complete captured-score rows; all four conditions and both mappings retained")

def native_overview(key, bundle):
    report = bundle["native_report"]
    native = report["calibration"]
    return {"model":pilot.MODELS[key]["name"], "data":bundle["origin"],
            "valid / requested":f"{native['valid_records']} / {native['requested_records']}",
            "mapping agreement":f"{report['coverage']['mapping_consistent']} / 288",
            "robust seeded Honesty pairs":native["conditions"]["seeded"]["on_target"]["robust_paired_items"],
            "robust seeded Honesty flip rate":native["conditions"]["seeded"]["on_target"]["robust_flip_rate"]}

display(pd.DataFrame([native_overview(key,bundle) for key,bundle in bundles.items()]))
print("A zero robust-flip rate only describes pairs with qualifying, mapping-consistent predictions.")
''')
md('''
## 8. Interact with the results

Choose a model, probe, and context comparison. The left panel shows **mean
absolute movement** on Honesty targets and Fairness/Safety controls. The right
shows standard versus reversed mapping scores for individual persona–item–context
means; points far from the diagonal reveal answer-format sensitivity.

The detail table also shows signed and reference-oriented changes. A larger
absolute shift has no direction by itself. Neither the bars nor the individual
points are independent simulation replicates; no significance test is implied.
''')
code('''
#@title Interactive condition and mapping diagnostics
COMPARISON_LABELS = {"Seeded − neutral":"seeded_minus_neutral",
                     "Quoted − neutral":"mentioned_minus_neutral",
                     "Seeded − quoted":"seeded_minus_mentioned",
                     "Identical repeat − neutral":"repeat_minus_neutral"}

def explore(model, probe, comparison):
    key = model
    values = metrics[key]
    selected = values[(values.probe == probe) & (values.comparison == comparison)]
    fig, axes = plt.subplots(1,2,figsize=(11,3.5),layout="constrained")
    axes[0].bar(["Honesty targets","Fairness / Safety"],
                [selected[selected.group==g].mean_absolute_change.iloc[0] for g in ["target","control"]],
                color=["#16746d","#a86a20"])
    axes[0].set_ylabel("Mean |Δz| · frozen selection SD units")
    axes[0].set_title(comparison.replace("_minus_", " − ").replace("mentioned","quoted"))
    frame = bundles[key]["scores"]
    mapped = frame.groupby(["agent","probe_id","condition","mapping"])[probe+"_z"].mean().unstack("mapping")
    axes[1].scatter(mapped.standard,mapped.reversed,s=20,alpha=.7,color="#16746d")
    low=min(mapped.min()); high=max(mapped.max())
    axes[1].plot([low,high],[low,high],"--",color="#687775",linewidth=1)
    axes[1].set(xlabel="Standard A/B mapping · z",ylabel="Reversed A/B mapping · z",
                title="Answer-format sensitivity (all contexts)")
    gap = mapping_gaps[key].set_index("probe").loc[probe,"mean_absolute_mapping_gap"]
    fig.suptitle(f"{pilot.MODELS[key]['name']} · {probe.upper()} · {MODE}")
    name = f"{key}-{probe}-{comparison}"
    fig.savefig(RESULTS/"analysis"/(name+".png"),dpi=180)
    fig.savefig(RESULTS/"analysis"/(name+".svg"))
    plt.show()
    display(selected[["group","persona_item_pairs","mean_absolute_change","mean_signed_change","reference_signed_change"]])
    print(f"Mean absolute standard/reversed gap across all contexts: {gap:.6f}")
    print("Context effects and mapping gaps summarize different contrasts; both are descriptive.")

controls = widgets.interactive(explore,
    model=widgets.Dropdown(options=[(pilot.MODELS[k]["name"],k) for k in SELECTED],description="Model"),
    probe=widgets.Dropdown(options=[("Difference in means","dim"),("Logistic","logistic")],description="Probe"),
    comparison=widgets.Dropdown(options=list(COMPARISON_LABELS.items()),description="Contrast"))
display(controls)
# A static initial chart is also rendered when widget output is not supported.
explore(SELECTED[0], "dim", "seeded_minus_neutral")
''',('interactive',))
md('''
## 9. Read the original result critically

In the original run, Gemma's 288 native classifications were correct and stable,
while its DIM readout moved. Mean absolute Honesty-target changes were **0.158
seeded / 0.119 quoted** for Gemma and **0.119 / 0.195** for Llama. The corresponding
mapping gaps were **0.124** and **1.304**. These are frozen selection-score SD units.
Llama had mapping agreement on only **144/288** observations. Its reported zero
robust flips does not cover the other half of requested classifications.

Try these three checks with the controls above:

1. Change **seeded − neutral** to **quoted − neutral**. Does a shift require adoption?
2. Compare **target** and **control** bars. Is the response specific to Honesty?
3. Switch **DIM** to **logistic** and inspect the signed reference-oriented column.
   Do the methods support the same directional account?

<details><summary>Suggested interpretation</summary>
The completed pilot establishes context-conditioned readout movement, including
movement with stable Gemma answers. Quotation and control effects, Llama mapping
sensitivity, and probe-method sign disagreement prevent us from treating this as
demonstrated moral alignment change. Identical repeats being zero is not a null
distribution; three personas and repeated easy items do not yield independent
replicates for a contagion claim.
</details>

**Next research step:** keep these anchors, develop distinct moral-conflict and
reasoning questions, evaluate on held-out items/paraphrases, and test persistence
after exposure ends. A subsequent randomized seeded-versus-control social study
would measure initially unseeded recipients across independent runs. We have not
executed that study here. See [the proposed protocol](https://github.com/stevenokada/multiagent/blob/571bbb980903e9a5aca219007fa099edba1e2c11/docs/papers/moral-relation-pilot/draft-v2/body.tex).
''')
md('''
## 10. Export and finish

The ZIP contains this session's analysis and, in live mode, the numerical
comparison, prompts/calls, all captured vectors, frozen scores, and provenance.
Replay exports the saved evidence it used and the newly recomputed plots/tables;
it does not manufacture raw activations. Each exported file has a SHA-256 digest.

Model caches, credentials, environments, and symlinks are excluded. Export before
disconnecting: Colab runtime files are ephemeral. After a live run, use **Runtime
→ Disconnect and delete runtime** when you are done; finishing a notebook cell is
not a command to release compute.
''')
code('''
#@title Download the results bundle
from importlib.metadata import version as package_version
session = {"mode":MODE,"models":SELECTED,"created_utc":SESSION,
           "project_revision":PROJECT_REVISION,"jeff_revision":JEFF_REVISION,
           "new_checkpoint_inference":MODE=="live","contagion_established":False,
           "packages":{p:package_version(p) for p in ["numpy","pandas","matplotlib"]}}
pilot.write_json(RESULTS/"session.json",session)
(RESULTS/"README.txt").write_text(
    f"Moral relation tutorial: {MODE}. New model inference: {MODE=='live'}.\\n"
    "Independent context calibration; no agent interaction or established contagion.\\n"
    "Read session.json and live/*/run-manifest.json for provenance.\\n")
export_stamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
archive = pilot.export_bundle(RESULTS, RESULTS.parent/(SESSION+"-"+export_stamp+".zip"))
print("Saved",archive,"—",archive.stat().st_size,"bytes")
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    from IPython.display import FileLink
    display(FileLink(str(archive)))
''',('export',))
md('''
## Sources and further reading

- [Project results and numerical limits](https://github.com/stevenokada/multiagent/blob/571bbb980903e9a5aca219007fa099edba1e2c11/docs/probe-context-pilot-2026-09-06.md).
- [Jeff's frozen probe artifacts and scoring contract](https://github.com/JeffVallyath/geometry-of-endorsement/blob/ff3588c1d15ac89c6a3edd5da41c6fa121ad4157/artifacts/probe_weights/README.md).
- Sorensen et al. (2024), [Value Kaleidoscope](https://arxiv.org/abs/2309.00779): pluralistic values and the ValuePrism resource. The current eight items are authored anchors.
- Hewitt & Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/): why probe performance needs controls.
- Marks & Tegmark (2024), [The Geometry of Truth](https://arxiv.org/abs/2310.06824): linear structure in a different semantic domain; not validation of our moral construct.
- Chen et al. (2025), [Persona Vectors](https://arxiv.org/abs/2507.21509): related monitoring and intervention ideas, not results from this experiment.

Prepared 9 September 2026. This notebook is a learning artifact for the completed
pilot. Local validation executes the CPU path and stub-based live-boundary tests;
a successful local check is not evidence of a new GPU rerun in Colab.
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{
    'colab':{'name':'moral_relation_pilot.ipynb','provenance':[]},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},
    'moral_pilot':{'source_revision':args.revision,'default_mode':'replay','new_gpu_validation':False}},
    'cells':cells}
path=Path(__file__).with_name('moral_relation_pilot.ipynb')
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
print(f'Wrote {len(cells)} tutorial cells to {path}')
