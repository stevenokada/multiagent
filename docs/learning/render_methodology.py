#!/usr/bin/env python3
"""Build the vector figure and verify the offline interactive companion.

Requires Playwright with Chromium and PyMuPDF. No network requests are made.
Example:
  /tmp/snaggletooth-bugfix-venv/bin/python docs/learning/render_methodology.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WIDTH, HEIGHT = 1380, 1110
INK, MUTED = "#18333d", "#4c646c"
TEAL, PALE, LINE = "#087e80", "#e9f5f2", "#cbdad9"
GOLD, GOLD_PALE = "#a36c12", "#fff5dd"


def build_svg() -> str:
    pieces = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">Completed context pilot: one prompt, two measurement channels</title>
<desc id="desc">Three personas, three repeats, eight authored anchors and four journal contexts produce 288 observations and 576 A/B forward evaluations per model. A pinned BF16 model produces native A/B probabilities and a hidden vector at the final real prompt token. Frozen probes score the saved vector offline. Matched comparisons inspect targets, controls, quotation and answer mapping. Gemma uses HF index 28, block 27, width 3584 and batch 2 after passing the gate. Llama uses HF index 20, block 19, width 4096 and batch 1 after the batch-two gate failed. Models ran sequentially on A6000 GPUs. A separate proposed future study would use a seeded agent, shared board and recipient journals. No agents interacted in the completed pilot.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{TEAL}"/></marker></defs>
<style>text{{font-family:Arial,Helvetica,sans-serif;fill:{INK}}}.muted{{fill:{MUTED}}}.teal{{fill:#076367}}.gold{{fill:#79520e}}.eyebrow{{font-size:12px;font-weight:700;letter-spacing:1.2px}}.title{{font-size:25px;font-weight:700}}.body{{font-size:17px}}.small{{font-size:15px}}.formula{{font-family:DejaVu Sans Mono,monospace;font-size:17px}}</style>
<rect width="1380" height="1110" fill="#f6f8f5"/>''']

    def rect(x, y, w, h, fill="white", stroke=LINE, radius=15, extra=""):
        pieces.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" {extra}/>')

    def text(x, y, value, size=None, cls=None, weight=None):
        attrs = f' class="{cls}"' if cls else ""
        attrs += f' font-size="{size}"' if size else ""
        attrs += f' font-weight="{weight}"' if weight else ""
        pieces.append(f'<text x="{x}" y="{y}"{attrs}>{html.escape(value)}</text>')

    def path(d, arrow=True, color=TEAL, width=2):
        pieces.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"' + (' marker-end="url(#arrow)"' if arrow else '') + '/>')

    text(44, 45, "METHODS COMPANION · COMPLETED 6 SEPTEMBER 2026", cls="eyebrow teal")
    text(44, 96, "One prompt. Two measurement channels.", 38, weight=700)
    text(44, 132, "Independent journal-conditioning calibration · fixed model weights · no agent interaction", 19, cls="muted")

    rect(44, 161, 1292, 143)
    text(66, 189, "01 · MATCHED DESIGN · PER MODEL · 8 AUTHORED ANCHORS (4 HONESTY, 2 FAIRNESS, 2 SAFETY)", cls="eyebrow teal")
    text(66, 230, "3 personas × 3 repeats × 8 anchors × 4 contexts", 27, weight=700)
    text(850, 230, "= 288 observations", 27, cls="teal", weight=700)
    path("M 66 249 H 1314", False, "#dce6e3", 1)
    text(66, 278, "Neutral · Identical repeat · Seeded conviction · Quoted without adoption", 17, cls="muted")
    text(888, 278, "Both A/B mappings → 576 calls/model", 17, cls="teal", weight=700)

    text(44, 343, "EACH MAPPING IS ONE FORWARD EVALUATION", cls="eyebrow muted")
    rect(44, 385, 246, 297)
    text(64, 413, "02 · RENDER PROMPT", cls="eyebrow teal")
    text(64, 454, "Persona + journal", 20, weight=700)
    text(64, 488, "Situation +", 21)
    text(64, 517, "named consideration", 19)
    path("M 64 538 H 270", False, "#dce6e3", 1)
    text(64, 567, "Pinned chat template", cls="body muted")
    text(64, 596, "A/B and reversed B/A", cls="body muted")
    text(64, 641, "Relation: supports / opposes", 15, cls="muted")

    rect(332, 385, 225, 297, PALE, "#a8cfca")
    text(352, 413, "03 · FIXED MODEL", cls="eyebrow teal")
    text(352, 453, "One forward", cls="title")
    text(352, 484, "pass", cls="title")
    for offset, fill in [(0, "#baded5"), (9, "#91c8bd"), (18, "#63aaa0"), (27, "#087e80")]:
        rect(369, 511 + offset, 144, 25, fill, "none", 5)
    text(352, 604, "Next-token logits +", cls="body teal")
    text(352, 633, "hidden states", cls="body teal")
    text(352, 661, "No generated answer text", 14, cls="muted")

    rect(613, 367, 334, 120)
    text(635, 395, "04A · NATIVE ANSWER", cls="eyebrow teal")
    text(635, 430, "A/B softmax", 23, weight=700)
    text(635, 462, "P(Supports | A or B) + classification", 16, cls="muted")

    rect(613, 506, 334, 176)
    text(635, 534, "04B · FROZEN PROBE", cls="eyebrow teal")
    text(635, 567, "Final real prompt token → h", 18, weight=700)
    text(635, 595, "Save float16 → score offline float32", 16, cls="muted")
    text(635, 631, "raw = h · d − midpoint", cls="formula teal")
    text(635, 660, "z = (raw − μ) / σ", cls="formula teal")

    rect(1015, 385, 321, 297)
    text(1035, 413, "05 · JOIN + COMPARE", cls="eyebrow teal")
    text(1035, 453, "Matched comparisons", 23, weight=700)
    text(1035, 485, "call_id + trial / item / mapping", 16, cls="muted")
    path("M 1035 504 H 1316", False, "#dce6e3", 1)
    text(1035, 534, "Average both mappings", cls="body")
    text(1035, 563, "Collapse identical repeats", cls="body")
    text(1035, 602, "Targets · controls · quotation", cls="body teal")
    text(1035, 631, "Retain A/B gaps", cls="body teal")
    text(1035, 660, "Compare DIM + logistic", cls="body teal")

    path("M 291 490 H 328")
    path("M 558 490 H 584 V 428 H 609")
    path("M 584 490 V 585 H 609")
    path("M 948 428 H 1011")
    path("M 948 585 H 1011")
    text(613, 711, "Frozen weights + scalers; semantic Supports direction is unchanged by an A/B swap.", 15, cls="muted")

    rect(44, 736, 1292, 142, "#edf2ef")
    text(66, 763, "EXTRACTION + NUMERICAL GATE", cls="eyebrow teal")
    text(524, 763, "Pinned revisions · BF16 · sequential A6000 (48 GB) runs", 16, cls="muted")
    text(66, 800, "Gemma 2 9B IT", 18, weight=700)
    text(327, 800, "HF 28 = block 27", cls="body")
    text(608, 800, "3,584 dimensions", cls="body")
    text(904, 800, "Batch-two gate: pass → batch 2", 17, cls="teal", weight=700)
    text(66, 832, "Llama 3.1 8B Instruct", 18, weight=700)
    text(327, 832, "HF 20 = block 19", cls="body")
    text(608, 832, "4,096 dimensions", cls="body")
    text(904, 832, "Batch-two gate: fail → batch 1", 17, cls="gold", weight=700)
    text(66, 862, "48 numerical-check requests/model. The parallel two-GPU launcher was not used for the completed calibrations.", 15, cls="muted")

    rect(44, 904, 1292, 121, GOLD_PALE, "#c4a164", 15, 'stroke-dasharray="6 5"')
    text(66, 933, "PROPOSED FUTURE STUDY · NOT RUN IN THIS PILOT", cls="eyebrow gold")
    text(66, 973, "Seeded agent  →  Shared board  →  Recipient journal  →  Same battery", 25, weight=700)
    text(66, 1005, "Validate measurement first; then matched seeded/unseeded recipients, pre-seeding baseline and persistence checks.", 16, cls="gold")
    text(44, 1060, "Supported: context-sensitive readouts. Moral adoption, persistence and contagion have not been established.", 17, cls="muted")
    text(44, 1088, "Source: docs/probe-context-pilot-2026-09-06.md · docs/papers/moral-relation-pilot/draft-v2/body.tex", 12, cls="muted")
    pieces.append("</svg>")
    return "\n".join(pieces)


def verify_and_render(chromium: str | None) -> dict:
    checks = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "viewports": [], "browser_errors": [], "external_requests": []}
    svg = build_svg()
    ET.fromstring(svg)
    (HERE / "methodology.svg").write_text(svg, encoding="utf-8")
    assert 3 * 3 * 8 * 4 == 288 and 288 * 2 == 576
    with sync_playwright() as p:
        launch = {"headless": True, "args": ["--no-sandbox"]}
        if chromium:
            launch["executable_path"] = chromium
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
        page.set_content(f'<!doctype html><html><head><style>@page{{size:{WIDTH}px {HEIGHT}px;margin:0}}html,body{{margin:0;width:{WIDTH}px;height:{HEIGHT}px}}svg{{display:block}}</style></head><body>{svg}</body></html>')
        page.evaluate("document.fonts.ready")
        boxes = page.locator("svg text").evaluate_all("nodes => nodes.map(n => {const b=n.getBBox();return {text:n.textContent,x:b.x,y:b.y,width:b.width,height:b.height}})")
        clipping = [box for box in boxes if box["x"] < 0 or box["y"] < 0 or box["x"] + box["width"] > WIDTH or box["y"] + box["height"] > HEIGHT]
        assert not clipping, f"SVG clipping: {clipping}"
        # Ensure text stays inside its card as well as the outer viewBox.
        cards = [(44,161,1292,143),(44,385,246,297),(332,385,225,297),(613,367,334,120),(613,506,334,176),(1015,385,321,297),(44,736,1292,142),(44,904,1292,121)]
        card_overflow = []
        for box in boxes:
            for x,y,w,h in cards:
                if x < box["x"] < x+w and y < box["y"] < y+h:
                    if box["x"]+box["width"] > x+w-10 or box["y"]+box["height"] > y+h-3:
                        card_overflow.append(box["text"])
        assert not card_overflow, f"SVG card overflow: {card_overflow}"
        page.screenshot(path=str(HERE / "methodology.png"), full_page=True, scale="css")
        page.pdf(path=str(HERE / "methodology.pdf"), print_background=True, prefer_css_page_size=True)
        checks["svg"] = {"text_elements": len(boxes), "viewbox": [WIDTH,HEIGHT], "clipped_labels": clipping, "card_overflow": card_overflow}
        page.close()
        for width,height in [(1440,1000),(768,1024),(390,844),(320,740)]:
            page = browser.new_page(viewport={"width":width,"height":height})
            page.emulate_media(reduced_motion="reduce")
            page.on("pageerror",lambda exc:checks["browser_errors"].append(str(exc)))
            page.on("request",lambda req:checks["external_requests"].append(req.url) if req.url.startswith(("http:","https:")) else None)
            page.goto((HERE / "methodology.html").as_uri())
            assert page.locator('button[role="tab"]').count() == 5
            for stage in ("inputs","prompt","forward","readouts","compare"):
                page.locator(f"#tab-{stage}").click()
                assert page.locator(f"#tab-{stage}").get_attribute("aria-selected") == "true"
                assert page.locator("#stage-panel").get_attribute("aria-labelledby") == f"tab-{stage}"
                assert page.locator("#stage-panel h3").inner_text()
            page.get_by_label("Llama 3.1 8B",exact=True).check()
            assert "failed" in page.locator(".gate").inner_text()
            assert "144/288" in page.locator("#stage-panel").inner_text()
            page.locator("#tab-forward").click()
            assert "4,096" in page.locator("#stage-panel").inner_text()
            assert "index 20" in page.locator("#stage-panel").inner_text()
            page.get_by_label("Gemma 2 9B",exact=True).check()
            assert "passed" in page.locator(".gate").inner_text()
            assert "3,584" in page.locator("#stage-panel").inner_text()
            page.locator("#tab-inputs").focus()
            page.keyboard.press("ArrowRight")
            assert page.evaluate("document.activeElement.id") == "tab-prompt"
            page.keyboard.press("End")
            assert page.evaluate("document.activeElement.id") == "tab-compare"
            page.keyboard.press("ArrowRight")
            assert page.evaluate("document.activeElement.id") == "tab-inputs"
            page.keyboard.press("ArrowLeft")
            assert page.evaluate("document.activeElement.id") == "tab-compare"
            page.keyboard.press("Home")
            assert page.evaluate("document.activeElement.id") == "tab-inputs"
            page.locator(".future summary").click()
            assert page.locator(".future details").get_attribute("open") is not None
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Horizontal overflow at {width}px"
            page.locator(".future summary").click()
            page.locator("#tab-readouts").click()
            assert "never negated" in page.locator("#stage-panel").inner_text()
            if width in (1440,390):
                page.screenshot(path=str(Path(tempfile.gettempdir()) / f"methodology-{width}.png"), full_page=True)
            checks["viewports"].append({"width":width,"height":height,"five_stages":True,"model_toggle":True,"keyboard_arrows_home_end":True,"future_disclosure":True,"horizontal_overflow":False})
            page.close()
        browser.close()
    assert not checks["browser_errors"]
    assert not checks["external_requests"]
    with pymupdf.open(HERE / "methodology.pdf") as pdf:
        assert len(pdf) == 1
        pdf_text = pdf[0].get_text()
        for term in ("288 observations", "576 calls/model", "HF 28", "HF 20", "PROPOSED FUTURE STUDY", "Final real prompt token"):
            assert term in pdf_text, f"Missing PDF text: {term}"
        checks["pdf"] = {"pages":len(pdf),"searchable_text":True,"page_size_points":list(pdf[0].rect)[2:]}
    sources = [ROOT / "docs/probe-context-pilot-2026-09-06.md", ROOT / "docs/papers/moral-relation-pilot/draft-v2/body.tex", ROOT / "audit/results/gemma-calibration-2026-09-06.json", ROOT / "audit/results/llama-calibration-2026-09-06.json"]
    checks["sources"] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    checks["artifacts"] = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("methodology.html","methodology.svg","methodology.pdf","methodology.png")}
    checks["visual_inspection"] = "Inspect methodology.png and /tmp/methodology-{1440,390}.png after rendering."
    checks["passed"] = True
    (HERE / "methodology-checks.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium", default="/home/orca/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
    args = parser.parse_args()
    result = verify_and_render(args.chromium)
    print(json.dumps({"passed":result["passed"],"viewports":[v["width"] for v in result["viewports"]],"pdf_pages":result["pdf"]["pages"],"svg_labels":result["svg"]["text_elements"]},indent=2))
