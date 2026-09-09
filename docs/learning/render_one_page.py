#!/usr/bin/env python3
"""Render the editable brief as standalone HTML and exactly one A4 PDF page.

Dependencies: markdown-it-py, playwright, pymupdf; a Chromium installation.
Run: python docs/learning/render_one_page.py [--chromium /path/to/chrome]
The renderer validates the published numbers against the frozen local evidence,
checks DOM/PDF bounds, and writes one-page-checks.json. It does not run models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "papers/moral-relation-pilot/draft-v2/evidence"
CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
html { color: #20313a; background: #edf0f1; }
body { margin: 0; font: 10pt/1.30 'DejaVu Serif', Georgia, serif; }
main {
  width: 210mm; min-height: 297mm; margin: 18px auto; padding: 12mm 13mm 11mm;
  background: white; box-shadow: 0 3px 24px #102d3c18;
}
h1, h2, table, main > p:first-of-type, footer {
  font-family: 'DejaVu Sans', Arial, sans-serif;
}
h1 { font-size: 24pt; font-weight: 700; line-height: 1.16;
  color: #173f50; letter-spacing: -.6pt; margin: 0 0 5pt; }
main > p:first-of-type { font-size: 8.4pt; color: #5a6b74; margin: 0 0 12pt; }
p { margin: 0 0 6pt; }
h2 { margin: 9pt 0 6pt; font-size: 10pt; line-height: 1.3; color: #173f50;
  text-transform: uppercase; letter-spacing: 1.15pt; }
blockquote { margin: 0 0 10pt; padding: 7pt 11pt; background: #edf5f6;
  border-left: 3pt solid #3c7d8b; font-size: 10.5pt; line-height: 1.4; }
blockquote p { margin: 0; }
strong { font-weight: 700; }
a { color: #175e78; text-decoration: underline; text-decoration-thickness: .45pt;
  text-underline-offset: 1.3pt; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0 9pt;
  font-size: 8.4pt; line-height: 1.35; font-variant-numeric: tabular-nums; }
th { background: #edf2f4; color: #183e50; font-weight: 700; }
th, td { padding: 3.8pt 5pt; border-bottom: .5pt solid #d4dfe3; vertical-align: top; }
th:first-child, td:first-child { padding-left: 6pt; }
table:first-of-type { font-size: 8pt; }
table:first-of-type th, table:first-of-type td { padding: 3.8pt 5pt; }
table:first-of-type td:nth-child(2) { white-space: nowrap; }
code { font: 7.25pt/1.4 'DejaVu Sans Mono', monospace; color: #304451; }
hr { height: 0; border: 0; border-top: .7pt solid #b7c8d0; margin: 10pt 0 7pt; }
main > p:last-child { font: 7.5pt/1.4 'DejaVu Sans', Arial, sans-serif;
  color: #5a6b74; margin: 0; }
main > p:last-child a { text-decoration: none; }
@media print {
  html, body { background: white; }
  main { margin: 0; box-shadow: none; }
  * { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
}
@media screen and (max-width: 820px) {
  main { width: 100%; min-height: auto; margin: 0; padding: 24px 18px; }
  h1 { font-size: 26px; }
  table { font-size: 10px; }
  table:first-of-type { font-size: 10px; }
  table:first-of-type td:first-child { min-width: 80px; }
  code { font-size: 9px; overflow-wrap: anywhere; }
  th, td { padding: 6px 3px; }
}
"""


def frozen_checks(markdown: str) -> dict:
    """Check every displayed numeric result against primary saved evidence."""
    models = {}
    for model in ("gemma", "llama"):
        probe_file = EVIDENCE / f"{model}-frozen-probe-context-2026-09-06.json"
        native_file = EVIDENCE / f"{model}-calibration-2026-09-06.json"
        probe = json.loads(probe_file.read_text())
        native = json.loads(native_file.read_text())
        revision = probe["model"]["revision"]
        assert revision in markdown, f"Missing revision: {model}"
        assert probe["calls"] == native["coverage"]["calls"] == 576
        assert native["coverage"]["observations"] == 288
        assert native["verification"]["all_vectors_finite"] is True
        assert f'**{native["coverage"]["mapping_consistent"]}/288**' in markdown
        assert probe["model"]["dtype"] == "bfloat16"
        assert probe["model"]["quantize_4bit"] is False
        seeded = probe["changes"]["seeded_minus_neutral"]["target"]
        assert seeded["dim"]["mean_reference_signed_z_change"] < 0
        assert seeded["logistic"]["mean_reference_signed_z_change"] > 0
        if model == "gemma":
            assert native["verification"]["reference_classifications_correct"] == 288
            assert native["numerical_check"]["passed"] is True
        else:
            assert native["status"]["batch_size"] == 1
            assert native["numerical_check"]["passed"] is False
        values = {}
        for contrast in ("repeat_minus_neutral", "seeded_minus_neutral", "mentioned_minus_neutral"):
            values[contrast] = {
                group: probe["changes"][contrast][group]["dim"]["mean_absolute_z_change"]
                for group in ("target", "control")
            }
        gap = probe["mapping_sensitivity"]["dim"]["mean_absolute_z_gap"]
        label = "Gemma" if model == "gemma" else "Llama"
        assert f"**{gap:.3f} ({label})**" in markdown
        models[model] = {
            "revision": revision,
            "hf_index": probe["hf_layer"],
            "width": 3584 if model == "gemma" else 4096,
            "vectors": probe["calls"],
            "native_mapping_agreement": native["coverage"]["mapping_consistent"],
            "dim_mean_absolute_changes": values,
            "dim_mean_absolute_mapping_gap": gap,
            "evidence_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (probe_file, native_file)
            },
        }
    for label, key in (
        ("Identical repeat − neutral", "repeat_minus_neutral"),
        ("Seeded − neutral", "seeded_minus_neutral"),
        ("Quoted − neutral", "mentioned_minus_neutral"),
    ):
        numbers = [
            models[model]["dim_mean_absolute_changes"][key][group]
            for model in ("gemma", "llama") for group in ("target", "control")
        ]
        assert f'| {label} | ' + " | ".join(f"{v:.3f}" for v in numbers) + " |" in markdown
    assert models["gemma"]["hf_index"] == 28
    assert models["llama"]["hf_index"] == 20
    return models


def browser_path(explicit: str | None) -> str | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    for name in ("chromium", "chromium-browser", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    cached = sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux*/chrome"))
    return str(cached[-1]) if cached else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium", help="Optional path to the Chromium executable")
    parser.add_argument("--preview", type=Path, help="Optional PNG preview path, outside this artifact set")
    args = parser.parse_args()
    markdown = (HERE / "one-page.md").read_text()
    models = frozen_checks(markdown)
    content = MarkdownIt("commonmark").enable("table").render(markdown)
    html = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="description" content="One-page brief on the completed frozen moral-relation probe calibration.">'
        '<title>When context moves a probe — completed pilot brief</title>'
        f"<style>{CSS}</style></head><body><main>{content}</main></body></html>\n"
    )
    html_path = HERE / "one-page.html"
    pdf_path = HERE / "one-page.pdf"
    html_path.write_text(html)
    with sync_playwright() as playwright:
        executable = browser_path(args.chromium)
        launch = {"headless": True}
        if executable:
            launch["executable_path"] = executable
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1280, "height": 1400})
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.emulate_media(media="print")
        dom = page.evaluate("""() => {
          const main = document.querySelector('main');
          const box = main.getBoundingClientRect();
          const style = getComputedStyle(main);
          const bottomLimit = 297 / 25.4 * 96 - parseFloat(style.paddingBottom);
          const overflow = Array.from(main.querySelectorAll('*')).filter(el => {
            const r = el.getBoundingClientRect();
            return r.left < box.left - .5 || r.right > box.right + .5 ||
              r.top < box.top - .5 || r.bottom > box.top + bottomLimit + .5;
          }).map(el => ({tag: el.tagName, text: el.textContent.slice(0, 90)}));
          return {
            width: box.width, height: box.height,
            contentBottom: main.lastElementChild.getBoundingClientRect().bottom - box.top,
            bottomLimit, overflow,
            visibleText: main.innerText,
            externalAssets: [...document.querySelectorAll('script[src], img[src], link[rel=stylesheet]')].length
          };
        }""")
        assert not dom["overflow"], f'Content exceeds page bounds: {dom["overflow"]}'
        assert dom["externalAssets"] == 0
        page.pdf(path=str(pdf_path), prefer_css_page_size=True, print_background=True,
                 display_header_footer=False, tagged=True)
        page.emulate_media(media="screen")
        page.set_viewport_size({"width": 390, "height": 844})
        mobile = page.evaluate("""() => ({
          viewport: innerWidth, documentWidth: document.documentElement.scrollWidth,
          overflowing: document.documentElement.scrollWidth > innerWidth
        })""")
        assert not mobile["overflowing"], f"Mobile horizontal overflow: {mobile}"
        browser.close()
    document = pymupdf.open(pdf_path)
    assert len(document) == 1, f"Expected exactly one page; got {len(document)}"
    pdf_page = document[0]
    # Chromium resolves local hrefs to this machine's absolute workspace paths.
    # Keep companion links usable when the learning folder is moved or shared.
    relative_links = []
    for link in pdf_page.get_links():
        if link["kind"] == pymupdf.LINK_LAUNCH and Path(link["file"]).is_absolute():
            link["file"] = os.path.relpath(link["file"], HERE)
            relative_links.append(link["file"])
            pdf_page.update_link(link)
    if relative_links:
        pdf_path.write_bytes(document.tobytes(garbage=4, deflate=True))
        document.close()
        document = pymupdf.open(pdf_path)
        pdf_page = document[0]
    assert not any(
        link["kind"] == pymupdf.LINK_LAUNCH and Path(link["file"]).is_absolute()
        for link in pdf_page.get_links()
    ), "Absolute workspace path leaked into a PDF link"
    text = pdf_page.get_text()
    assert "2026-09-09" in text, "Footer missing from PDF"
    assert "Matched social simulations" in text, "Final content missing from PDF"
    outside = [
        list(word) for word in pdf_page.get_text("words")
        if not pdf_page.rect.contains(pymupdf.Rect(word[:4]))
    ]
    assert not outside, f"Text outside the PDF page: {outside}"
    visible_words = re.findall(r"\S+", dom.pop("visibleText"))
    assert len(visible_words) <= 500, f"Brief exceeds 500 words: {len(visible_words)}"
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        pdf_page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6)).save(args.preview)
    checks = {
        "artifact": "one-page.pdf",
        "brief_date": "2026-09-09",
        "status": "passed",
        "pdf_pages": len(document),
        "pdf_page_points": list(pdf_page.rect),
        "visible_word_count": len(visible_words),
        "pdf_text_word_count": len(pdf_page.get_text("words")),
        "pdf_text_outside_page": outside,
        "pdf_relative_companion_links": relative_links,
        "dom_print_bounds": dom,
        "mobile_layout": mobile,
        "numerical_evidence": models,
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HERE / "one-page.md", html_path, pdf_path, Path(__file__))
        },
    }
    (HERE / "one-page-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({k: checks[k] for k in (
        "status", "pdf_pages", "visible_word_count", "pdf_text_outside_page", "mobile_layout"
    )}, indent=2))


if __name__ == "__main__":
    main()
