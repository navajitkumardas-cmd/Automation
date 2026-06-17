"""Walk every recipe in the kitchen console, validate required fields, and
normalise units / text formatting. Writes a JSON report of issues found and,
when `apply_changes` is enabled, saves corrections back through the web UI.

Usage:
    pip install -r requirements.txt
    playwright install chromium
    export RECIPE_USER=...           # your console username
    export RECIPE_PASS=...           # your console password
    python check_recipes.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from playwright.sync_api import Page, TimeoutError as PWTimeout, sync_playwright


@dataclass
class RecipeIssue:
    recipe_url: str
    recipe_name: str
    field: str
    kind: str        # "missing" | "normalise"
    before: Any = None
    after: Any = None
    applied: bool = False


@dataclass
class Report:
    checked: int = 0
    issues: list[RecipeIssue] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "checked": self.checked,
            "issue_count": len(self.issues),
            "issues": [i.__dict__ for i in self.issues],
        }


# ---------- normalisation helpers ----------

_WS_RE = re.compile(r"\s+")


def normalise_unit(raw: str | None, aliases: dict[str, str]) -> str | None:
    if raw is None:
        return None
    key = raw.strip().lower().rstrip(".")
    if not key:
        return raw
    return aliases.get(key, raw.strip())


def normalise_text(value: str | None, rules: dict, field_name: str) -> str | None:
    if value is None:
        return None
    out = value
    if field_name in rules.get("trim_fields", []):
        out = out.strip()
    if field_name in rules.get("collapse_whitespace_fields", []):
        out = _WS_RE.sub(" ", out)
    if field_name in rules.get("title_case_fields", []):
        out = out.title()
    return out


# ---------- page-level operations ----------

def login(page: Page, cfg: dict) -> None:
    user = os.environ.get("RECIPE_USER")
    pw = os.environ.get("RECIPE_PASS")
    if not user or not pw:
        sys.exit("ERROR: set RECIPE_USER and RECIPE_PASS environment variables.")
    page.goto(cfg["site"]["login_url"])
    page.fill(cfg["auth"]["username_selector"], user)
    page.fill(cfg["auth"]["password_selector"], pw)
    page.click(cfg["auth"]["submit_selector"])
    page.wait_for_selector(cfg["auth"]["logged_in_marker_selector"], timeout=30_000)


def iter_recipe_urls(page: Page, cfg: dict, limit: int):
    """Yield the URL of every recipe's edit page, walking pagination."""
    page.goto(cfg["site"]["recipe_list_url"])
    page.wait_for_selector(cfg["list_page"]["row_selector"], timeout=30_000)
    seen = 0
    while True:
        rows = page.query_selector_all(cfg["list_page"]["row_selector"])
        for row in rows:
            link = row.query_selector(cfg["list_page"]["open_selector"])
            if not link:
                continue
            href = link.get_attribute("href")
            if href:
                yield page.url_for(href) if hasattr(page, "url_for") else _absolute(page, href)
                seen += 1
                if limit and seen >= limit:
                    return
        next_btn = page.query_selector(cfg["site"]["next_page_selector"])
        if not next_btn or not next_btn.is_enabled():
            return
        next_btn.click()
        page.wait_for_load_state("networkidle")


def _absolute(page: Page, href: str) -> str:
    if href.startswith("http"):
        return href
    from urllib.parse import urljoin
    return urljoin(page.url, href)


def read_field(page: Page, selector: str) -> str | None:
    el = page.query_selector(selector)
    if not el:
        return None
    tag = (el.evaluate("e => e.tagName") or "").lower()
    if tag == "select":
        return el.evaluate("e => e.value")
    return el.input_value() if tag in {"input", "textarea"} else el.inner_text()


def write_field(page: Page, selector: str, value: str) -> None:
    el = page.query_selector(selector)
    if not el:
        return
    tag = (el.evaluate("e => e.tagName") or "").lower()
    if tag == "select":
        el.select_option(value=value)
    else:
        el.fill(value)


def process_recipe(page: Page, url: str, cfg: dict, report: Report) -> None:
    page.goto(url)
    page.wait_for_selector(cfg["detail_page"]["ready_selector"], timeout=30_000)

    fields_cfg = cfg["detail_page"]["fields"]
    name_value = read_field(page, fields_cfg.get("name", "")) or "<unknown>"
    changes_to_apply: list[tuple[str, str]] = []

    # 1. Required-field validation + text normalisation on top-level fields.
    for field_name, selector in fields_cfg.items():
        raw = read_field(page, selector)
        if field_name in cfg["required_fields"]["recipe"] and not (raw and raw.strip()):
            report.issues.append(RecipeIssue(url, name_value, field_name, "missing", before=raw))
            continue
        normalised = normalise_text(raw, cfg["text_normalisation"], field_name)
        if normalised is not None and normalised != raw:
            report.issues.append(
                RecipeIssue(url, name_value, field_name, "normalise", before=raw, after=normalised)
            )
            changes_to_apply.append((selector, normalised))

    # 2. Ingredient rows: required fields + unit normalisation.
    ing_rows = page.query_selector_all(cfg["detail_page"]["ingredient_row_selector"])
    ing_fields = cfg["detail_page"]["ingredient_fields"]
    for idx, row in enumerate(ing_rows):
        for sub in cfg["required_fields"]["ingredient"]:
            sel = ing_fields.get(sub)
            if not sel:
                continue
            cell = row.query_selector(sel)
            val = cell.input_value() if cell else None
            if not (val and val.strip()):
                report.issues.append(
                    RecipeIssue(url, name_value, f"ingredient[{idx}].{sub}", "missing", before=val)
                )
        unit_sel = ing_fields.get("unit")
        if unit_sel:
            cell = row.query_selector(unit_sel)
            if cell:
                current = cell.input_value()
                fixed = normalise_unit(current, cfg["unit_aliases"])
                if fixed and fixed != current:
                    report.issues.append(
                        RecipeIssue(
                            url, name_value, f"ingredient[{idx}].unit",
                            "normalise", before=current, after=fixed,
                        )
                    )
                    # Fill via the cell element directly so we hit the right row.
                    if cfg["runtime"]["apply_changes"]:
                        cell.fill(fixed)

    # 3. Step count check.
    steps = page.query_selector_all(cfg["detail_page"]["steps_selector"])
    if len(steps) < cfg["required_fields"].get("min_steps", 1):
        report.issues.append(
            RecipeIssue(url, name_value, "steps", "missing", before=len(steps))
        )

    # 4. Apply queued top-level changes and save.
    if cfg["runtime"]["apply_changes"] and (changes_to_apply or _has_unit_fixes(report, url)):
        for sel, val in changes_to_apply:
            write_field(page, sel, val)
        page.click(cfg["detail_page"]["save_selector"])
        try:
            page.wait_for_selector(
                cfg["detail_page"]["save_confirmation_selector"], timeout=15_000
            )
            for issue in report.issues:
                if issue.recipe_url == url and issue.kind == "normalise":
                    issue.applied = True
        except PWTimeout:
            print(f"  WARNING: save confirmation never appeared for {url}")

    report.checked += 1


def _has_unit_fixes(report: Report, url: str) -> bool:
    return any(
        i.recipe_url == url and i.kind == "normalise" and i.field.startswith("ingredient")
        for i in report.issues
    )


# ---------- entry point ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"ERROR: config file {cfg_path} not found. Copy config.example.yaml.")
    cfg = yaml.safe_load(cfg_path.read_text())

    runtime = cfg["runtime"]
    report = Report()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=runtime["headless"], slow_mo=runtime.get("slow_mo_ms", 0)
        )
        context = browser.new_context()
        page = context.new_page()

        login(page, cfg)

        urls = list(iter_recipe_urls(page, cfg, runtime["max_recipes"]))
        print(f"Found {len(urls)} recipes. apply_changes={runtime['apply_changes']}")

        for url in urls:
            try:
                process_recipe(page, url, cfg, report)
            except Exception as exc:  # one bad recipe shouldn't stop the run
                print(f"  ERROR processing {url}: {exc}")

        browser.close()

    Path(runtime["report_path"]).write_text(json.dumps(report.to_json(), indent=2))
    print(f"\nChecked {report.checked} recipes, {len(report.issues)} issues -> {runtime['report_path']}")


if __name__ == "__main__":
    main()
