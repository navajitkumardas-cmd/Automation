# Recipe checker

Walks every recipe in the Resoee kitchen console, flags missing required
fields, and standardises unit / text formatting. Defaults to a safe
**dry-run** that only writes a JSON report.

## Setup

```bash
cd recipe_checker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp config.example.yaml config.yaml
```

Open `config.yaml` and replace the selectors under `auth`, `list_page`, and
`detail_page` with ones that match the actual DOM of the console. Open the
recipe list in Chrome DevTools, inspect the elements, and paste the CSS
selectors in. Everything else (required-field list, unit aliases, text
rules) can stay as-is for a first run.

## Run

```bash
export RECIPE_USER="you@example.com"
export RECIPE_PASS="•••••"

# Dry run — produces report.json, changes nothing.
python check_recipes.py --config config.yaml

# Inspect report.json. When you trust the output, flip:
#   runtime.apply_changes: true
# in config.yaml and re-run to actually save the fixes.
```

Set `runtime.headless: false` and `runtime.slow_mo_ms: 200` while you are
still tuning selectors so you can watch what the browser is doing.

## What it checks

- **Required fields** on each recipe (name, category, serving size, prep /
  cook time) and on each ingredient row (name, quantity, unit).
- **Minimum step count** (default ≥ 1).
- **Unit standardisation** — converts `grams`, `gm`, `Gram` etc. to the
  canonical `g`; same for `kg`, `ml`, `l`, `tbsp`, `tsp`, `pc`, `oz`,
  `lb`, `cup`. Extend the `unit_aliases` map in `config.yaml`.
- **Text normalisation** — trims whitespace, collapses internal runs of
  whitespace, title-cases the recipe name. Configurable per field.

## Report shape

```json
{
  "checked": 312,
  "issue_count": 47,
  "issues": [
    {
      "recipe_url": "https://fa.resoee.com/.../recipe/123/edit",
      "recipe_name": "Paneer Tikka",
      "field": "ingredient[2].unit",
      "kind": "normalise",
      "before": "grams",
      "after": "g",
      "applied": false
    }
  ]
}
```

`kind` is either `missing` (you'll need to fill it in by hand) or
`normalise` (the script will fix it when `apply_changes` is true).
