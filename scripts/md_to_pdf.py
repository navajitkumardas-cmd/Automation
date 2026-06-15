"""Convert HANDOUT.md into a styled PDF."""
from pathlib import Path
import markdown
from weasyprint import HTML, CSS

SRC = Path("/home/user/Automation/HANDOUT.md")
OUT = Path("/home/user/Automation/HANDOUT.pdf")

md_text = SRC.read_text(encoding="utf-8")
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

css = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Helvetica", "Arial", sans-serif; font-size: 10.5pt;
       color: #222; line-height: 1.45; }
h1 { font-size: 22pt; color: #1a3a6c; border-bottom: 2px solid #1a3a6c;
     padding-bottom: 4pt; margin-top: 14pt; }
h2 { font-size: 15pt; color: #1a3a6c; margin-top: 20pt;
     border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 12pt; color: #2c5697; margin-top: 14pt; }
p, li { margin: 4pt 0; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px;
       font-family: "Menlo", "Consolas", monospace; font-size: 9.5pt; }
pre { background: #f3f4f6; padding: 8pt; border-radius: 4px; overflow-x: auto;
      font-size: 9pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #d0d4da; padding: 4pt 6pt; text-align: left; }
th { background: #eef2f8; color: #1a3a6c; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote { border-left: 4px solid #1a3a6c; margin: 6pt 0; padding: 4pt 10pt;
             background: #f7f9fc; color: #444; }
hr { border: none; border-top: 1px solid #ccc; margin: 14pt 0; }
"""

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Bakingo PO Forecasting Handout</title></head><body>{html_body}</body></html>"""

HTML(string=html).write_pdf(OUT, stylesheets=[CSS(string=css)])
print(f"Wrote {OUT}")
