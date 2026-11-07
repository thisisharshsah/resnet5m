# Report

The project report and everything needed to rebuild it.

| File | What it is |
|---|---|
| `report.pdf` | the final report, 8 pages |
| `report.docx` | the same report as an editable Word file |
| `figures/` | the four figures used in the report |
| `make_figures.py` | draws `figures/` from the saved results in `../runs/` |
| `build_report.js` | writes `report.docx` with [docx](https://docx.js.org) |

## Rebuild

From the repository root:

```bash
make report
```

This needs the notebook extras (`pip install -e ".[notebook]"`), Node 18 or newer and
LibreOffice for the PDF. On macOS, point the target at the LibreOffice app:

```bash
make report SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
```

`report.pdf` was made with LibreOffice and the Liberation Serif font. A rebuild with a different
font or Word can break a few lines differently, so the committed PDF is the reference version.
