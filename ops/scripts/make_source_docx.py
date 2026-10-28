"""Build a Word document containing the complete project source code.

Usage
-----
    python ops/scripts/make_source_docx.py

Writes ``docs/HR_Insight_Source_Code.docx``: a title page, a contents list,
then every source file grouped by area, each starting on a fresh page in
monospace with page numbers in the footer. Configuration files (requirements,
Makefile, helper batch scripts) are included as a final section. Secrets
(``.env``) are deliberately excluded.

Re-run it after any code change to refresh the document.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "docs" / "HR_Insight_Source_Code.docx"

#: azure-ish accents consistent with the app's day palette.
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x59, 0x62, 0x73)

#: One ``*`` entry = "glob pattern" relative to the project root.
CODE_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Application - entry point and design system",
        ["app/__init__.py", "app/main.py", "app/theme.py"],
    ),
    (
        "Services - application logic",
        [
            "app/services/__init__.py",
            "app/services/assistant_service.py",
            "app/services/audit_service.py",
            "app/services/data_service.py",
            "app/services/model_service.py",
        ],
    ),
    (
        "Pages - Streamlit views",
        [
            "app/views/__init__.py",
            "app/views/overview.py",
            "app/views/employees.py",
            "app/views/employee_detail.py",
            "app/views/fairness.py",
            "app/views/model_performance.py",
            "app/views/audit.py",
            "app/views/assistant.py",
        ],
    ),
    (
        "Machine-learning pipeline",
        [
            "ml/__init__.py",
            "ml/train_all.py",
            "ml/build_dataset.py",
            "ml/train_attrition.py",
            "ml/train_ninebox.py",
            "ml/train_fairness.py",
            "ml/train_survival.py",
            "ml/generate_synthetic.py",
        ],
    ),
    (
        "Shared ML utilities (ml/common)",
        [
            "ml/common/__init__.py",
            "ml/common/features.py",
            "ml/common/fairness.py",
            "ml/common/metrics.py",
            "ml/common/explain.py",
            "ml/common/drift.py",
            "ml/common/io.py",
            "ml/common/tracking.py",
        ],
    ),
    (
        "Test suite",
        [
            "tests/conftest.py",
            "tests/test_assistant.py",
            "tests/test_features.py",
            "tests/test_fairness.py",
            "tests/test_models.py",
            "tests/test_app.py",
        ],
    ),
    (
        "Operations scripts",
        [
            "ops/scripts/assistant_smoke.py",
            "ops/scripts/make_source_docx.py",
            "ops/scripts/screenshots.py",
            "ops/scripts/retrain.sh",
            "ops/scripts/seed_data.sh",
        ],
    ),
    (
        "Configuration",
        [
            "requirements.txt",
            "requirements-dev.txt",
            "Makefile",
            "run.bat",
            "retrain.bat",
        ],
    ),
]

#: Files that may be missing without aborting the build (optional config).
ALLOW_MISSING = {"tests/conftest.py", "Makefile"}


def _existing_files() -> list[str]:
    """All paths that will be documented and exist on disk."""
    out = []
    for _title, paths in CODE_SECTIONS:
        out.extend(str(ROOT / p) for p in paths if (ROOT / p).exists())
    return out


def _add_style(doc: Document) -> None:
    """Define the CodeBlock / FileName styles used for every file."""
    code = doc.styles.add_style("CodeBlock", WD_STYLE_TYPE.PARAGRAPH)
    code.base_style = doc.styles["Normal"]
    code.font.name = "Consolas"
    code.font.size = Pt(8)
    rpr = code.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "Consolas")
    rfonts.set(qn("w:cs"), "Consolas")
    code.paragraph_format.space_after = Pt(0)
    code.paragraph_format.space_before = Pt(0)
    code.paragraph_format.line_spacing = 1.0

    name = doc.styles.add_style("FileName", WD_STYLE_TYPE.PARAGRAPH)
    name.font.name = "Consolas"
    name.font.size = Pt(10.5)
    name.font.bold = True
    name.font.color.rgb = ACCENT
    name.paragraph_format.space_after = Pt(6)
    name.paragraph_format.keep_with_next = True


def _page_number_footer(doc: Document) -> None:
    """Centered footer: 'HR Insight - Complete Source Code | Page N'."""
    footer = doc.sections[0].footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("HR Insight - Complete Source Code  |  Page ")
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    placeholder = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    placeholder.append(text)
    field.append(placeholder)
    paragraph._p.append(field)


def _title_page(doc: Document, n_files: int, n_lines: int) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(140)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("HR Insight")
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.color.rgb = ACCENT

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run("Smart HR Decision Support Platform")
    sub_run.font.size = Pt(16)
    sub_run.font.color.rgb = MUTED

    tag = doc.add_paragraph()
    tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tag_run = tag.add_run("Complete Source Code")
    tag_run.font.size = Pt(13)
    tag_run.font.bold = True

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_run = meta.add_run(
        f"Machine Learning (CCSML0301) - BTech AI - Semester 3\n"
        f"{n_files} files - {n_lines:,} lines of source\n"
        f"Generated {dt.date.today().strftime('%d %B %Y')}"
    )
    meta_run.font.size = Pt(10)
    meta_run.font.color.rgb = MUTED
    doc.add_page_break()


def _contents_page(doc: Document) -> None:
    heading = doc.add_paragraph()
    heading_run = heading.add_run("Contents")
    heading_run.font.size = Pt(18)
    heading_run.font.bold = True
    heading_run.font.color.rgb = ACCENT

    for index, (title, paths) in enumerate(CODE_SECTIONS, start=1):
        line = doc.add_paragraph()
        line.paragraph_format.space_before = Pt(8)
        line.paragraph_format.space_after = Pt(2)
        label_run = line.add_run(f"{index}. {title}")
        label_run.font.bold = True
        label_run.font.size = Pt(10)
        for path in paths:
            if (ROOT / path).exists():
                item = doc.add_paragraph()
                item.paragraph_format.space_after = Pt(0)
                item_run = item.add_run("    " + path)
                item_run.font.name = "Consolas"
                item_run.font.size = Pt(8.5)
                item_run.font.color.rgb = MUTED
    doc.add_page_break()


def _add_file(doc: Document, relative: str) -> None:
    absolute = ROOT / relative
    text = absolute.read_text(encoding="utf-8")
    text = text.replace("\t", "    ").rstrip("\n")

    name_paragraph = doc.add_paragraph(style="FileName")
    name_paragraph.add_run(relative)

    code_paragraph = doc.add_paragraph(style="CodeBlock")
    code_paragraph.add_run(text)
    doc.add_page_break()


def main() -> None:
    files = _existing_files()
    total_lines = sum((ROOT / p).read_text(encoding="utf-8").count("\n") + 1 for p in files)

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(1.6)
        section.bottom_margin = Cm(1.6)
        section.left_margin = Cm(1.8)
        section.right_margin = Cm(1.4)

    _add_style(doc)
    _page_number_footer(doc)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)

    _title_page(doc, len(files), total_lines)
    _contents_page(doc)

    for index, (title, paths) in enumerate(CODE_SECTIONS, start=1):
        section_heading = doc.add_paragraph()
        heading_run = section_heading.add_run(f"{index}. {title}")
        heading_run.font.size = Pt(15)
        heading_run.font.bold = True
        heading_run.font.color.rgb = ACCENT
        section_heading.paragraph_format.space_after = Pt(4)

        for path in paths:
            if (ROOT / path).exists():
                _add_file(doc, path)
            elif path in ALLOW_MISSING:
                missing = doc.add_paragraph()
                missing_run = missing.add_run(f"{path} - (not present in this checkout)")
                missing_run.font.color.rgb = MUTED
            else:
                raise FileNotFoundError(f"documented source file is missing: {path}")

    doc.save(OUT_PATH)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[docx] {OUT_PATH.relative_to(ROOT)}")
    print(f"[docx] {len(files)} files, ~{total_lines:,} lines, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
