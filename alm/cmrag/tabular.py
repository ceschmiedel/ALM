"""Tabular ingestion — spreadsheets and CSVs as retrievable domain context.

Most enterprise domain knowledge does not arrive as prose. It arrives as a
sheet: a chart of accounts, a supplier register, a headcount table. Dropping
such a file into a text-chunking pipeline destroys exactly what makes it
useful — the header row. A chunk that reads ``1 439 064 | 912 346 | 0.31``
retrieves against nothing, because the words a question uses ("revenue",
"gross margin", "Vestuário") live in the header and the key column, not in the
cell the answer needs.

So a table is rendered, not split:

* every row becomes a labelled record — ``coluna: valor`` for each column — so
  the vocabulary of the question appears in the text of the answer's chunk;
* rows are grouped into chunks that each **repeat the sheet name and column
  list**, so a retrieved fragment is self-describing;
* a synthetic **schema document** is emitted per sheet, carrying the column
  names, the row count and the range of every numeric column. This is what
  answers "what is in this dataset?", a question no individual row can answer.

Numbers are formatted, not stringified: a cell holding ``1439064.0`` is written
``1439064`` so a lexical match on a figure quoted in a question can actually
land.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from alm.core.errors import ALMError

logger = logging.getLogger(__name__)

#: Rows per retrievable chunk. Small enough that a chunk stays specific, large
#: enough that neighbouring rows (which a comparison question needs together)
#: usually land in the same fragment.
ROWS_PER_CHUNK = 12

CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}
TABULAR_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES

_BLANK = re.compile(r"^\s*$")


class TabularError(ALMError):
    """A spreadsheet or delimited file could not be read."""

    code = "tabular_error"


@dataclass
class Sheet:
    """One parsed table."""

    name: str
    headers: list[str]
    rows: list[list[str]] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _format_cell(value: Any) -> str:
    """Render a cell as the text a question would plausibly contain."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        # Whole floats are how spreadsheets store integers; printing "1439064.0"
        # would stop a lexical match on the figure as a person would write it.
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _is_blank_row(cells: Sequence[str]) -> bool:
    return all(not c or _BLANK.match(c) for c in cells)


def _dedupe_headers(raw: Sequence[str]) -> list[str]:
    """Name every column, uniquely — an unnamed column cannot be cited."""
    headers: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(raw):
        name = (value or "").strip() or f"coluna_{index + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        headers.append(name if count == 0 else f"{name}_{count + 1}")
    return headers


# -- readers -----------------------------------------------------------------


def read_csv(data: bytes, *, name: str = "dados", delimiter: str = "") -> list[Sheet]:
    """Parse delimited text, sniffing the delimiter when not given."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")

    if not delimiter:
        sample = text[:8192]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            # Sniffing fails on single-column files; comma is the safe default.
            delimiter = ","

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    records = [[_format_cell(c) for c in row] for row in reader]
    records = [r for r in records if not _is_blank_row(r)]
    if not records:
        raise TabularError(f"{name}: the file has no readable rows")

    headers = _dedupe_headers(records[0])
    width = len(headers)
    rows = [(r + [""] * width)[:width] for r in records[1:]]
    return [Sheet(name=name, headers=headers, rows=rows)]


def read_excel(data: bytes, *, name: str = "planilha") -> list[Sheet]:
    """Parse every non-empty sheet of an XLSX workbook."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - openpyxl is a core dependency
        raise TabularError(
            "reading .xlsx needs openpyxl: pip install openpyxl"
        ) from exc

    try:
        # data_only: formulas are useless as context; their cached results are
        # the actual domain fact. A workbook saved without cached values yields
        # empty cells, which is reported honestly rather than papered over.
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise TabularError(f"{name}: could not read the workbook ({exc})") from exc

    sheets: list[Sheet] = []
    for worksheet in workbook.worksheets:
        records: list[list[str]] = []
        for row in worksheet.iter_rows(values_only=True):
            cells = [_format_cell(c) for c in row]
            if not _is_blank_row(cells):
                records.append(cells)
        if len(records) < 2:
            # A sheet with only a header (or nothing) carries no records.
            continue
        headers = _dedupe_headers(records[0])
        width = len(headers)
        rows = [(r + [""] * width)[:width] for r in records[1:]]
        sheets.append(Sheet(name=worksheet.title or name, headers=headers, rows=rows))

    workbook.close()
    if not sheets:
        raise TabularError(f"{name}: no sheet in this workbook has data rows")
    return sheets


def read_table(data: bytes, filename: str) -> list[Sheet]:
    """Dispatch on the file extension."""
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem or "dados"
    if suffix in EXCEL_SUFFIXES:
        return read_excel(data, name=stem)
    if suffix in CSV_SUFFIXES:
        return read_csv(data, name=stem)
    raise TabularError(
        f"unsupported file type {suffix or '(none)'}; expected one of: "
        + ", ".join(sorted(TABULAR_SUFFIXES))
    )


# -- rendering ---------------------------------------------------------------


def _numeric_summary(sheet: Sheet) -> list[str]:
    """Describe numeric columns by range — the shape of the data, in words."""
    lines: list[str] = []
    for index, header in enumerate(sheet.headers):
        values: list[float] = []
        for row in sheet.rows:
            raw = row[index].replace(".", "").replace(",", ".") if row[index].count(",") == 1 else row[index]
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        # Only call a column numeric when most of it is; a mostly-text column
        # with a few stray numbers would otherwise get a meaningless range.
        if len(values) >= max(2, int(0.6 * len(sheet.rows))):
            lines.append(
                f"- {header}: numérica, de {_format_cell(min(values))} a "
                f"{_format_cell(max(values))} ({len(values)} valores)"
            )
    return lines


def schema_document(sheet: Sheet, *, source: str) -> str:
    """The self-description of a sheet — what a 'what is in this data' question hits."""
    lines = [
        f"# {sheet.name} — estrutura dos dados",
        "",
        f"Origem: {source}",
        f"Registros: {sheet.row_count}",
        f"Colunas ({len(sheet.headers)}): {', '.join(sheet.headers)}",
    ]
    numeric = _numeric_summary(sheet)
    if numeric:
        lines += ["", "Colunas numéricas:", *numeric]

    if sheet.rows:
        lines += ["", "Primeiros registros como amostra:"]
        for row in sheet.rows[:3]:
            pairs = [f"{h}: {v}" for h, v in zip(sheet.headers, row, strict=False) if v]
            lines.append("- " + "; ".join(pairs))
    return "\n".join(lines)


def render_chunks(sheet: Sheet, *, source: str, rows_per_chunk: int = ROWS_PER_CHUNK) -> list[dict[str, Any]]:
    """Render a sheet as self-describing, retrievable chunks."""
    chunks: list[dict[str, Any]] = []
    header_line = ", ".join(sheet.headers)

    for start in range(0, len(sheet.rows), rows_per_chunk):
        block = sheet.rows[start : start + rows_per_chunk]
        lines = [
            f"{sheet.name} — registros {start + 1} a {start + len(block)} de {sheet.row_count}",
            f"Colunas: {header_line}",
            "",
        ]
        for offset, row in enumerate(block):
            pairs = [
                f"{header}: {value}"
                for header, value in zip(sheet.headers, row, strict=False)
                if value
            ]
            if pairs:
                lines.append(f"[{start + offset + 1}] " + "; ".join(pairs))
        chunks.append(
            {
                "text": "\n".join(lines),
                "ordinal": len(chunks),
                "metadata": {
                    "section": sheet.name,
                    "source": source,
                    "kind": "table_rows",
                    "row_start": start + 1,
                    "row_end": start + len(block),
                },
            }
        )
    return chunks


def sheet_to_document(
    sheet: Sheet, *, source: str, rows_per_chunk: int = ROWS_PER_CHUNK
) -> dict[str, Any]:
    """Everything the ingestor needs to store one sheet."""
    schema_text = schema_document(sheet, source=source)
    chunks: list[dict[str, Any]] = [
        {
            "text": schema_text,
            "ordinal": 0,
            "metadata": {"section": sheet.name, "source": source, "kind": "table_schema"},
        }
    ]
    for chunk in render_chunks(sheet, source=source, rows_per_chunk=rows_per_chunk):
        chunk["ordinal"] = len(chunks)
        chunks.append(chunk)

    return {
        "title": sheet.name,
        "sheet": sheet.name,
        "rows": sheet.row_count,
        "columns": list(sheet.headers),
        "chunks": chunks,
        # The full rendering is what the content hash is taken over, so
        # re-uploading an unchanged file is correctly a no-op.
        "text": "\n\n".join(c["text"] for c in chunks),
    }
