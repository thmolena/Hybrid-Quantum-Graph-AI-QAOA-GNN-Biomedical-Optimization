from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import nbformat
from bs4 import BeautifulSoup
from nbconvert.exporters.html import HTMLExporter
from nbconvert.exporters.templateexporter import TemplateExporter
from nbconvert.filters.highlight import Highlight2HTML
from nbconvert.filters.widgetsdatatypefilter import WidgetsDataTypeFilter
from nbformat import NotebookNode


PLACEHOLDER_ALT = "No description has been provided for this image"
INLINE_IMAGE_PATTERN = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL)
SUPPORTED_MIME_TYPES = ("image/png", "image/jpeg", "image/jpg")
EXPORT_HELPER_PATTERNS = (
    "export_notebook_html(",
    "export_notebook_html_artifact(",
    "nbconvert",
)


def _normalize_base64(payload: Any) -> str:
    if isinstance(payload, list):
        payload = "".join(payload)
    return "".join(str(payload).split())


def _collect_inline_image_alt_map(nb: NotebookNode) -> dict[tuple[str, str], str]:
    alt_map: dict[tuple[str, str], str] = {}
    for cell in nb.cells:
        cell_metadata = cell.get("metadata", {}) or {}
        cell_alt = cell_metadata.get("alt") if isinstance(cell_metadata, dict) else None
        for output in cell.get("outputs", []):
            data = output.get("data", {}) or {}
            metadata = output.get("metadata", {}) or {}
            output_alt = output.get("alt")
            for mime_type in SUPPORTED_MIME_TYPES:
                payload = data.get(mime_type)
                if not payload:
                    continue
                mime_metadata = metadata.get(mime_type, {})
                metadata_alt = mime_metadata.get("alt") if isinstance(mime_metadata, dict) else None
                alt_text = metadata_alt or output_alt or cell_alt
                if not alt_text:
                    continue
                alt_map[(mime_type, _normalize_base64(payload))] = str(alt_text)
    return alt_map


def _lookup_inline_alt(src: str, alt_map: dict[tuple[str, str], str]) -> str | None:
    match = INLINE_IMAGE_PATTERN.match(src)
    if not match:
        return None
    mime_type, payload = match.groups()
    return alt_map.get((mime_type, _normalize_base64(payload)))


def _strip_export_helper_outputs(nb: NotebookNode) -> NotebookNode:
    sanitized_nb = copy.deepcopy(nb)
    for cell in sanitized_nb.cells:
        if cell.get("cell_type") != "code":
            continue
        source_text = "".join(cell.get("source", []))
        if not any(pattern in source_text for pattern in EXPORT_HELPER_PATTERNS):
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
    return sanitized_nb


class MetadataAwareHTMLExporter(HTMLExporter):
    def from_notebook_node(  # type: ignore[override]
        self, nb: NotebookNode, resources: dict[str, Any] | None = None, **kw: Any
    ) -> tuple[str, dict[str, Any]]:
        langinfo = nb.metadata.get("language_info", {})
        lexer = langinfo.get("pygments_lexer", langinfo.get("name", None))
        highlight_code = self.filters.get(
            "highlight_code", Highlight2HTML(pygments_lexer=lexer, parent=self)
        )

        resources = self._init_resources(resources)

        filter_data_type = WidgetsDataTypeFilter(
            notebook_metadata=self._nb_metadata, parent=self, resources=resources
        )

        self.register_filter("highlight_code", highlight_code)
        self.register_filter("filter_data_type", filter_data_type)
        html, resources = TemplateExporter.from_notebook_node(self, nb, resources, **kw)

        alt_map = _collect_inline_image_alt_map(nb)
        soup = BeautifulSoup(html, features="html.parser")
        missing_alt = 0
        for elem in soup.select("img:not([alt])"):
            alt_text = _lookup_inline_alt(elem.attrs.get("src", ""), alt_map)
            if alt_text:
                elem.attrs["alt"] = alt_text
                continue
            elem.attrs["alt"] = PLACEHOLDER_ALT
            missing_alt += 1
        if missing_alt:
            self.log.warning("Alternative text is missing on %s image(s).", missing_alt)
        for elem in soup.select(".jp-Notebook div.jp-Cell-inputWrapper"):
            elem.attrs["tabindex"] = "0"
        for elem in soup.select(".jp-Notebook div.jp-OutputArea-output"):
            elem.attrs["tabindex"] = "0"

        return str(soup), resources


def export_notebook_html(
    notebook_path: str | Path,
    output_name: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    notebook_path = Path(notebook_path).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else notebook_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = output_name or f"{notebook_path.stem}.html"
    output_path = output_dir / output_name

    with notebook_path.open("r", encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)
    export_nb = _strip_export_helper_outputs(nb)

    exporter = MetadataAwareHTMLExporter()
    body, resources = exporter.from_notebook_node(
        export_nb,
        resources={
            "metadata": {
                "name": notebook_path.stem,
                "path": str(notebook_path.parent),
            }
        },
    )
    output_path.write_text(body, encoding="utf-8")

    for relative_name, payload in (resources.get("outputs") or {}).items():
        target_path = output_dir / relative_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            target_path.write_bytes(payload)
        else:
            target_path.write_text(str(payload), encoding="utf-8")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a notebook to HTML with deterministic image alt text.")
    parser.add_argument("notebook_path", help="Path to the notebook to export.")
    parser.add_argument("--output", dest="output_name", help="Output HTML filename.")
    parser.add_argument("--output-dir", dest="output_dir", help="Directory for exported HTML.")
    args = parser.parse_args()

    output_path = export_notebook_html(
        notebook_path=args.notebook_path,
        output_name=args.output_name,
        output_dir=args.output_dir,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())