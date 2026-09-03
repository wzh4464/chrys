# Copyright (c) 2026 Chrys. All rights reserved.

"""Catalog extraction, update, compilation, and pseudo-locale tests."""

from __future__ import annotations

import gettext
import io
import json
import os
from pathlib import Path

import pytest
from babel.messages.catalog import Catalog
from babel.messages.mofile import read_mo, write_mo
from babel.messages.pofile import read_po, write_po
from scripts import i18n

from chrys.foundation.i18n.formatting import has_visible_content, parse_placeholder_names, validate_authored_template
from tests.support.ci import CI_LINUX_ONLY


def _source_root(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    (root / "messages.py").write_text(source, encoding="utf-8")
    return root


def _catalog_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "locales" / "chrys.pot",
        tmp_path / "locales" / "zh-Hans" / "LC_MESSAGES" / "chrys.po",
        tmp_path / "catalogs" / "zh-Hans" / "LC_MESSAGES" / "chrys.mo",
    )


def _read_catalog(path: Path) -> Catalog:
    with path.open("rb") as stream:
        return read_po(stream, domain="chrys", abort_invalid=True)


def _write_catalog(path: Path, catalog: Catalog) -> None:
    with path.open("wb") as stream:
        write_po(stream, catalog, width=0, sort_output=True, include_lineno=True)


def _message(catalog: Catalog, key: str):
    message = catalog.get(key)
    assert message is not None
    return message


def _prepare_translated_catalog(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "CLOSE = msg('dialog.close', fallback='Close {name}')\n"
        "FILES = msg(\n"
        "    'dialog.files',\n"
        "    fallback='One {name}',\n"
        "    plural_fallback='{count} {name}s',\n"
        ")\n"
        "EMPTY = msg('dialog.empty', fallback='Untranslated')\n"
        "FUZZY = msg('dialog.fuzzy', fallback='Review me')\n",
    )
    pot_path, po_path, mo_path = _catalog_paths(tmp_path)
    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)
    catalog = _read_catalog(po_path)
    _message(catalog, "dialog.close").string = "关闭 {name}"
    _message(catalog, "dialog.files").string = ("{count} 个 {name}",)
    fuzzy = _message(catalog, "dialog.fuzzy")
    fuzzy.string = "需要复核"
    fuzzy.flags.add("fuzzy")
    _write_catalog(po_path, catalog)
    i18n.compile_catalog(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
        mo_path=mo_path,
        location_root=source_root,
    )
    return source_root, pot_path, po_path, mo_path


_SEMANTIC_METADATA_HEADERS = (
    "Project-Id-Version",
    "Report-Msgid-Bugs-To",
    "POT-Creation-Date",
    "PO-Revision-Date",
    "Last-Translator",
    "Language",
    "Language-Team",
    "Plural-Forms",
    "MIME-Version",
    "Content-Type",
    "Content-Transfer-Encoding",
    "Generated-By",
)


def _normalized_catalog_metadata(catalog: Catalog) -> dict[str, str | int]:
    headers = {name.casefold(): " ".join(value.split()) for name, value in catalog.mime_headers}
    report_address = "report-msgid-bugs-to"
    if headers.get(report_address) == "EMAIL@ADDRESS":
        # Babel's public MO reader synthesizes its default for the PO's empty
        # value, so these two public-reader representations are equivalent.
        headers[report_address] = ""
    normalized = {name.casefold(): headers.get(name.casefold(), "") for name in _SEMANTIC_METADATA_HEADERS}
    normalized.update(
        {
            "charset": (catalog.charset or "").casefold(),
            "locale": str(catalog.locale),
            "num-plurals": catalog.num_plurals,
            "plural-expression": " ".join(catalog.plural_expr.split()),
        }
    )
    return normalized


def _entry_forms(message) -> tuple[str, ...]:
    if isinstance(message.string, list):
        return tuple(form or "" for form in message.string)
    return i18n._translation_forms(message)


def _entry_id(message) -> str | tuple[str, str]:
    return message.id if isinstance(message.id, str) else tuple(message.id)


def _effective_catalog_entries(catalog: Catalog) -> dict[str | tuple[str, str], tuple[str, ...]]:
    effective: dict[str | tuple[str, str], tuple[str, ...]] = {}
    for message in catalog:
        if not message.id or message.fuzzy:
            continue
        forms = _entry_forms(message)
        if forms and all(has_visible_content(form) for form in forms):
            effective[_entry_id(message)] = forms
    return effective


def _mo_catalog_entries(catalog: Catalog) -> dict[str | tuple[str, str], tuple[str, ...]]:
    # The compiler only emits effective entries, so an MO row without visible
    # content in every form is a violation in its own right — filtering it out
    # the way the PO side does would let the runtime serve invisible text.
    entries: dict[str | tuple[str, str], tuple[str, ...]] = {}
    for message in catalog:
        if not message.id:
            continue
        forms = _entry_forms(message)
        assert forms and all(has_visible_content(form) for form in forms), (
            "MO entries must all carry visible translation content"
        )
        entries[_entry_id(message)] = forms
    return entries


def _plural_representative_counts(catalog: Catalog) -> dict[int, int]:
    plural_index = gettext.c2py(catalog.plural_expr)
    representatives: dict[int, int] = {}
    for count in range(1001):
        index = plural_index(count)
        if 0 <= index < catalog.num_plurals:
            representatives.setdefault(index, count)
        if len(representatives) == catalog.num_plurals:
            break
    assert set(representatives) == set(range(catalog.num_plurals)), "plural indexes need representative counts"
    return representatives


def _assert_po_mo_semantically_consistent(
    *,
    source_root: Path,
    pot_path: Path,
    po_path: Path,
    mo_path: Path,
) -> None:
    # No explicit ``location_root``: extraction derives it from *source_root*,
    # which is the only spelling that is right for both callers — a tmp tree
    # roots at itself, the live tree roots at the repo, and the tracked
    # catalogs record ``src/chrys/...`` paths that only the latter produces.
    _messages, po = i18n.check_catalogs(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
    )
    try:
        mo_bytes = mo_path.read_bytes()
        mo = read_mo(io.BytesIO(mo_bytes))
        runtime = gettext.GNUTranslations(io.BytesIO(mo_bytes))
    except (OSError, EOFError, LookupError, RuntimeError, SyntaxError, TypeError, ValueError) as error:
        raise AssertionError("tracked MO must be loadable by Babel and stdlib gettext") from error

    assert all(message.context is None for message in po if message.id), "PO context entries are forbidden"
    assert all(message.context is None for message in mo if message.id), "MO context entries are forbidden"
    assert _normalized_catalog_metadata(mo) == _normalized_catalog_metadata(po), "catalog metadata is stale"

    # Babel's MO reader synthesizes defaults for missing header fields, so the
    # parsed-catalog comparison above cannot prove the header block is
    # physically present; stdlib's own parse of the MO header is the physical
    # truth the runtime reads.
    po_headers = {name.casefold(): " ".join(value.split()) for name, value in po.mime_headers}
    mo_physical_headers = {name.casefold(): " ".join(value.split()) for name, value in runtime.info().items()}
    for name in _SEMANTIC_METADATA_HEADERS:
        expected = po_headers.get(name.casefold(), "")
        if expected:
            assert mo_physical_headers.get(name.casefold()) == expected, (
                f"MO physical header {name} is missing or does not match the PO"
            )

    po_entries = _effective_catalog_entries(po)
    mo_entries = _mo_catalog_entries(mo)
    assert set(mo_entries) == set(po_entries), "MO effective entry set does not match the PO"
    assert mo_entries == po_entries, "MO translation content is stale relative to the PO"

    plural_forms = tuple(field.strip() for field in runtime.info().get("plural-forms", "").split(";") if field.strip())
    assert plural_forms == (f"nplurals={po.num_plurals}", f"plural={po.plural_expr}"), (
        "stdlib gettext plural metadata does not match the PO"
    )
    representatives = _plural_representative_counts(po)
    for message_id, forms in po_entries.items():
        if isinstance(message_id, tuple):
            singular_id, plural_id = message_id
            for index, count in representatives.items():
                assert runtime.ngettext(singular_id, plural_id, count) == forms[index]
        else:
            assert runtime.gettext(message_id) == forms[0]


def _read_mo_catalog(path: Path) -> Catalog:
    with path.open("rb") as stream:
        return read_mo(stream)


def _write_mo_catalog(path: Path, catalog: Catalog) -> None:
    with path.open("wb") as stream:
        write_mo(stream, catalog, use_fuzzy=False)


def test_extract_records_plural_shape_and_round_trip_metadata_deterministically(tmp_path: Path) -> None:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "TITLE = msg('dialog.title', fallback='Hello {name}')\n"
        "FILES = msg(\n"
        "    'dialog.files',\n"
        "    fallback='One {name}',\n"
        "    plural_fallback='{count} {name}s',\n"
        ")\n"
        "HELP = msg('dialog.help', fallback='First line\\nSecond line', multiline=True)\n",
    )
    pot_path, _, _ = _catalog_paths(tmp_path)

    extracted = i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    first_bytes = pot_path.read_bytes()
    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)

    assert pot_path.read_bytes() == first_bytes
    assert [message.key for message in extracted] == ["dialog.files", "dialog.help", "dialog.title"]
    catalog = _read_catalog(pot_path)
    plural = _message(catalog, "dialog.files")
    assert plural.id == ("dialog.files", "dialog.files#plural")
    assert any(comment.startswith("English: ") for comment in plural.auto_comments)
    assert any(comment.startswith("English-plural: ") for comment in plural.auto_comments)
    assert any(comment.startswith("chrys-meta=") for comment in plural.auto_comments)
    metadata = i18n._parse_metadata(plural, label="test POT")
    files = next(message for message in extracted if message.key == "dialog.files")
    assert metadata.fingerprint == files.fingerprint
    assert metadata.placeholders == {"name"}
    assert metadata.multiline is False
    assert i18n._parse_metadata(_message(catalog, "dialog.help"), label="test POT").multiline is True


def test_extractor_ignores_imported_reuse_of_one_definition(tmp_path: Path) -> None:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Close')\n",
    )
    (source_root / "consumer.py").write_text(
        "from .messages import MESSAGE\nREFERENCE = MESSAGE.bind()\n", encoding="utf-8"
    )

    extracted = i18n.extract_messages(source_root, location_root=source_root)

    assert [message.key for message in extracted] == ["dialog.close"]


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("KEY = 'dialog.close'\nMESSAGE = msg(KEY, fallback='Close')\n", "key must be a literal string"),
        ("MESSAGE = msg('Close', fallback='Close')\n", "lowercase dotted segments"),
        ("FALLBACK = 'Close'\nMESSAGE = msg('dialog.close', fallback=FALLBACK)\n", "fallback must be a literal"),
        ("MESSAGE = msg('dialog.close')\n", "requires a literal fallback"),
        (
            "FIRST = msg('dialog.close', fallback='Close')\nSECOND = msg('dialog.close', fallback='Close')\n",
            r"Duplicate msg\(\) key",
        ),
        ("MESSAGE = msg('dialog.close', fallback='{value!r}')\n", "placeholder names"),
        ("MESSAGE = msg('dialog.close', fallback='{value.attr}')\n", "placeholder names"),
        ("MESSAGE = msg('dialog.close', fallback='{value[index]}')\n", "placeholder names"),
        ("MESSAGE = msg('dialog.close', fallback='{value:>10}')\n", "placeholder names"),
        ("MESSAGE = msg('dialog.close', fallback='{value:{width}}')\n", "placeholder names"),
        ("MESSAGE = msg('dialog.close', fallback='{value:}')\n", "placeholder names"),
        (
            "MESSAGE = msg('dialog.close', fallback='{name}', plural_fallback='{value}s')\n",
            "share their non-count placeholders",
        ),
        ("MESSAGE = msg('dialog.close', fallback='{count}')\n", "count placeholder requires plural_fallback"),
        ("MESSAGE = msg('dialog.close', fallback='')\n", "visible content"),
        ("MESSAGE = msg('dialog.close', fallback=' \\n ', multiline=True)\n", "visible content"),
        ("MESSAGE = msg('dialog.close', fallback='\u200b')\n", "visible content"),
        ("MESSAGE = msg('dialog.close', fallback='\u200b \u200b')\n", "visible content"),
        (
            "MESSAGE = msg('dialog.close', fallback='One', plural_fallback='\u200b')\n",
            "plural_fallback must have visible content",
        ),
        ("MESSAGE = msg('dialog.close', fallback='Bad\\ttext')\n", "Control characters"),
        ("MESSAGE = msg('dialog.close', fallback='Bad\\rtext')\n", "Control characters"),
        ("MESSAGE = msg('dialog.close', fallback='Bad\\x1btext')\n", "Control characters"),
        ("MESSAGE = msg('dialog.close', fallback='First\\nSecond')\n", "LF is forbidden"),
        ("MESSAGE = msg('dialog.close', fallback='[bold]Close[/bold]')\n", "markup"),
        (
            "def build():\n    MESSAGE = msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            "class Messages:\n    CLOSE = msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        (
            "if enabled:\n    MESSAGE = msg('dialog.close', fallback='Close')\n",
            "module-level assignment",
        ),
        ("REFERENCE = msg('dialog.close', fallback='Close').bind()\n", "module-level assignment"),
        (
            (
                "PLURAL = msg('dialog.item', fallback='One', plural_fallback='Many')\n"
                "COLLISION = msg('dialog.item#plural', fallback='Collision')\n"
            ),
            "lookup ID.*collides",
        ),
    ],
    ids=[
        "dynamic-key",
        "invalid-key",
        "dynamic-fallback",
        "missing-fallback",
        "duplicate-key",
        "conversion",
        "attribute-traversal",
        "index-traversal",
        "format-spec",
        "nested-format-spec",
        "empty-format-spec",
        "schema-disagreement",
        "count-without-plural",
        "empty-fallback",
        "whitespace-fallback",
        "zero-width-fallback",
        "mixed-invisible-fallback",
        "zero-width-plural-fallback",
        "tab",
        "carriage-return",
        "escape",
        "single-line-lf",
        "markup",
        "function-local",
        "class-local",
        "conditional",
        "inline-bind",
        "lookup-collision",
    ],
)
def test_extractor_rejects_invalid_definitions(tmp_path: Path, source: str, match: str) -> None:
    source_root = _source_root(tmp_path, "from chrys.foundation.i18n import msg\n" + source)

    with pytest.raises(i18n.CatalogToolError, match=match):
        i18n.extract_messages(source_root, location_root=source_root)


def test_update_refreshes_english_metadata_and_marks_wording_change_fuzzy(tmp_path: Path) -> None:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Close {name}')\n",
    )
    pot_path, po_path, _ = _catalog_paths(tmp_path)
    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)
    catalog = _read_catalog(po_path)
    _message(catalog, "dialog.close").string = "关闭 {name}"
    _write_catalog(po_path, catalog)

    (source_root / "messages.py").write_text(
        "from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Dismiss {name}')\n",
        encoding="utf-8",
    )
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)

    updated = _read_catalog(po_path)
    message = _message(updated, "dialog.close")
    metadata = i18n._parse_metadata(message, label="updated PO")
    fresh = i18n.extract_messages(source_root, location_root=source_root)
    assert metadata.fingerprint == fresh[0].fingerprint
    assert any("Dismiss" in comment for comment in message.auto_comments)
    assert message.string == "关闭 {name}"
    assert message.fuzzy


def test_compile_filters_fuzzy_and_untranslated_entries(tmp_path: Path) -> None:
    _, _, _, mo_path = _prepare_translated_catalog(tmp_path)

    with mo_path.open("rb") as stream:
        translations = gettext.GNUTranslations(stream)

    assert translations.gettext("dialog.close") == "关闭 {name}"
    assert translations.ngettext("dialog.files", "dialog.files#plural", 2) == "{count} 个 {name}"
    assert translations.gettext("dialog.empty") == "dialog.empty"
    assert translations.gettext("dialog.fuzzy") == "dialog.fuzzy"


@pytest.mark.parametrize("failure", ["corrupt", "partial-plural", "stale-source"])
def test_compile_failure_preserves_existing_mo_bytes(tmp_path: Path, failure: str) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    original = mo_path.read_bytes()

    if failure == "stale-source":
        source = (source_root / "messages.py").read_text(encoding="utf-8")
        (source_root / "messages.py").write_text(source.replace("Close {name}", "Dismiss {name}"), encoding="utf-8")
    elif failure == "corrupt":
        catalog = _read_catalog(po_path)
        _message(catalog, "dialog.close").string = "\x1b[31m坏翻译"
        _write_catalog(po_path, catalog)
    else:
        # A truthful nplurals=1 header cannot represent a partial plural —
        # Babel's parser rejects surplus forms — so the on-disk shape of this
        # failure class is a hand-tampered multi-form header.
        po_text = po_path.read_text(encoding="utf-8")
        po_text = po_text.replace("nplurals=1; plural=0;", "nplurals=2; plural=(n != 1);")
        po_text = po_text.replace('msgstr[0] "{count} 个 {name}"', 'msgstr[0] "{count} 个 {name}"\nmsgstr[1] ""')
        po_path.write_text(po_text, encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError):
        i18n.compile_catalog(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
            location_root=source_root,
        )

    assert mo_path.read_bytes() == original


def test_partially_translated_plural_is_a_gate_error_for_multi_form_locales(tmp_path: Path) -> None:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "FILES = msg('dialog.files', fallback='One {name}', plural_fallback='{count} {name}s')\n",
    )
    extracted = i18n.extract_messages(source_root, location_root=source_root)
    catalog = Catalog(domain="chrys")
    assert catalog.num_plurals == 2
    catalog.add(("dialog.files", "dialog.files#plural"), string=("一个 {name}", ""))

    with pytest.raises(i18n.CatalogToolError, match="Partially translated plural"):
        i18n.validate_translation_catalog(extracted, catalog)


def test_semantic_gate_rejects_an_effective_entry_missing_from_the_mo(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    mo = _read_mo_catalog(mo_path)
    mo.delete("dialog.close")
    _write_mo_catalog(mo_path, mo)

    with pytest.raises(AssertionError, match="effective entry set"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_semantic_gate_rejects_an_extra_effective_entry_in_the_mo(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    mo = _read_mo_catalog(mo_path)
    mo.add("dialog.extra", string="额外")
    _write_mo_catalog(mo_path, mo)

    with pytest.raises(AssertionError, match="effective entry set"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_semantic_gate_rejects_an_invisible_extra_mo_entry(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    mo = _read_mo_catalog(mo_path)
    mo.add("dialog.sneaky", string="​⁠")
    _write_mo_catalog(mo_path, mo)

    with pytest.raises(AssertionError, match="visible translation content"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_semantic_gate_rejects_mo_content_compiled_from_stale_po_text(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    po = _read_catalog(po_path)
    _message(po, "dialog.close").string = "关闭窗口 {name}"
    _write_catalog(po_path, po)

    with pytest.raises(AssertionError, match="translation content is stale"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_semantic_gate_rejects_mo_plural_metadata_that_differs_from_the_po(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    original = mo_path.read_bytes()
    tampered = original.replace(b"nplurals=1; plural=0;", b"nplurals=2; plural=0;")
    assert tampered != original
    mo_path.write_bytes(tampered)

    with pytest.raises(AssertionError, match="metadata is stale"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_semantic_gate_rejects_a_physically_stripped_mo_header(tmp_path: Path) -> None:
    # An all-ASCII translation keeps the tampered MO loadable — stdlib gettext
    # needs the header charset only for non-ASCII text — while Babel's MO
    # reader synthesizes the missing Content-Type on the parsed-catalog side.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nOK = msg('dialog.ok', fallback='OK')\n",
    )
    pot_path, po_path, mo_path = _catalog_paths(tmp_path)
    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)
    po = _read_catalog(po_path)
    _message(po, "dialog.ok").string = "Confirmed"
    _write_catalog(po_path, po)
    i18n.compile_catalog(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
        mo_path=mo_path,
        location_root=source_root,
    )
    original = mo_path.read_bytes()
    tampered = original.replace(b"Content-Type", b"Xontent-Type")
    assert tampered != original
    mo_path.write_bytes(tampered)

    with pytest.raises(AssertionError, match="physical header"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_semantic_gate_rejects_a_corrupt_mo(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    mo_path.write_bytes(b"not a GNU MO file")

    with pytest.raises(AssertionError, match="loadable by Babel and stdlib gettext"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


@pytest.mark.parametrize(
    ("key", "translation", "match"),
    [
        ("dialog.empty", "\x1b", "catalog text safety"),
        ("dialog.fuzzy", "[bold]需要复核[/bold]", "catalog text safety"),
        ("dialog.fuzzy", "\u200b dialog.files#plural \u200b", "any catalog lookup ID"),
    ],
    ids=["control-in-non-effective-form", "markup-in-fuzzy-form", "foreign-plural-lookup-id-in-fuzzy-form"],
)
def test_semantic_gate_rejects_unsafe_text_in_every_active_translation_form(
    tmp_path: Path,
    key: str,
    translation: str,
    match: str,
) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    po = _read_catalog(po_path)
    _message(po, key).string = translation
    _write_catalog(po_path, po)

    with pytest.raises(i18n.CatalogToolError, match=match):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


@pytest.mark.parametrize(
    ("translation", "match"),
    [
        ("关闭", "placeholder schema"),
        ("关闭 {other}", "placeholder schema"),
        ("关闭 {name!r}", "placeholder names"),
        ("关闭 {name.attr}", "placeholder names"),
        ("关闭 {name[index]}", "catalog text safety"),
        ("关闭 {name:>10}", "placeholder names"),
    ],
    ids=["omitted-slot", "unresolved-slot", "conversion", "attribute-traversal", "index-traversal", "format-spec"],
)
def test_semantic_gate_rejects_schema_violations_in_effective_translations(
    tmp_path: Path,
    translation: str,
    match: str,
) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    po = _read_catalog(po_path)
    _message(po, "dialog.close").string = translation
    _write_catalog(po_path, po)

    with pytest.raises(i18n.CatalogToolError, match=match):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_semantic_gate_exempts_fuzzy_text_only_from_schema_dependent_checks(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    po = _read_catalog(po_path)
    fuzzy = _message(po, "dialog.fuzzy")
    assert fuzzy.fuzzy
    fuzzy.string = "旧占位符 {old!r}"
    _write_catalog(po_path, po)

    _assert_po_mo_semantically_consistent(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
        mo_path=mo_path,
    )


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        ("fallback='Original'", "fallback='Changed'"),
        ("multiline=True", "multiline=False"),
    ],
    ids=["fallback-fingerprint", "multiline-policy"],
)
def test_semantic_gate_rejects_source_metadata_drift(
    tmp_path: Path,
    original: str,
    changed: str,
) -> None:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.message', fallback='Original', multiline=True)\n",
    )
    pot_path, po_path, mo_path = _catalog_paths(tmp_path)
    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)
    i18n.compile_catalog(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
        mo_path=mo_path,
        location_root=source_root,
    )
    source_path = source_root / "messages.py"
    source_path.write_text(source_path.read_text(encoding="utf-8").replace(original, changed), encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="stale"):
        _assert_po_mo_semantically_consistent(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
        )


def test_check_is_non_mutating_and_detects_stale_source(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    before = {path: path.read_bytes() for path in (pot_path, po_path, mo_path)}

    i18n.check_catalogs(
        source_root=source_root,
        pot_path=pot_path,
        po_path=po_path,
        location_root=source_root,
    )

    assert {path: path.read_bytes() for path in before} == before
    source = (source_root / "messages.py").read_text(encoding="utf-8")
    (source_root / "messages.py").write_text(source.replace("Review me", "Review this"), encoding="utf-8")
    with pytest.raises(i18n.CatalogToolError, match="stale"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


@pytest.mark.parametrize("stale_header", ["version", "project"])
@pytest.mark.parametrize("target", ["pot", "po"])
def test_check_rejects_stale_project_id_version_headers(tmp_path: Path, target: str, stale_header: str) -> None:
    source_root, pot_path, po_path, _mo_path = _prepare_translated_catalog(tmp_path)

    path = pot_path if target == "pot" else po_path
    catalog = _read_catalog(path)
    if stale_header == "version":
        catalog.version = "0.0.0"
    else:
        catalog.project = "NotChrys"
    _write_catalog(path, catalog)

    with pytest.raises(i18n.CatalogToolError, match="Project-Id-Version is stale"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_duplicate_physical_po_entries_are_rejected_not_collapsed(tmp_path: Path) -> None:
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    original = mo_path.read_bytes()
    po_text = po_path.read_text(encoding="utf-8")
    po_path.write_text(po_text + '\nmsgid "dialog.close"\nmsgstr "重复条目"\n', encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="duplicate"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )
    with pytest.raises(i18n.CatalogToolError, match="duplicate"):
        i18n.compile_catalog(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
            location_root=source_root,
        )

    assert mo_path.read_bytes() == original


def test_missing_physical_plural_forms_header_is_rejected(tmp_path: Path) -> None:
    # Babel infers nplurals=1/plural=0 from Language: zh_Hans alone, so the
    # semantic header pin passes even after the Plural-Forms line is deleted;
    # external gettext tools reading the same PO would fall back to their own
    # plural defaults instead.
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    original = mo_path.read_bytes()
    lines = po_path.read_text(encoding="utf-8").splitlines()
    stripped = [line for line in lines if "Plural-Forms" not in line]
    assert len(stripped) == len(lines) - 1
    po_path.write_text("\n".join(stripped) + "\n", encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="Plural-Forms"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )
    with pytest.raises(i18n.CatalogToolError, match="Plural-Forms"):
        i18n.compile_catalog(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
            location_root=source_root,
        )

    assert mo_path.read_bytes() == original


def test_missing_physical_content_type_header_is_rejected(tmp_path: Path) -> None:
    # Babel synthesizes catalog.charset == "utf-8" when the Content-Type
    # declaration is missing, so the parsed-charset pin passes while GNU
    # msgfmt rejects the headerless file.
    source_root, pot_path, po_path, _mo_path = _prepare_translated_catalog(tmp_path)
    lines = po_path.read_text(encoding="utf-8").splitlines()
    stripped = [line for line in lines if "Content-Type" not in line]
    assert len(stripped) == len(lines) - 1
    po_path.write_text("\n".join(stripped) + "\n", encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="Content-Type"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_duplicated_physical_plural_forms_header_is_rejected(tmp_path: Path) -> None:
    source_root, pot_path, po_path, _mo_path = _prepare_translated_catalog(tmp_path)
    header_line = '"Plural-Forms: nplurals=1; plural=0;\\n"'
    po_text = po_path.read_text(encoding="utf-8")
    assert po_text.count(header_line) == 1
    po_path.write_text(po_text.replace(header_line, header_line + "\n" + header_line), encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="Plural-Forms"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_translated_copy_of_plural_forms_line_cannot_replace_the_header(tmp_path: Path) -> None:
    # A multiline translation may legally contain the header text verbatim;
    # the physical pin must scan only the header entry, or a deleted header
    # could masquerade behind the translated copy.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nNOTE = msg('dialog.note', fallback='First\\nSecond', multiline=True)\n",
    )
    pot_path = tmp_path / "locales" / "chrys.pot"
    po_path = tmp_path / "locales" / "zh-Hans" / "LC_MESSAGES" / "chrys.po"
    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)

    header_line = '"Plural-Forms: nplurals=1; plural=0;\\n"'
    text = po_path.read_text(encoding="utf-8")
    assert text.count(header_line) == 1
    text = text.replace(header_line + "\n", "")
    text = text.replace('msgstr ""\n\n', 'msgstr ""\n"第一行\\n"\n' + header_line + "\n\n", 1)
    po_path.write_text(text, encoding="utf-8")
    assert po_path.read_text(encoding="utf-8").count("Plural-Forms") == 1

    with pytest.raises(i18n.CatalogToolError, match="Plural-Forms"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_physical_msgctxt_on_the_header_entry_is_rejected(tmp_path: Path) -> None:
    # Babel drops a msgctxt attached to the header entry before any
    # catalog-level check can see it, while GNU tools reject the file.
    source_root, pot_path, po_path, _mo_path = _prepare_translated_catalog(tmp_path)
    text = po_path.read_text(encoding="utf-8")
    po_path.write_text(text.replace('msgid ""', 'msgctxt "evil"\nmsgid ""', 1), encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="msgctxt"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_invalid_declared_po_charset_is_reported_not_a_crash(tmp_path: Path) -> None:
    source_root, pot_path, po_path, _mo_path = _prepare_translated_catalog(tmp_path)
    po_text = po_path.read_text(encoding="utf-8")
    assert "charset=utf-8" in po_text
    po_path.write_text(po_text.replace("charset=utf-8", "charset=no_such_codec"), encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="Could not read catalog"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_valid_non_utf8_declared_charset_is_rejected(tmp_path: Path) -> None:
    # A loadable charset like iso-8859-1 makes Babel decode the UTF-8 bytes
    # under the wrong codec, shipping mojibake translations silently.
    source_root, pot_path, po_path, _mo_path = _prepare_translated_catalog(tmp_path)
    po_text = po_path.read_text(encoding="utf-8")
    assert "charset=utf-8" in po_text
    po_path.write_text(po_text.replace("charset=utf-8", "charset=iso-8859-1"), encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="charset=UTF-8"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_extraction_rejects_msg_calls_without_the_canonical_import(tmp_path: Path) -> None:
    # A file-local callable that happens to be named msg must fail extraction
    # loudly instead of forging catalog entries from its call sites.
    source_root = _source_root(
        tmp_path,
        "def msg(key, fallback=None):\n"
        "    return (key, fallback)\n"
        "\n"
        "MESSAGE = msg('rogue.key', fallback='Rogue {name}')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="canonical"):
        i18n.extract_messages(source_root, location_root=source_root)


def test_extraction_rejects_a_conditional_canonical_import(tmp_path: Path) -> None:
    # A conditional canonical import can lose to a rogue same-name binding at
    # runtime while the extractor would still record the call site.
    source_root = _source_root(
        tmp_path,
        "import os\n"
        "if os.environ.get('ROGUE'):\n"
        "    from rogue_module import msg\n"
        "else:\n"
        "    from chrys.foundation.i18n import msg\n"
        "MESSAGE = msg('dialog.rogue', fallback='Rogue')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="canonical"):
        i18n.extract_messages(source_root, location_root=source_root)


def test_lone_surrogate_fallback_is_a_catalog_tool_error(tmp_path: Path) -> None:
    # A lone surrogate is a valid Python literal but cannot be encoded to
    # UTF-8, so it must fail validation instead of crashing the PO writer.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Bad\\udcff')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="surrogates not allowed"):
        i18n.extract_catalog(source_root=source_root, pot_path=tmp_path / "chrys.pot", location_root=source_root)


def test_duplicate_msgstr_fields_inside_one_entry_are_rejected(tmp_path: Path) -> None:
    # Babel keeps the first msgstr of a tampered duplicate pair while GNU
    # msgfmt rejects the file, so the duplicate would ship invisibly.
    source_root, pot_path, po_path, mo_path = _prepare_translated_catalog(tmp_path)
    original = mo_path.read_bytes()
    text = po_path.read_text(encoding="utf-8")
    target = 'msgid "dialog.close"\nmsgstr "关闭 {name}"'
    assert target in text
    po_path.write_text(
        text.replace(target, target + '\nmsgstr "\\x1b[31m邪恶 {name}"'),
        encoding="utf-8",
    )

    with pytest.raises(i18n.CatalogToolError, match="duplicate msgstr"):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )
    with pytest.raises(i18n.CatalogToolError, match="duplicate msgstr"):
        i18n.compile_catalog(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            mo_path=mo_path,
            location_root=source_root,
        )

    assert mo_path.read_bytes() == original


def test_extraction_rejects_a_shadowed_canonical_import(tmp_path: Path) -> None:
    # A class named msg rebinds the canonical import at runtime while the
    # extractor would still read literal call arguments.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nclass msg:\n    pass\nMESSAGE = msg('rogue.key', fallback='Rogue')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="rebound"):
        i18n.extract_messages(source_root, location_root=source_root)


def test_extraction_rejects_a_match_star_capture_shadow(tmp_path: Path) -> None:
    # A match-pattern rest capture rebinds msg at runtime, so later calls
    # would hit the captured list while extraction still reads the literals.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "FIRST = msg('dialog.first', fallback='First')\n"
        "def sort_names(names):\n"
        "    match names:\n"
        "        case [*msg]:\n"
        "            return list(msg)\n"
        "SECOND = msg('dialog.second', fallback='Second')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="rebound"):
        i18n.extract_messages(source_root, location_root=source_root)


def test_extraction_rejects_an_except_handler_shadow(tmp_path: Path) -> None:
    # ExceptHandler.name is a plain string in the AST, and Python deletes the
    # target after the handler, so later msg() calls raise NameError at
    # runtime while the extractor would still read the literals.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "FIRST = msg('dialog.first', fallback='First')\n"
        "try:\n"
        "    pass\n"
        "except* RuntimeError as msg:\n"
        "    pass\n"
        "SECOND = msg('dialog.second', fallback='Second')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="rebound"):
        i18n.extract_messages(source_root, location_root=source_root)


@pytest.mark.parametrize(
    "source",
    [
        "MESSAGE = msg('dialog.close', fallback='Close')\nfrom chrys.foundation.i18n import msg\n",
        "MESSAGE = msg('dialog.close', fallback='Close'); from chrys.foundation.i18n import msg\n",
    ],
    ids=["previous-line", "same-line-semicolon"],
)
def test_extraction_rejects_a_msg_call_before_the_canonical_import(tmp_path: Path, source: str) -> None:
    # A definition in a statement before the import raises NameError when the
    # module executes, while extraction would still record the literals;
    # semicolon-joined statements share a line, so ordinals decide, not
    # line numbers.
    source_root = _source_root(tmp_path, source)

    with pytest.raises(i18n.CatalogToolError, match="preceding canonical"):
        i18n.extract_messages(source_root, location_root=source_root)


def test_extraction_rejects_a_dotted_plain_import_root_rebind(tmp_path: Path) -> None:
    # ``import msg.submodule`` binds the ROOT name msg, so later calls hit a
    # module object at runtime while extraction still reads the literals.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "import msg.submodule\n"
        "MESSAGE = msg('dialog.close', fallback='Close')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="rebound"):
        i18n.extract_messages(source_root, location_root=source_root)


def test_extraction_rejects_a_member_import_rebind(tmp_path: Path) -> None:
    # A later i18n member import bound to the name msg wins at runtime, so
    # the calls would construct raw MessageDef objects, not references.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "from chrys.foundation.i18n import MessageDef as msg\n"
        "MESSAGE = msg('dialog.close', fallback='Close')\n",
    )

    with pytest.raises(i18n.CatalogToolError, match="rebound"):
        i18n.extract_messages(source_root, location_root=source_root)


def _mutate_first_metadata_comment(po_path: Path, respell) -> None:
    text = po_path.read_text(encoding="utf-8")
    line = next(candidate for candidate in text.splitlines() if candidate.startswith("#. chrys-meta={"))
    data = json.loads(line[len("#. chrys-meta=") :])
    po_path.write_text(text.replace(line, "#. chrys-meta=" + respell(data), 1), encoding="utf-8")


@pytest.mark.parametrize(
    "respell",
    [
        lambda data: json.dumps(data, sort_keys=True, separators=(", ", ": ")),
        lambda data: json.dumps({**data, "rogue": 1}, sort_keys=True, separators=(",", ":")),
        lambda data: json.dumps(
            {**data, "fingerprint": data["fingerprint"].upper()}, sort_keys=True, separators=(",", ":")
        ),
    ],
    ids=["spaced-json", "extra-key", "uppercase-fingerprint"],
)
def test_noncanonical_metadata_spellings_are_not_machine_metadata(tmp_path: Path, respell) -> None:
    # Babel's comment wrapping re-wraps any spelling other than the writer's
    # compact encoding on the next update, corrupting the very line a lenient
    # parser would have accepted, so check must reject it up front.
    source_root, pot_path, po_path, _ = _prepare_translated_catalog(tmp_path)
    _mutate_first_metadata_comment(po_path, respell)

    with pytest.raises(i18n.CatalogToolError, match="missing or duplicate Chrys source metadata"):
        i18n.check_catalogs(source_root=source_root, pot_path=pot_path, po_path=po_path, location_root=source_root)


def test_prose_containing_the_metadata_prefix_survives_comment_wrapping(tmp_path: Path) -> None:
    # Babel wraps prose comments at 76 columns, so legal English fallback text
    # can spill a chrys-meta=-prefixed fragment onto its own comment line;
    # only lines decoding to the canonical object are machine metadata.
    filler = "x" * 68
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        f"MESSAGE = msg('dialog.close', fallback='{filler} chrys-meta=forged')\n",
    )
    pot_path = tmp_path / "locales" / "chrys.pot"
    po_path = tmp_path / "locales" / "zh-Hans" / "LC_MESSAGES" / "chrys.po"

    i18n.extract_catalog(source_root=source_root, pot_path=pot_path, location_root=source_root)
    assert "#. chrys-meta=forged" in pot_path.read_text(encoding="utf-8").splitlines()
    i18n.update_catalog(source_root=source_root, po_path=po_path, location_root=source_root)
    i18n.check_catalogs(source_root=source_root, pot_path=pot_path, po_path=po_path, location_root=source_root)


@pytest.mark.parametrize(
    ("anchor", "replacement", "match"),
    [
        (
            'msgid "dialog.close"\nmsgstr "关闭 {name}"',
            'msgid "dialog.close"\nmsgstr "关闭 {name}"\n\nmsgstr "\\x1b[31m隐藏"',
            "stray msgstr",
        ),
        (
            'msgstr[0] "{count} 个 {name}"',
            'msgstr[0] "{count} 个 {name}"\nmsgstr[00] "\\x1b[31m重复"',
            "duplicate msgstr",
        ),
        (
            'msgid "dialog.close"\nmsgstr "关闭 {name}"',
            'msgid "dialog.close"\nmsgstr[0] "关闭"\nmsgstr[1] "坏[bold]文本"',
            "without msgid_plural",
        ),
        (
            'msgstr[0] "{count} 个 {name}"',
            'msgstr "{count} 个 {name}"',
            "plain msgstr on a plural entry",
        ),
        (
            'msgid "dialog.close"\nmsgstr "关闭 {name}"',
            'msgid "dialog.close"\nmsgstr "关闭 {name}"\n\nmsgid  "dialog.close"\nmsgstr "\\x1b[31m邪恶"',
            "duplicate entries",
        ),
        (
            'msgid "dialog.close"\nmsgstr "关闭 {name}"',
            'msgid "dialog.close"\nmsgstr [0] "\\x1b[31m翻译"',
            "malformed entry field",
        ),
        (
            'msgstr[0] "{count} 个 {name}"',
            'msgstr[0] "{count} 个 {name}"\nmsgstr[+0] "\\x1b[31m恶意"',
            "malformed entry field",
        ),
    ],
    ids=[
        "stray-after-blank",
        "zero-padded-index-duplicate",
        "indexed-forms-on-singular-entry",
        "plain-msgstr-on-plural-entry",
        "double-space-duplicate-msgid",
        "space-before-index-bracket",
        "signed-index-duplicate",
    ],
)
def test_physical_entry_field_tampering_is_rejected(tmp_path: Path, anchor: str, replacement: str, match: str) -> None:
    # Babel tolerates these shapes by keeping the first field it saw, while
    # GNU msgfmt rejects each of them outright.
    source_root, pot_path, po_path, _mo_path = _prepare_translated_catalog(tmp_path)
    text = po_path.read_text(encoding="utf-8")
    assert anchor in text
    po_path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match=match):
        i18n.check_catalogs(
            source_root=source_root,
            pot_path=pot_path,
            po_path=po_path,
            location_root=source_root,
        )


def test_mo_plural_metadata_validation_is_exact_not_substring(tmp_path: Path) -> None:
    # nplurals=10 contains nplurals=1 as a substring; the temp-MO gate must
    # compare the parsed fields exactly.
    catalog = Catalog(locale="zh_Hans", domain="chrys", fuzzy=False)
    catalog.add("dialog.close", string="关闭")
    buffer = io.BytesIO()
    write_mo(buffer, catalog, use_fuzzy=False)
    tampered = buffer.getvalue().replace(b"nplurals=1; plural=0;", b"nplurals=10; plural=0")
    assert tampered != buffer.getvalue()
    mo_path = tmp_path / "chrys.mo"
    mo_path.write_bytes(tampered)

    loaded = gettext.GNUTranslations(io.BytesIO(tampered))
    assert "nplurals=10" in loaded.info()["plural-forms"]
    with pytest.raises(i18n.CatalogToolError, match="plural metadata"):
        i18n._validate_mo(mo_path)


def test_check_command_reports_clear_nonzero_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_check() -> None:
        raise i18n.CatalogToolError("catalog metadata is stale")

    monkeypatch.setattr(i18n, "check_catalogs", fail_check)

    assert i18n.main(["check"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: catalog metadata is stale\n"


def test_pseudo_refuses_direct_repository_destination(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path, "")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(i18n.CatalogToolError, match="outside the repository"):
        i18n.generate_pseudo_catalog(
            repo_root / "locales",
            source_root=source_root,
            repo_root=repo_root,
            location_root=source_root,
        )


def test_pseudo_command_refuses_relative_locales_destination(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(i18n.REPO_ROOT)

    assert i18n.main(["pseudo", "--output", "locales"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "outside the repository root" in captured.err


def test_pseudo_refuses_symlink_resolving_into_repository(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path, "")
    repo_root = tmp_path / "repo"
    destination = repo_root / "generated"
    destination.mkdir(parents=True)
    symlink = tmp_path / "outside-link"
    try:
        symlink.symlink_to(destination, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    with pytest.raises(i18n.CatalogToolError, match="outside the repository"):
        i18n.generate_pseudo_catalog(
            symlink,
            source_root=source_root,
            repo_root=repo_root,
            location_root=source_root,
        )


def test_pseudo_refuses_descendant_symlink_back_into_repository(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path, "")
    repo_root = tmp_path / "repo"
    trap_target = repo_root / "generated"
    trap_target.mkdir(parents=True)
    output = tmp_path / "outside"
    output.mkdir()
    try:
        (output / "en-XA").symlink_to(trap_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    with pytest.raises(i18n.CatalogToolError, match="outside the repository"):
        i18n.generate_pseudo_catalog(
            output,
            source_root=source_root,
            repo_root=repo_root,
            location_root=source_root,
        )

    assert list(trap_target.iterdir()) == []


def test_pseudo_refuses_case_variant_spelling_of_repository_root(tmp_path: Path) -> None:
    # On case-insensitive filesystems a case-variant spelling survives
    # resolve() verbatim while naming the repository directory itself.
    source_root = _source_root(tmp_path, "")
    repo_root = tmp_path / "RepoCase"
    repo_root.mkdir()
    if not (tmp_path / "rEPOcASE").exists():
        pytest.skip("requires a case-insensitive filesystem")

    with pytest.raises(i18n.CatalogToolError, match="outside the repository"):
        i18n.generate_pseudo_catalog(
            tmp_path / "rEPOcASE" / "generated",
            source_root=source_root,
            repo_root=repo_root,
            location_root=source_root,
        )

    assert list(repo_root.iterdir()) == []


def test_pseudo_never_writes_through_a_hardlink_to_a_repository_file(tmp_path: Path) -> None:
    # A pre-existing MO hardlinked to a repository file shares its inode, so
    # an in-place open("wb") would rewrite the repository file's bytes even
    # though the target path itself sits outside the repository.
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\nMESSAGE = msg('dialog.close', fallback='Close')\n",
    )
    repo_root = tmp_path / "repo"
    victim = repo_root / "docs" / "notes.txt"
    victim.parent.mkdir(parents=True)
    victim.write_text("precious repository content", encoding="utf-8")
    output = tmp_path / "outside"
    target_dir = output / "en-XA" / "LC_MESSAGES"
    target_dir.mkdir(parents=True)
    try:
        os.link(victim, target_dir / "chrys.mo")
    except OSError as error:
        pytest.skip(f"hardlinks are unavailable: {error}")

    generated = i18n.generate_pseudo_catalog(
        output,
        source_root=source_root,
        repo_root=repo_root,
        location_root=source_root,
    )

    assert victim.read_text(encoding="utf-8") == "precious repository content"
    assert not os.path.samestat(generated.stat(), victim.stat())
    translations = gettext.GNUTranslations(io.BytesIO(generated.read_bytes()))
    assert translations.gettext("dialog.close") != "dialog.close"


def test_repository_identity_guard_contains_unbuilt_descendants(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert i18n._is_inside_repository(repo_root, repo_root)
    assert i18n._is_inside_repository(repo_root / "locales" / "unbuilt", repo_root)
    assert not i18n._is_inside_repository(tmp_path / "elsewhere" / "unbuilt", repo_root)


def test_pseudo_reports_a_file_typed_output_as_a_catalog_tool_error(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path, "")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output = tmp_path / "occupied"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(i18n.CatalogToolError, match="pseudo catalog"):
        i18n.generate_pseudo_catalog(
            output,
            source_root=source_root,
            repo_root=repo_root,
            location_root=source_root,
        )


def test_pseudo_generates_safe_loadable_plural_catalog_and_preserves_placeholders(tmp_path: Path) -> None:
    source_root = _source_root(
        tmp_path,
        "from chrys.foundation.i18n import msg\n"
        "FILES = msg(\n"
        "    'dialog.files',\n"
        "    fallback='One {name}',\n"
        "    plural_fallback='{count} files for {name}',\n"
        ")\n",
    )
    output = tmp_path / "pseudo-output"
    repo_root = tmp_path / "different-repository"
    repo_root.mkdir()

    mo_path = i18n.generate_pseudo_catalog(
        output,
        source_root=source_root,
        repo_root=repo_root,
        location_root=source_root,
    )

    assert mo_path == output / "en-XA" / "LC_MESSAGES" / "chrys.mo"
    with mo_path.open("rb") as stream:
        translations = gettext.GNUTranslations(stream)
    singular = translations.ngettext("dialog.files", "dialog.files#plural", 1)
    plural = translations.ngettext("dialog.files", "dialog.files#plural", 2)
    assert singular.startswith("«") and singular.endswith("··»")
    assert plural.startswith("«") and plural.endswith("··»")
    assert "{name}" in singular
    assert "{count}" in plural and "{name}" in plural
    assert "[" not in singular + plural and "]" not in singular + plural
    assert parse_placeholder_names(singular) == {"name"}
    assert parse_placeholder_names(plural) == {"count", "name"}
    validate_authored_template(singular, multiline=False)
    validate_authored_template(plural, multiline=False)


@CI_LINUX_ONLY
def test_live_catalog_artifacts_contain_effective_translations_and_are_loadable() -> None:
    pot = _read_catalog(i18n.POT_PATH)
    po = _read_catalog(i18n.ZH_PO_PATH)

    expected_ids = {
        ("attachments.attachment_error", "attachments.attachment_error#plural"),
        "attachments.history_vision_unsupported",
        "attachments.history_vision_unsupported_unnamed",
        "attachments.retry_history_image_unsupported",
        ("attachments.retry_image_unsupported", "attachments.retry_image_unsupported#plural"),
        ("attachments.vision_unsupported", "attachments.vision_unsupported#plural"),
        ("attachments.vision_unsupported_unnamed", "attachments.vision_unsupported_unnamed#plural"),
        "builder.memory_truncated",
        "builder.protected_chat_options_stripped",
        "construction.global_hooks_invalid",
        "construction.project_hooks_invalid",
        "construction.service_session_incompatible",
        "construction.trajectory_activation_failed",
        "controls.model_switch_not_ready",
        "controls.no_registry",
        "controls.profile_not_found",
        "controls.settings_restart_required",
        "controls.workspace_switch_not_ready",
        "coordinator.engine_not_started",
        "coordinator.interrupt_ignored_loading",
        "coordinator.prompt_admission_conflict",
        "engine.fork_agent_loading",
        "engine.fork_empty_session",
        "engine.fork_failed",
        "engine.fork_lock_not_owned",
        "engine.fork_not_ready",
        "engine.fork_prepare_failed",
        "engine.fork_prepare_timeout",
        "engine.fork_restoring",
        "engine.fork_session_changed",
        "engine.fork_session_not_found",
        "engine.fork_state_store_missing",
        "engine.fork_turn_active",
        ("history_markers.awaiting_sub_agents", "history_markers.awaiting_sub_agents#plural"),
        "history_markers.execution_failed",
        "history_markers.execution_interrupted",
        "history_markers.session_closed",
        "history_markers.sub_agent_state_discarded",
        "hosted_tools.failure_fallback",
        "hosted_tools.title_code",
        "hosted_tools.title_fetch",
        "hosted_tools.title_file_operation",
        "hosted_tools.title_generic",
        "hosted_tools.title_image",
        "hosted_tools.title_mcp",
        "hosted_tools.title_search",
        "hosted_tools.title_shell",
        "hosted_tools.title_tool_discovery",
        "i18n.catalog_load_failed",
        "prompt_content.image_attachment_while_running",
        "restore.agent_profile_unresolved",
        "restore.agent_profile_unresolved_using_current",
        "restore.requested_agent_profile_unresolved",
        "restore.service_session_incompatible",
        "restore.sub_agents_discarded",
        "retry.invalid_response",
        "retry.last_words_compaction",
        "retry.missing_user_anchor",
        "retry.stream_stalled",
        "retry.sub_agent_paused",
        "rollback.conversation_advanced",
        "rollback.conversation_changed",
        "rollback.no_session",
        "rollback.refused",
        "rollback.relative_turns_invalid",
        "rollback.reset_failed",
        "rollback.runtime_changed",
        "rollback.session_changed",
        "rollback.snapshot_missing",
        "rollback.swap_failed",
        "rollback.swap_locked",
        "rollback.target_turn_invalid",
        "rollback.turn_unavailable",
        "rollback.turns_unavailable",
        "run.headless_timeout",
        "run.interrupted",
        "run.model_profile_not_found",
        "run.model_profile_not_found_with_available",
        "session_host.agent_profile_not_found",
        "session_host.agent_profile_not_found_with_available",
        "session_host.headless_interaction_required",
        "session_host.no_final_response",
        "session_host.session_id_ambiguous",
        "session_host.session_not_found",
        "session_host.session_not_found_with_recent",
        "settings.max_transient_retries_clamped",
        "settings.max_transient_retries_invalid",
        "settings.clamped.maximum",
        "settings.clamped.minimum",
        "settings.project_config_dormant",
        "settings.rejected.bool",
        "settings.rejected.choice",
        "settings.rejected.directory",
        "settings.rejected.finite_number",
        "settings.rejected.int",
        "settings.rejected.non_negative_int",
        "settings.rejected.number",
        "settings.rejected.project_key",
        "settings.rejected.project_loosens",
        "settings.rejected.text",
        "settings.unknown_keys",
        "startup.invalid_no_proxy_env",
        "startup.notifications_migration_failed",
        "startup.raw_http_capture_on",
        "startup.settings_migration_failed",
        "transcript.interrupted.continue_action",
        "transcript.interrupted.error_header",
        "transcript.interrupted.reason_by_user",
        "transcript.interrupted.reason_plain",
        "transcript.interrupted.retry_action",
        "transcript.interrupted.warning_header",
        "tui.acp.connection_test.subject",
        "tui.agent_config.already_default",
        "tui.agent_config.deleted",
        "tui.agent_config.last_agent",
        "tui.agent_config.last_main_agent",
        "tui.agent_config.more_suffix",
        "tui.agent_config.removed",
        "tui.agent_config.reset_complete",
        "tui.agent_config.reset_pending",
        "tui.agent_config.saved",
        "tui.agent_config.still_referenced",
        "tui.agent_config.title.agent",
        "tui.agent_config.title.save_error",
        "tui.agent_config.title.settings",
        "tui.agent_config.title.validation_error",
        "tui.agent_load.button.ok",
        "tui.agent_load.finish.agent_ready",
        "tui.agent_load.finish.profile_switched",
        "tui.agent_load.finish.session_ready",
        "tui.agent_load.finish.session_restored",
        "tui.agent_load.flow.applying_agent_changes",
        "tui.agent_load.flow.checking_session_availability",
        "tui.agent_load.flow.failed",
        "tui.agent_load.flow.loading_agent",
        "tui.agent_load.flow.preparing_agent",
        "tui.agent_load.flow.preparing_session",
        "tui.agent_load.flow.restoring_session_history",
        "tui.agent_load.flow.restoring_session_history_percent",
        "tui.agent_load.flow.session_availability_checked",
        "tui.agent_load.progress.with_count",
        "tui.agent_load.progress.with_count_failed",
        "tui.agent_load.result.agent_loaded",
        "tui.agent_load.result.unable_to_load",
        "tui.agent_load.stage.agent_finalized",
        "tui.agent_load.stage.builtin_tools_loaded",
        "tui.agent_load.stage.capturing_workspace_context",
        "tui.agent_load.stage.checking_session_availability",
        "tui.agent_load.stage.connected_mcp_server",
        "tui.agent_load.stage.connecting_mcp_server",
        "tui.agent_load.stage.connecting_mcp_servers",
        "tui.agent_load.stage.failed_mcp_server",
        "tui.agent_load.stage.finalizing_agent",
        "tui.agent_load.stage.loaded_sub_agent",
        "tui.agent_load.stage.loading_builtin_tools",
        "tui.agent_load.stage.loading_mcp_server",
        "tui.agent_load.stage.loading_skills",
        "tui.agent_load.stage.loading_sub_agent",
        "tui.agent_load.stage.loading_sub_agents",
        "tui.agent_load.stage.mcp_servers_connected",
        "tui.agent_load.stage.mcp_servers_failed",
        "tui.agent_load.stage.model_profile_resolved",
        "tui.agent_load.stage.preparing_agent",
        "tui.agent_load.stage.resolving_model_profile",
        "tui.agent_load.stage.session_availability_checked",
        "tui.agent_load.stage.skills_loaded",
        "tui.agent_load.stage.skipped_sub_agent",
        "tui.agent_load.stage.skipped_sub_agent_reason",
        "tui.agent_load.stage.sub_agents_loaded",
        "tui.agent_load.status.count_failed",
        "tui.agent_load.title.initializing",
        "tui.agent_load.title.loading",
        "tui.agent_load.title.reloading",
        "tui.agent_load.title.resetting_session",
        "tui.agent_load.title.restoring_session",
        "tui.agent_load.title.starting_session",
        "tui.agent_load.title.switching",
        "tui.agent_load.title.workspace",
        "tui.approval.button.approve",
        "tui.approval.button.decline",
        "tui.approval.evaluating",
        "tui.approval.file_edit.content",
        "tui.approval.file_edit.planned_diff",
        "tui.approval.file_edit.prepare_diff_error",
        "tui.approval.file_edit.preparing_diff",
        ("tui.approval.file_edit.replacements", "tui.approval.file_edit.replacements#plural"),
        "tui.approval.flagged",
        "tui.approval.mode_changed",
        "tui.approval.presentation.edit_files",
        "tui.approval.presentation.read_files",
        "tui.approval.presentation.remote_tool",
        "tui.approval.presentation.run_command",
        "tui.approval.presentation.search",
        "tui.approval.reason_placeholder",
        "tui.approval.required_title",
        "tui.approval.sub_agent.detail",
        "tui.approval.sub_agent.prompt_title",
        "tui.approval.sub_agent.review",
        "tui.approval.title",
        "tui.approval_mode.description.auto",
        "tui.approval_mode.description.bypass",
        "tui.approval_mode.description.manual",
        "tui.approval_mode.title",
        "tui.ask_user.button.answer_inline",
        "tui.ask_user.button.submit",
        "tui.ask_user.title",
        "tui.binding.agents",
        "tui.binding.back",
        "tui.binding.break_agent",
        "tui.binding.cancel",
        "tui.binding.chat_page_down",
        "tui.binding.chat_page_up",
        "tui.binding.chat_scroll_bottom",
        "tui.binding.close",
        "tui.binding.copy",
        "tui.binding.cycle_level",
        "tui.binding.delete",
        "tui.binding.logs",
        "tui.binding.models",
        "tui.binding.paste",
        "tui.binding.quit",
        "tui.binding.trajectory",
        "tui.binding.sessions",
        "tui.binding.sidebar",
        "tui.binding.themes",
        "tui.binding.toggle_change_list",
        "tui.binding.toggle_check",
        "tui.binding.toggle_graph",
        "tui.binding.toggle_split_view",
        "tui.binding.toggle_view",
        "tui.buddy.hums",
        "tui.buddy.hatch_already",
        "tui.buddy.hatch_success",
        "tui.buddy.intro_no_buddy",
        "tui.buddy.intro_with_buddy",
        "tui.buddy.muted",
        "tui.buddy.mute_off",
        "tui.buddy.mute_on",
        "tui.buddy.name_empty",
        "tui.buddy.name_success",
        "tui.buddy.no_buddy_to_pet",
        "tui.buddy.not_hatched",
        "tui.buddy.pet_love",
        "tui.buddy.ponders",
        "tui.buddy.subcommand.hatch",
        "tui.buddy.subcommand.info",
        "tui.buddy.subcommand.mute",
        "tui.buddy.subcommand.name",
        "tui.buddy.subcommand.pet",
        "tui.buddy.thinking",
        "tui.buddy.thinking_sounds",
        "tui.buddy.title",
        "tui.buddy.unknown",
        "tui.chat.scroll_to_bottom",
        "tui.chat.scroll_to_bottom_tooltip",
        "tui.chat.session_json.loading",
        "tui.chat.session_json.not_found",
        "tui.chat.session_json.path_copied",
        "tui.chat.session_json.read_error",
        "tui.chat.session_json.session_title",
        "tui.chat.session_json.title",
        "tui.chat.session_title",
        "tui.chrome.approval_mode.auto",
        "tui.chrome.approval_mode.badge",
        "tui.chrome.approval_mode.bypass",
        "tui.chrome.approval_mode.manual",
        "tui.commands.description.longrun",
        "tui.commands.description.quick",
        "tui.commands.description.route",
        "tui.commands.title.disabled",
        "tui.commands.title.invalid",
        "tui.commands.title.man_page",
        "tui.commands.agents_target.basic",
        "tui.commands.agents_target.compaction",
        "tui.commands.agents_target.instructions",
        "tui.commands.agents_target.mcp",
        "tui.commands.agents_target.memory",
        "tui.commands.agents_target.skills",
        "tui.commands.agents_target.sub_agents",
        "tui.commands.agents_target.tools",
        "tui.commands.approval.manual",
        "tui.commands.description.agents",
        "tui.commands.description.approval",
        "tui.commands.description.buddy",
        "tui.commands.description.chdir",
        "tui.commands.description.clear",
        "tui.commands.description.copy",
        "tui.commands.description.diff",
        "tui.commands.description.exit",
        "tui.commands.description.fold",
        "tui.commands.description.fork",
        "tui.commands.description.language",
        "tui.commands.description.man",
        "tui.commands.description.models",
        "tui.commands.description.new",
        "tui.commands.description.rename",
        "tui.commands.description.resume",
        "tui.commands.description.rollback",
        "tui.commands.description.runtime",
        "tui.commands.description.sessions",
        "tui.commands.description.theme",
        "tui.commands.man.command",
        "tui.commands.unavailable_while_running",
        "tui.commands.unknown_agents_target",
        "tui.commands.unknown_command",
        "tui.compaction.copy.summary",
        "tui.compaction.copy.summary_unavailable",
        "tui.config.agent.loading",
        "tui.config.agent.no_main_profiles",
        "tui.config.agent.switched",
        "tui.config.model.loading",
        "tui.config.model.updated",
        "tui.config.title.agent",
        "tui.config.title.busy",
        "tui.config.title.model_settings",
        "tui.config.title.settings",
        "tui.config.updated",
        "tui.confirm.button.cancel",
        "tui.confirm.button.confirm",
        "tui.confirm.message",
        "tui.confirm.title",
        "tui.connection_test.button.ok",
        "tui.connection_test.failed",
        "tui.connection_test.success",
        "tui.connection_test.testing",
        "tui.context_pressure.conversation.disabled",
        "tui.context_pressure.conversation.generation_failure",
        "tui.context_pressure.conversation.generic",
        "tui.context_pressure.conversation.internal_limit",
        "tui.context_pressure.conversation.no_progress",
        "tui.context_pressure.conversation.round_limit",
        "tui.context_pressure.conversation.side_call_budget",
        "tui.context_pressure.conversation.spill_abort",
        "tui.context_pressure.sub_agent.disabled",
        "tui.context_pressure.sub_agent.generation_failure",
        "tui.context_pressure.sub_agent.generic",
        "tui.context_pressure.sub_agent.internal_limit",
        "tui.context_pressure.sub_agent.no_progress",
        "tui.context_pressure.sub_agent.round_limit",
        "tui.context_pressure.sub_agent.side_call_budget",
        "tui.context_pressure.sub_agent.spill_abort",
        "tui.copy.agent_response",
        ("tui.copy.copied.messages", "tui.copy.copied.messages#plural"),
        ("tui.copy.copied.responses", "tui.copy.copied.responses#plural"),
        ("tui.copy.copied.user_turns", "tui.copy.copied.user_turns#plural"),
        "tui.copy.empty.agent_responses",
        "tui.copy.empty.messages",
        "tui.copy.empty.user_turns",
        "tui.copy.invalid_argument",
        "tui.copy.title.action",
        "tui.copy.title.success",
        "tui.debug.copy.event_stream",
        "tui.diff.no_file_changes",
        "tui.diff.title",
        "tui.editor.button.commit",
        "tui.editor.button.discard",
        "tui.editor.character_limit",
        "tui.editor.draft_changed_warning",
        "tui.editor.draft_too_large",
        "tui.editor.hint.command",
        "tui.editor.hint.commit",
        "tui.editor.hint.discard",
        "tui.editor.hint.edit",
        "tui.editor.hint.mode",
        "tui.editor.hint.select",
        "tui.editor.hint.visual",
        "tui.editor.input_unavailable",
        "tui.editor.mode.emacs",
        "tui.editor.mode.standard",
        "tui.editor.mode.vim",
        "tui.editor.mode_label",
        "tui.editor.paste_truncated",
        "tui.editor.status_label",
        "tui.editor.title",
        "tui.editor.title.character_limit",
        "tui.editor.title.draft_too_large",
        "tui.editor.title.paste_truncated",
        "tui.editor.unsaved_warning",
        "tui.file_picker.button.cancel",
        "tui.file_picker.button.select",
        "tui.file_picker.recent",
        "tui.file_picker.tab.drives",
        "tui.file_picker.tab.favorites",
        "tui.file_picker.title.file",
        "tui.file_picker.title.folder",
        "tui.fork_session.button.ok",
        "tui.fork_session.button.open_new_window",
        "tui.fork_session.button.stay",
        "tui.fork_session.button.switch",
        "tui.fork_session.created",
        "tui.fork_session.creating",
        "tui.fork_session.title.error",
        "tui.fork_session.title.loading",
        "tui.fork_session.title.success",
        "tui.hosted.arguments",
        ("tui.hosted.artifact.more", "tui.hosted.artifact.more#plural"),
        "tui.hosted.artifact.unnamed",
        "tui.hosted.artifacts",
        "tui.hosted.code.empty",
        "tui.hosted.code.images",
        "tui.hosted.code.input",
        "tui.hosted.code.stderr",
        "tui.hosted.code.stdout",
        "tui.hosted.code.title",
        ("tui.hosted.discovery.discovered_count", "tui.hosted.discovery.discovered_count#plural"),
        "tui.hosted.discovery.discovered_tools",
        "tui.hosted.discovery.empty",
        "tui.hosted.empty.no_output",
        "tui.hosted.files",
        "tui.hosted.image.empty",
        ("tui.hosted.image.partial_preview", "tui.hosted.image.partial_preview#plural"),
        "tui.hosted.image.prompt",
        "tui.hosted.mcp.empty",
        "tui.hosted.output",
        "tui.hosted.query",
        "tui.hosted.result",
        "tui.hosted.search.citations",
        "tui.hosted.search.empty",
        "tui.hosted.search.queries",
        "tui.hosted.search.query_action",
        ("tui.hosted.search.result_count", "tui.hosted.search.result_count#plural"),
        "tui.hosted.search.results",
        "tui.hosted.shell.commands",
        "tui.hosted.shell.empty",
        "tui.hosted.shell.exit",
        "tui.hosted.shell.no",
        "tui.hosted.shell.status.timed_out",
        "tui.hosted.shell.stderr",
        "tui.hosted.shell.stdout",
        "tui.hosted.shell.timed_out",
        "tui.hosted.shell.yes",
        "tui.hosted.status.completed",
        "tui.hosted.status.failed",
        "tui.hosted.status.interrupted",
        "tui.hosted.status.running",
        ("tui.image_compression.title", "tui.image_compression.title#plural"),
        "tui.input.continue",
        "tui.input.hint.newline",
        "tui.input.interrupt",
        "tui.input.new",
        "tui.input.placeholder.agents",
        "tui.input.placeholder.commands",
        "tui.input.placeholder.files",
        "tui.input.placeholder.inject",
        "tui.input.placeholder.models",
        "tui.input.placeholder.prompt",
        "tui.input.placeholder.shell",
        "tui.input.queued",
        "tui.input.retry",
        "tui.input.send",
        "tui.language_command.unknown_locale",
        "tui.language_picker.english",
        "tui.language_picker.simplified_chinese",
        "tui.language_picker.system",
        "tui.language_picker.title",
        "tui.loading.label",
        ("tui.logs.copy.copied_lines", "tui.logs.copy.copied_lines#plural"),
        "tui.logs.copy.no_active_tab",
        "tui.logs.copy.no_logs",
        "tui.logs.copy.title",
        "tui.logs.empty.filtered_module",
        "tui.logs.empty.module",
        "tui.logs.level_hint",
        "tui.logs.title",
        "tui.long_horizon.phase",
        "tui.main.clear.empty_session",
        "tui.main.clear.no_active_session",
        "tui.main.confirm.clear",
        "tui.main.confirm.clear_message",
        "tui.main.confirm.clear_title",
        "tui.main.confirm.exit",
        "tui.main.confirm.exit_message",
        "tui.main.confirm.interrupt",
        "tui.main.confirm.interrupt_message",
        "tui.main.session_in_use.message",
        "tui.main.session_in_use.ok",
        "tui.main.session_in_use.title",
        "tui.man.agents.body",
        "tui.man.aliases_none",
        "tui.man.approval.body",
        "tui.man.approval.mode_hint",
        "tui.man.buddy.body",
        "tui.man.chdir.body",
        "tui.man.clear.body",
        "tui.man.copy.body",
        "tui.man.copy.useful_for",
        "tui.man.diff.body",
        "tui.man.example.show_all",
        "tui.man.example.show_diff",
        "tui.man.example.show_theme",
        "tui.man.examples_label",
        "tui.man.exit.body",
        "tui.man.fold.body",
        "tui.man.fork.body",
        "tui.man.heading.aliases",
        "tui.man.heading.available_commands",
        "tui.man.heading.description",
        "tui.man.heading.name",
        "tui.man.heading.options",
        "tui.man.heading.see_also",
        "tui.man.heading.synopsis",
        "tui.man.index.description",
        "tui.man.index.name",
        "tui.man.index.see_also",
        "tui.man.language.body",
        "tui.man.longrun.body",
        "tui.man.man.body",
        "tui.man.models.body",
        "tui.man.new.body",
        "tui.man.option.copy_agent",
        "tui.man.option.copy_all",
        "tui.man.option.copy_count",
        "tui.man.option.copy_user",
        "tui.man.option.rollback_count",
        "tui.man.option.rollback_target",
        "tui.man.options.no_additional_options",
        "tui.man.options.supports_subcommands",
        "tui.man.quick.body",
        "tui.man.rename.body",
        "tui.man.resume.body",
        "tui.man.rollback.body",
        "tui.man.route.body",
        "tui.man.runtime.body",
        "tui.man.sessions.body",
        "tui.man.theme.body",
        "tui.man_page.footer",
        "tui.markdown.copy.code_block",
        "tui.mcp.connection_test.subject",
        "tui.mcp.title",
        "tui.mcp.wait_before_adding",
        "tui.mcp.wait_before_removing",
        "tui.model_config.saved",
        "tui.model_config.title.save_error",
        "tui.model_config.title.settings",
        "tui.model_config.title.validation_error",
        "tui.notifications.event.approval_required.description",
        "tui.notifications.event.approval_required.label",
        "tui.notifications.event.ask_user.description",
        "tui.notifications.event.ask_user.label",
        "tui.notifications.event.turn_complete.description",
        "tui.notifications.event.turn_complete.label",
        "tui.notifications.event.turn_error.description",
        "tui.notifications.event.turn_error.label",
        "tui.notifications.save_failed",
        "tui.notifications.test_failed",
        "tui.notifications.test_sent",
        "tui.notifications.title",
        "tui.requirement_clarification.baseline_provisional",
        "tui.requirement_clarification.clarification",
        "tui.requirement_clarification.finalizing",
        "tui.requirement_clarification.initial",
        "tui.requirement_clarification.repair",
        "tui.requirement_clarification.snapshot",
        "tui.rollback.active_session_changed",
        "tui.rollback.agent_running",
        "tui.rollback.button.close",
        "tui.rollback.button.discard_after_turn",
        "tui.rollback.button.discard_all",
        "tui.rollback.exclusion.contested",
        "tui.rollback.exclusion.foreign",
        "tui.rollback.exclusion.move_poisoned",
        "tui.rollback.exclusion.peer_modified_since",
        "tui.rollback.exclusion.preview_item",
        "tui.rollback.exclusion.unrestorable",
        "tui.rollback.exclusions.more",
        "tui.rollback.failed",
        ("tui.rollback.files_excluded_note", "tui.rollback.files_excluded_note#plural"),
        "tui.rollback.no_file_changes",
        "tui.rollback.no_snapshots",
        "tui.rollback.picker.discard_all_suffix",
        "tui.rollback.picker.prompt",
        "tui.rollback.picker.session_start",
        "tui.rollback.picker.turn",
        "tui.rollback.picker.turn_current",
        "tui.rollback.picker.turn_current_preview",
        "tui.rollback.picker.turn_preview",
        "tui.rollback.picker_conversation_changed",
        "tui.rollback.picker_runtime_changed",
        "tui.rollback.preview.load_error",
        "tui.rollback.preview.refresh_error",
        "tui.rollback.result.already_matched",
        "tui.rollback.result.error_reason",
        "tui.rollback.result.failed_count",
        "tui.rollback.result.failed_preview",
        ("tui.rollback.result.files_excluded", "tui.rollback.result.files_excluded#plural"),
        ("tui.rollback.result.files_restored", "tui.rollback.result.files_restored#plural"),
        ("tui.rollback.result.files_reverted", "tui.rollback.result.files_reverted#plural"),
        "tui.rollback.result.more_suffix",
        "tui.rollback.result.session_start",
        "tui.rollback.result.turn",
        ("tui.rollback.result.warnings_count", "tui.rollback.result.warnings_count#plural"),
        ("tui.rollback.revert_file_changes", "tui.rollback.revert_file_changes#plural"),
        "tui.rollback.target_nonnegative",
        "tui.rollback.title",
        "tui.rollback.turn_count_positive",
        "tui.rollback.unavailable",
        "tui.rollback.usage",
        "tui.rollback.warning_note",
        "tui.route.announcement",
        "tui.route.downgrade_hint",
        "tui.route.reroute_queued",
        "tui.route.status.title",
        "tui.route.unknown_argument",
        "tui.runtime_details.boolean.off",
        "tui.runtime_details.boolean.on",
        "tui.runtime_details.empty.entries",
        "tui.runtime_details.empty.exposed_tools",
        "tui.runtime_details.empty.hooks",
        "tui.runtime_details.empty.hooks_in_source",
        "tui.runtime_details.empty.mcp_tools",
        "tui.runtime_details.empty.model_profile",
        "tui.runtime_details.empty.preconfigured_files",
        "tui.runtime_details.empty.skills",
        "tui.runtime_details.empty.tools",
        "tui.runtime_details.files",
        "tui.runtime_details.hooks",
        "tui.runtime_details.label.api_style",
        "tui.runtime_details.label.base_url",
        "tui.runtime_details.label.context",
        "tui.runtime_details.label.description",
        "tui.runtime_details.label.enabled",
        "tui.runtime_details.label.event",
        "tui.runtime_details.label.mode",
        "tui.runtime_details.label.model_id",
        "tui.runtime_details.label.name",
        "tui.runtime_details.label.profile_id",
        "tui.runtime_details.label.provider",
        "tui.runtime_details.label.streaming",
        "tui.runtime_details.label.vision",
        "tui.runtime_details.mcp",
        "tui.runtime_details.model",
        "tui.runtime_details.section.auto_loaded_files",
        "tui.runtime_details.section.failed_mcp_servers",
        "tui.runtime_details.section.global_hooks",
        "tui.runtime_details.section.inline_skills",
        "tui.runtime_details.section.mcp_servers",
        "tui.runtime_details.section.model_profile",
        "tui.runtime_details.section.project_hooks",
        "tui.runtime_details.section.sub_agent_tools",
        "tui.runtime_details.skills",
        "tui.runtime_details.title",
        "tui.runtime_details.token_count",
        "tui.runtime_details.tools",
        "tui.session.clear.failed",
        "tui.session.fork.created",
        "tui.session.fork.dialog_failed",
        "tui.session.fork.empty_session",
        "tui.session.fork.no_active_session",
        "tui.session.fork.turn_running",
        "tui.session.resume.none",
        "tui.session.title.fork",
        "tui.session.title.resume",
        "tui.session.title.save_failed",
        "tui.session.title.session_title",
        "tui.session_title.button.cancel",
        "tui.session_title.button.save",
        "tui.session_title.hint",
        "tui.session_title.placeholder",
        "tui.session_title.title",
        "tui.sessions.button.close",
        "tui.sessions.button.delete",
        "tui.sessions.button.resume",
        "tui.sessions.column.directory",
        "tui.sessions.column.last_active",
        "tui.sessions.column.session_id",
        "tui.sessions.column.size",
        "tui.sessions.column.title",
        "tui.sessions.column.turns",
        "tui.sessions.count",
        "tui.sessions.delete.confirm_message",
        "tui.sessions.delete.open_elsewhere",
        "tui.sessions.empty",
        "tui.sessions.loading",
        "tui.sessions.search_placeholder",
        ("tui.sessions.time.days_ago", "tui.sessions.time.days_ago#plural"),
        "tui.sessions.time.hours_ago",
        "tui.sessions.time.just_now",
        "tui.sessions.time.minutes_ago",
        ("tui.sessions.time.months_ago", "tui.sessions.time.months_ago#plural"),
        ("tui.sessions.time.years_ago", "tui.sessions.time.years_ago#plural"),
        "tui.sessions.title",
        "tui.sessions.title.delete",
        "tui.sessions.tooltip.agent",
        "tui.sessions.tooltip.directory",
        "tui.sessions.tooltip.forked_from",
        "tui.sessions.tooltip.last_interaction",
        "tui.sessions.tooltip.prompt_match",
        "tui.sessions.tooltip.title",
        "tui.sessions.tooltip.total_tokens",
        "tui.sessions.tooltip.turns",
        "tui.sidebar.buddy.click_to_pet",
        "tui.sidebar.buddy.empty",
        "tui.sidebar.buddy.level",
        "tui.sidebar.buddy.notifications_muted",
        "tui.sidebar.buddy.personality",
        "tui.sidebar.buddy.petting",
        "tui.sidebar.buddy.rarity",
        "tui.sidebar.buddy.shiny",
        "tui.sidebar.buddy.species",
        ("tui.sidebar.context.block_messages", "tui.sidebar.context.block_messages#plural"),
        "tui.sidebar.context.cached",
        "tui.sidebar.context.compressed_messages",
        "tui.sidebar.context.input",
        "tui.sidebar.context.output",
        "tui.sidebar.context.token_usage",
        "tui.sidebar.context.total",
        "tui.sidebar.context.turn_range",
        "tui.sidebar.context.turn_single",
        "tui.sidebar.context.usage",
        "tui.sidebar.debug.event_stream",
        "tui.sidebar.tab.buddy",
        "tui.sidebar.tab.context",
        "tui.sidebar.tab.debug",
        "tui.sidebar.tab.messages",
        "tui.sidebar.tab.tasks",
        "tui.sidebar.tasks.counter",
        "tui.sidebar.tasks.empty",
        "tui.sidebar.tasks.title",
        "tui.sidebar.toc.empty",
        "tui.startup.session.not_found",
        "tui.startup.session.read_failed",
        "tui.startup.session.restore_failed",
        "tui.startup.session.restore_fallback",
        "tui.startup.title.agent_failed",
        "tui.startup.title.session",
        "tui.status.agent_load_failed",
        "tui.status.agent_selector_label",
        "tui.status.agent_startup_failed",
        "tui.status.compacting",
        "tui.status.completed",
        "tui.status.custom_title_cleared",
        "tui.status.details_tooltip",
        "tui.status.elapsed_minutes_seconds",
        "tui.status.elapsed_seconds",
        "tui.status.error",
        "tui.status.fork_created",
        "tui.status.fork_notice",
        "tui.status.interactive_mode",
        "tui.status.interrupted",
        "tui.status.model_selector_label",
        "tui.status.opened_fork",
        "tui.status.resuming",
        "tui.status.retrying",
        "tui.status.running_tool",
        ("tui.status.runtime_files", "tui.status.runtime_files#plural"),
        ("tui.status.runtime_hooks", "tui.status.runtime_hooks#plural"),
        ("tui.status.runtime_skills", "tui.status.runtime_skills#plural"),
        ("tui.status.runtime_tools", "tui.status.runtime_tools#plural"),
        "tui.status.session_restored",
        "tui.status.session_title_updated",
        "tui.status.shell_mode",
        "tui.status.streaming",
        "tui.status.thinking",
        ("tui.status.tool_calls", "tui.status.tool_calls#plural"),
        "tui.suggestions.agents_title",
        "tui.suggestions.bounded_index",
        "tui.suggestions.commands_title",
        "tui.suggestions.files_title",
        "tui.suggestions.index_counts",
        "tui.suggestions.loaded_skills",
        "tui.suggestions.models_title",
        "tui.suggestions.no_results",
        "tui.suggestions.prompt_history_title",
        "tui.suggestions.shadowed",
        "tui.suggestions.system_commands",
        "tui.tool.copy.details",
        "tui.tool.copy.details_too_large",
        "tui.tool.copy.details_unavailable",
        "tui.tool_card.action.copy",
        "tui.tool_card.action.copy_tooltip",
        "tui.tool_card.action.view",
        "tui.tool_card.action.view_tooltip",
        "tui.tool_card.ask_user.answer",
        "tui.tool_card.ask_user.question",
        "tui.tool_card.ask_user.waiting",
        "tui.tool_card.execute.no_command",
        "tui.tool_card.execute.output",
        "tui.tool_card.execute.timeout",
        "tui.tool_card.file_edit.overwrite",
        "tui.tool_card.file_edit.status.change_counts",
        "tui.tool_card.file_edit.status.changes",
        "tui.tool_card.file_edit.editing",
        "tui.tool_card.file_edit.status.error",
        "tui.tool_card.file_edit.status.lines_written",
        "tui.tool_card.file_edit.writing",
        "tui.tool_card.group.title",
        "tui.tool_card.group.title_timed",
        "tui.tool_card.image.resolution",
        "tui.tool_card.image.type",
        "tui.tool_card.read_file.error_suffix",
        ("tui.tool_card.read_file.line_count", "tui.tool_card.read_file.line_count#plural"),
        "tui.tool_card.read_file.line_range",
        "tui.tool_card.read_file.reading",
        "tui.tool_card.search.error_suffix",
        "tui.tool_card.search.searching",
        ("tui.tool_card.skill.char_count", "tui.tool_card.skill.char_count#plural"),
        "tui.tool_card.skill.completed",
        "tui.tool_card.skill.dir",
        "tui.tool_card.skill.empty",
        "tui.tool_card.skill.empty_instructions",
        "tui.tool_card.skill.empty_preview",
        "tui.tool_card.skill.error",
        "tui.tool_card.skill.exit",
        ("tui.tool_card.skill.line_count", "tui.tool_card.skill.line_count#plural"),
        "tui.tool_card.skill.loaded",
        "tui.tool_card.skill.loading",
        "tui.tool_card.skill.more",
        "tui.tool_card.skill.noun.resource",
        "tui.tool_card.skill.noun.script",
        "tui.tool_card.skill.noun.skill",
        "tui.tool_card.skill.reading_resource",
        ("tui.tool_card.skill.resource_count", "tui.tool_card.skill.resource_count#plural"),
        "tui.tool_card.skill.resources",
        "tui.tool_card.skill.running_script",
        ("tui.tool_card.skill.script_count", "tui.tool_card.skill.script_count#plural"),
        "tui.tool_card.skill.scripts",
        "tui.tool_card.skill.truncated_lines",
        "tui.tool_card.sleep.invalid_duration",
        "tui.tool_card.sleep.remaining",
        "tui.tool_card.sleep.skip",
        "tui.tool_card.sleep.sleeping",
        "tui.tool_card.sleep.title",
        "tui.tool_card.status.approved",
        "tui.tool_card.status.completed",
        "tui.tool_card.status.errored",
        "tui.tool_card.status.errored_with_code",
        "tui.tool_card.status.interrupted",
        "tui.tool_card.status.rejected",
        "tui.tool_card.status.running",
        "tui.tool_card.status.skipped",
        ("tui.tool_card.sub_agent.after_retries", "tui.tool_card.sub_agent.after_retries#plural"),
        ("tui.tool_card.sub_agent.artifacts", "tui.tool_card.sub_agent.artifacts#plural"),
        "tui.tool_card.sub_agent.button.abort",
        "tui.tool_card.sub_agent.button.retry",
        "tui.tool_card.sub_agent.button.skip_sleep",
        "tui.tool_card.sub_agent.button.skip_sleep_tooltip",
        "tui.tool_card.sub_agent.compacted",
        "tui.tool_card.sub_agent.compacted_warning",
        "tui.tool_card.sub_agent.compacting",
        "tui.tool_card.sub_agent.compaction_failed",
        "tui.tool_card.sub_agent.compaction_failed_reason",
        "tui.tool_card.sub_agent.compaction_name",
        "tui.tool_card.sub_agent.compactions",
        "tui.tool_card.sub_agent.ctx_tokens",
        "tui.tool_card.sub_agent.diagnostics",
        "tui.tool_card.sub_agent.duration",
        ("tui.tool_card.sub_agent.images", "tui.tool_card.sub_agent.images#plural"),
        "tui.tool_card.sub_agent.paused",
        "tui.tool_card.sub_agent.reason.acp_interrupted",
        "tui.tool_card.sub_agent.reason.compaction_failed",
        "tui.tool_card.sub_agent.reason.failed",
        "tui.tool_card.sub_agent.reason.paused",
        "tui.tool_card.sub_agent.reason.stream_stalled",
        "tui.tool_card.sub_agent.rendering_markdown",
        "tui.tool_card.sub_agent.retrying",
        "tui.tool_card.sub_agent.spend",
        "tui.tool_card.sub_agent.task",
        "tui.tool_card.sub_agent.task_prompt",
        "tui.tool_card.sub_agent.tokens",
        "tui.tool_card.sub_agent.tool_calls",
        "tui.tool_card.sub_agent.tool_interrupted",
        "tui.tool_card.sub_agent.tool_skipped",
        ("tui.tool_card.sub_agent.unreported_attempts", "tui.tool_card.sub_agent.unreported_attempts#plural"),
        "tui.tool_card.sub_agent.zero_reported_tokens",
        "tui.tool_card.todo.progress_title",
        "tui.tool_card.todo.title",
        "tui.tool_view.copy.input",
        "tui.tool_view.copy.input_too_large",
        "tui.tool_view.copy.input_unavailable",
        "tui.tool_view.copy.output",
        "tui.tool_view.copy.output_too_large",
        "tui.tool_view.copy.output_unavailable",
        "tui.transcript.fold.compressed",
        "tui.vision_unsupported.button.ok",
        "tui.vision_unsupported.button.use_paths",
        "tui.vision_unsupported.title",
        "tui.vision_unsupported.title.not_attached",
        "tui.warning.title",
        "tui.workspace.change_directory_busy",
        "tui.workspace.invalid_directory",
        "tui.workspace.title.busy",
        "tui.workspace.title.invalid_path",
        "turn_hooks.prompt_blocked",
    }
    expected_ids.update(
        {
            "tui.acp.add_item",
            "tui.acp.allow_external_cwd",
            "tui.acp.allow_external_cwd_description",
            "tui.acp.argument",
            "tui.acp.arguments",
            "tui.acp.best_effort",
            "tui.acp.best_effort_description",
            "tui.acp.config_options",
            "tui.acp.description",
            "tui.acp.enabled",
            "tui.acp.environment_variables",
            "tui.acp.executable",
            "tui.acp.handshake_timeout",
            "tui.acp.idle_timeout",
            "tui.acp.model_id",
            "tui.acp.option",
            "tui.acp.option_type.boolean",
            "tui.acp.option_type.string",
            "tui.acp.placeholder.argument",
            "tui.acp.placeholder.executable",
            "tui.acp.placeholder.model_id",
            "tui.acp.placeholder.option_id",
            "tui.acp.placeholder.session_mode",
            "tui.acp.placeholder.value",
            "tui.acp.placeholder.variable_name",
            "tui.acp.placeholder.working_directory",
            "tui.acp.result",
            "tui.acp.result.full_transcript",
            "tui.acp.result.last_segment",
            "tui.acp.session_mode",
            "tui.acp.test",
            "tui.acp.test_note",
            "tui.acp.title",
            "tui.acp.unsafe_mode_caution",
            "tui.acp.variable",
            "tui.acp.working_directory",
            "tui.agent_config.basic.agent_profile",
            "tui.agent_config.basic.agent_profile_description",
            "tui.agent_config.basic.agent_type",
            "tui.agent_config.basic.agent_type_note",
            "tui.agent_config.basic.built_in",
            "tui.agent_config.basic.default_profile",
            "tui.agent_config.basic.description",
            "tui.agent_config.basic.display_name",
            "tui.agent_config.basic.display_name_placeholder",
            "tui.agent_config.basic.external_acp",
            "tui.agent_config.basic.model_profile",
            "tui.agent_config.basic.model_profile_description",
            "tui.agent_config.basic.name",
            "tui.agent_config.basic.profile_name_placeholder",
            "tui.agent_config.basic.sub_agent_only",
            "tui.agent_config.basic.sub_agent_only_description",
            "tui.agent_config.basic.use_active_model_profile",
            "tui.agent_config.basic.use_profile",
            "tui.agent_config.button.clone",
            "tui.agent_config.button.close",
            "tui.agent_config.button.delete",
            "tui.agent_config.button.new",
            "tui.agent_config.button.reset",
            "tui.agent_config.button.save",
            "tui.agent_config.compaction.description",
            "tui.agent_config.compaction.last_words_supplement",
            "tui.agent_config.compaction.max_output_tokens",
            "tui.agent_config.compaction.title",
            "tui.agent_config.confirm.delete.message",
            "tui.agent_config.confirm.delete.title",
            "tui.agent_config.confirm.reset.message",
            "tui.agent_config.confirm.reset.title",
            "tui.agent_config.instructions.description",
            "tui.agent_config.instructions.title",
            "tui.agent_config.memory.absolute_path_error",
            "tui.agent_config.memory.add_file",
            "tui.agent_config.memory.add_folder",
            "tui.agent_config.memory.current_path",
            "tui.agent_config.memory.description",
            "tui.agent_config.memory.empty_files",
            "tui.agent_config.memory.empty_folders",
            "tui.agent_config.memory.file",
            "tui.agent_config.memory.file_missing",
            "tui.agent_config.memory.file_placeholder.absolute_linux",
            "tui.agent_config.memory.file_placeholder.absolute_macos",
            "tui.agent_config.memory.file_placeholder.absolute_windows",
            "tui.agent_config.memory.file_placeholder.relative_posix",
            "tui.agent_config.memory.file_placeholder.relative_windows",
            "tui.agent_config.memory.files",
            "tui.agent_config.memory.folder",
            "tui.agent_config.memory.folder_missing",
            "tui.agent_config.memory.folder_placeholder.absolute_linux",
            "tui.agent_config.memory.folder_placeholder.absolute_macos",
            "tui.agent_config.memory.folder_placeholder.absolute_windows",
            "tui.agent_config.memory.folder_placeholder.relative_posix",
            "tui.agent_config.memory.folder_placeholder.relative_windows",
            "tui.agent_config.memory.folder_scan_note",
            "tui.agent_config.memory.folders",
            "tui.agent_config.memory.missing_in_workspace",
            "tui.agent_config.memory.preview_note",
            "tui.agent_config.memory.relative_path_error",
            "tui.agent_config.memory.resolved_at_runtime",
            "tui.agent_config.memory.select_file",
            "tui.agent_config.memory.select_folder",
            "tui.agent_config.memory.title",
            "tui.agent_config.panel_error.detail",
            "tui.agent_config.panel_error.mounting",
            "tui.agent_config.panel_error.read",
            "tui.agent_config.path_entry.browse",
            "tui.agent_config.path_entry.workspace_relative",
            "tui.agent_config.profile.modified",
            "tui.agent_config.read_only_notice",
            "tui.agent_config.skills.absolute_path_error",
            "tui.agent_config.skills.add",
            "tui.agent_config.skills.allowed_extensions",
            "tui.agent_config.skills.auto_load_covered",
            "tui.agent_config.skills.current_path",
            "tui.agent_config.skills.description",
            "tui.agent_config.skills.directory",
            "tui.agent_config.skills.empty",
            "tui.agent_config.skills.execution_timeout",
            "tui.agent_config.skills.load_user_folder",
            "tui.agent_config.skills.load_user_folder_tooltip",
            "tui.agent_config.skills.load_working_folder",
            "tui.agent_config.skills.load_working_folder_tooltip",
            "tui.agent_config.skills.missing_in_workspace",
            "tui.agent_config.skills.path_missing",
            "tui.agent_config.skills.placeholder.absolute_linux",
            "tui.agent_config.skills.placeholder.absolute_macos",
            "tui.agent_config.skills.placeholder.absolute_windows",
            "tui.agent_config.skills.placeholder.relative_posix",
            "tui.agent_config.skills.placeholder.relative_windows",
            "tui.agent_config.skills.preview_note",
            "tui.agent_config.skills.relative_path_error",
            "tui.agent_config.skills.resolved_at_runtime",
            "tui.agent_config.skills.title",
            "tui.agent_config.subagents.add",
            "tui.agent_config.subagents.default_profile_description",
            "tui.agent_config.subagents.default_profile_name",
            "tui.agent_config.subagents.defaults_to",
            "tui.agent_config.subagents.defaults_to_description",
            "tui.agent_config.subagents.description",
            "tui.agent_config.subagents.empty",
            "tui.agent_config.subagents.max_concurrency",
            "tui.agent_config.subagents.max_total_concurrency",
            "tui.agent_config.subagents.no_description",
            "tui.agent_config.subagents.profile",
            "tui.agent_config.subagents.sub_agent",
            "tui.agent_config.subagents.title",
            "tui.agent_config.subagents.tool_description",
            "tui.agent_config.subagents.tool_name",
            "tui.agent_config.tab.acp_settings",
            "tui.agent_config.tab.acp_short",
            "tui.agent_config.tab.basic",
            "tui.agent_config.tab.compaction",
            "tui.agent_config.tab.instructions",
            "tui.agent_config.tab.mcp",
            "tui.agent_config.tab.memory",
            "tui.agent_config.tab.skills",
            "tui.agent_config.tab.sub_agents",
            "tui.agent_config.tab.tools",
            "tui.agent_config.title.configuration",
            "tui.agent_config.tools.ask_user",
            "tui.agent_config.tools.ask_user_description",
            "tui.agent_config.tools.categories",
            "tui.agent_config.tools.categories_description",
            "tui.agent_config.tools.document_converter",
            "tui.agent_config.tools.document_converter_description",
            "tui.agent_config.tools.filesystem_read",
            "tui.agent_config.tools.filesystem_read_description",
            "tui.agent_config.tools.filesystem_write",
            "tui.agent_config.tools.filesystem_write_description",
            "tui.agent_config.tools.search",
            "tui.agent_config.tools.search_description",
            "tui.agent_config.tools.shell",
            "tui.agent_config.tools.shell_description",
            "tui.agent_config.tools.sleep",
            "tui.agent_config.tools.sleep_description",
            "tui.agent_config.tools.todo_list",
            "tui.agent_config.tools.todo_list_description",
            "tui.mcp.add",
            "tui.mcp.always_load_tooltip",
            "tui.mcp.bypass_proxy",
            "tui.mcp.command",
            "tui.mcp.description",
            "tui.mcp.empty",
            "tui.mcp.enabled",
            "tui.mcp.environment_variables",
            "tui.mcp.expose_instructions_hint",
            "tui.mcp.expose_instructions_tooltip",
            "tui.mcp.expose_server_instructions",
            "tui.mcp.expose_server_prompts",
            "tui.mcp.headers",
            "tui.mcp.http_options",
            "tui.mcp.initially_visible_tools",
            "tui.mcp.insecure_tls",
            "tui.mcp.load_prompts_hint",
            "tui.mcp.load_prompts_tooltip",
            "tui.mcp.loading_strategy",
            "tui.mcp.naming_and_limits",
            "tui.mcp.permitted_tool_set",
            "tui.mcp.placeholder.command",
            "tui.mcp.placeholder.header_name",
            "tui.mcp.placeholder.header_value",
            "tui.mcp.placeholder.initial_tools",
            "tui.mcp.placeholder.request_timeout",
            "tui.mcp.placeholder.selected_tools",
            "tui.mcp.placeholder.server_name",
            "tui.mcp.placeholder.tool_name_prefix",
            "tui.mcp.placeholder.url",
            "tui.mcp.placeholder.value",
            "tui.mcp.placeholder.variable_name",
            "tui.mcp.progressive_disclosure_tooltip",
            "tui.mcp.prompts_and_instructions",
            "tui.mcp.request_timeout",
            "tui.mcp.selected_tool_names",
            "tui.mcp.server",
            "tui.mcp.server_name",
            "tui.mcp.servers",
            "tui.mcp.servers_description",
            "tui.mcp.skip_tls_verification",
            "tui.mcp.stdio_options",
            "tui.mcp.test",
            "tui.mcp.testing",
            "tui.mcp.tool_access",
            "tui.mcp.tool_access.all",
            "tui.mcp.tool_access.none",
            "tui.mcp.tool_access.selected",
            "tui.mcp.tool_access_tooltip",
            "tui.mcp.tool_loading",
            "tui.mcp.tool_loading.full",
            "tui.mcp.tool_loading.progressive",
            "tui.mcp.tool_name_prefix",
            "tui.mcp.transport",
            "tui.mcp.url",
            "tui.model_config.api_key",
            "tui.model_config.api_style",
            "tui.model_config.base_url",
            "tui.model_config.button.add",
            "tui.model_config.button.clone",
            "tui.model_config.button.close",
            "tui.model_config.button.delete",
            "tui.model_config.button.new",
            "tui.model_config.button.save",
            "tui.model_config.bypass_proxy",
            "tui.model_config.chat_options",
            "tui.model_config.confirm_delete.message",
            "tui.model_config.confirm_delete.title",
            "tui.model_config.connection_options",
            "tui.model_config.extra_options",
            "tui.model_config.http_connect_timeout",
            "tui.model_config.http_extra_headers",
            "tui.model_config.http_max_retries",
            "tui.model_config.http_options",
            "tui.model_config.http_read_timeout",
            "tui.model_config.insecure_tls",
            "tui.model_config.max_context_window",
            "tui.model_config.max_output_tokens",
            "tui.model_config.model",
            "tui.model_config.model_options",
            "tui.model_config.placeholder.header_name",
            "tui.model_config.placeholder.header_value",
            "tui.model_config.placeholder.leave_blank_for_default",
            "tui.model_config.placeholder.model",
            "tui.model_config.placeholder.option_name",
            "tui.model_config.placeholder.option_value",
            "tui.model_config.placeholder.output_cap",
            "tui.model_config.placeholder.profile_name",
            "tui.model_config.profile_name",
            "tui.model_config.provider",
            "tui.model_config.provider_api_key",
            "tui.model_config.provider_base_url",
            "tui.model_config.provider_model",
            "tui.model_config.read_only_notice",
            "tui.model_config.skip_tls_verification",
            "tui.model_config.streaming",
            "tui.model_config.title.configuration",
            "tui.model_config.vision",
        }
    )
    expected_ids.update(
        {
            "tui.agent_config.validation.acp.allow_external_cwd_boolean",
            "tui.agent_config.validation.acp.arguments_nul",
            "tui.agent_config.validation.acp.arguments_strings",
            "tui.agent_config.validation.acp.best_effort_boolean",
            "tui.agent_config.validation.acp.config_keys",
            "tui.agent_config.validation.acp.config_mapping",
            "tui.agent_config.validation.acp.config_option_row_label",
            "tui.agent_config.validation.acp.config_value",
            "tui.agent_config.validation.acp.cwd_nul",
            "tui.agent_config.validation.acp.cwd_outside",
            "tui.agent_config.validation.acp.cwd_outside_enable",
            "tui.agent_config.validation.acp.cwd_string",
            "tui.agent_config.validation.acp.draft_handshake_timeout_label",
            "tui.agent_config.validation.acp.draft_idle_timeout_label",
            "tui.agent_config.validation.acp.environment_mapping",
            "tui.agent_config.validation.acp.environment_name_duplicate",
            "tui.agent_config.validation.acp.environment_name_invalid",
            "tui.agent_config.validation.acp.environment_name_reserved",
            "tui.agent_config.validation.acp.environment_row_label",
            "tui.agent_config.validation.acp.environment_value_nul",
            "tui.agent_config.validation.acp.environment_value_string",
            "tui.agent_config.validation.acp.executable_nul",
            "tui.agent_config.validation.acp.executable_required",
            "tui.agent_config.validation.acp.expanded_launch_nul",
            "tui.agent_config.validation.acp.field_string",
            "tui.agent_config.validation.acp.handshake_timeout_label",
            "tui.agent_config.validation.acp.idle_timeout_label",
            "tui.agent_config.validation.acp.launch_nul",
            "tui.agent_config.validation.acp.model_id_field",
            "tui.agent_config.validation.acp.no_builtin_tools",
            "tui.agent_config.validation.acp.no_custom_tools",
            "tui.agent_config.validation.acp.no_mcp",
            "tui.agent_config.validation.acp.no_nested_subagents",
            "tui.agent_config.validation.acp.result_mode",
            "tui.agent_config.validation.acp.session_mode_field",
            "tui.agent_config.validation.acp.timeout_number",
            "tui.agent_config.validation.acp.timeout_range",
            "tui.agent_config.validation.at_least_one_agent",
            "tui.agent_config.validation.at_least_one_main_agent",
            "tui.agent_config.validation.compaction_range",
            "tui.agent_config.validation.compaction_required",
            "tui.agent_config.validation.compaction_whole_number",
            "tui.agent_config.validation.context_error",
            "tui.agent_config.validation.duplicate_key_row",
            "tui.agent_config.validation.duplicate_skill_directory",
            "tui.agent_config.validation.duplicate_sub_agent_profile",
            "tui.agent_config.validation.duplicate_sub_agent_profile_lower",
            "tui.agent_config.validation.duplicate_sub_agent_tool",
            "tui.agent_config.validation.duplicate_sub_agent_tool_lower",
            "tui.agent_config.validation.field.description_lower",
            "tui.agent_config.validation.field.display_name",
            "tui.agent_config.validation.field.display_name_lower",
            "tui.agent_config.validation.field.instructions",
            "tui.agent_config.validation.field.instructions_lower",
            "tui.agent_config.validation.field.last_words_max_output_tokens",
            "tui.agent_config.validation.field.max_concurrency",
            "tui.agent_config.validation.field.max_total_concurrency",
            "tui.agent_config.validation.field.max_total_concurrency_lower",
            "tui.agent_config.validation.field.name",
            "tui.agent_config.validation.field.path",
            "tui.agent_config.validation.field.profile_name",
            "tui.agent_config.validation.field.profile_name_lower",
            "tui.agent_config.validation.field.script_timeout",
            "tui.agent_config.validation.field.script_timeout_lower",
            "tui.agent_config.validation.field_positive_integer",
            "tui.agent_config.validation.field_required",
            "tui.agent_config.validation.field_valid_integer",
            "tui.agent_config.validation.fields_required",
            "tui.agent_config.validation.fix_before_structural_change",
            "tui.agent_config.validation.greater_than_zero",
            "tui.agent_config.validation.item.name",
            "tui.agent_config.validation.item.tool_name",
            "tui.agent_config.validation.key_required_row",
            "tui.agent_config.validation.last_words_range",
            "tui.agent_config.validation.mcp.command_required",
            "tui.agent_config.validation.mcp.duplicate_server",
            "tui.agent_config.validation.mcp.duplicate_server_lower",
            "tui.agent_config.validation.mcp.initial_tools_permitted",
            "tui.agent_config.validation.mcp.initially_visible_tools_label",
            "tui.agent_config.validation.mcp.invalid_command",
            "tui.agent_config.validation.mcp.row_key_required",
            "tui.agent_config.validation.mcp.row_value_required",
            "tui.agent_config.validation.mcp.select_permitted",
            "tui.agent_config.validation.mcp.server_context",
            "tui.agent_config.validation.mcp.timeout_positive",
            "tui.agent_config.validation.mcp.timeout_valid",
            "tui.agent_config.validation.mcp.tool_list_duplicate",
            "tui.agent_config.validation.mcp.tool_list_empty",
            "tui.agent_config.validation.mcp.transport",
            "tui.agent_config.validation.mcp.url_required",
            "tui.agent_config.validation.mcp.url_scheme",
            "tui.agent_config.validation.memory_context",
            "tui.agent_config.validation.path_absolute_disable",
            "tui.agent_config.validation.path_relative_enable",
            "tui.agent_config.validation.paths_match_case_insensitive",
            "tui.agent_config.validation.paths_match_normalized",
            "tui.agent_config.validation.profile_already_exists",
            "tui.agent_config.validation.profile_does_not_exist",
            "tui.agent_config.validation.profile_name_format",
            "tui.agent_config.validation.profile_selection_required",
            "tui.agent_config.validation.select_model_profile",
            "tui.agent_config.validation.selected_model_missing",
            "tui.agent_config.validation.server_error",
            "tui.agent_config.validation.skill_directory_context",
            "tui.agent_config.validation.sub_agent_context",
            "tui.agent_config.validation.tool_name_identifier",
            "tui.agent_config.validation.zero_or_greater",
            "tui.model_config.validation.base_url_scheme",
            "tui.model_config.validation.chat_option.boolean",
            "tui.model_config.validation.chat_option.detail",
            "tui.model_config.validation.chat_option.header_reserved",
            "tui.model_config.validation.chat_option.header_value_string",
            "tui.model_config.validation.chat_option.logit_bias",
            "tui.model_config.validation.chat_option.must_be",
            "tui.model_config.validation.chat_option.nested_protected",
            "tui.model_config.validation.chat_option.object_example",
            "tui.model_config.validation.chat_option.object_type",
            "tui.model_config.validation.chat_option.output_cap",
            "tui.model_config.validation.chat_option.protected",
            "tui.model_config.validation.chat_option.stop",
            "tui.model_config.validation.chat_option.string",
            "tui.model_config.validation.clause.json_integer",
            "tui.model_config.validation.clause.json_integer_max",
            "tui.model_config.validation.clause.json_integer_min",
            "tui.model_config.validation.clause.json_integer_range",
            "tui.model_config.validation.clause.json_number",
            "tui.model_config.validation.clause.json_number_max",
            "tui.model_config.validation.clause.json_number_min",
            "tui.model_config.validation.clause.json_number_range",
            "tui.model_config.validation.clause.json_positive_integer",
            "tui.model_config.validation.field_positive_integer",
            "tui.model_config.validation.field_positive_number",
            "tui.model_config.validation.field_required_example",
            "tui.model_config.validation.field_valid_integer",
            "tui.model_config.validation.field_valid_number",
            "tui.model_config.validation.http_retries_non_negative",
            "tui.model_config.validation.http_retries_valid",
            "tui.model_config.validation.kv_detail",
            "tui.model_config.validation.kv_duplicate_key",
            "tui.model_config.validation.kv_header_reserved",
            "tui.model_config.validation.kv_key_required",
            "tui.model_config.validation.kv_value_required",
            "tui.model_config.validation.label.http_connect_timeout",
            "tui.model_config.validation.label.http_read_timeout",
            "tui.model_config.validation.label.max_context_tokens",
            "tui.model_config.validation.label.max_output_tokens",
            "tui.model_config.validation.max_context_minimum",
            "tui.model_config.validation.max_output_less_than_context",
            "tui.model_config.validation.model_required",
            "tui.model_config.validation.name_required",
            "tui.model_config.validation.profile_exists",
            "tui.model_config.validation.provider_allowed",
            "tui.model_config.warning.protected_chat_options",
            "tui.model_config.warning.responses_store_continuation",
        }
    )
    expected_ids.update(
        {
            "tui.acp.connection_report.current",
            "tui.acp.connection_report.current_value",
            "tui.acp.connection_report.env",
            "tui.acp.connection_report.fallback_title",
            "tui.acp.connection_report.header.capability",
            "tui.acp.connection_report.header.description",
            "tui.acp.connection_report.header.id",
            "tui.acp.connection_report.header.model",
            "tui.acp.connection_report.header.supported",
            "tui.acp.connection_report.heading.auth_methods",
            "tui.acp.connection_report.heading.capabilities",
            "tui.acp.connection_report.heading.config_options",
            "tui.acp.connection_report.heading.models",
            "tui.acp.connection_report.heading.modes",
            "tui.acp.connection_report.identity_not_reported",
            "tui.acp.connection_test.stderr_tail",
            "tui.agent_load.default_title",
            "tui.agent_load.failed",
            "tui.agent_picker.title",
            "tui.approval.judge.auto_approved",
            "tui.approval.judge.flagged",
            "tui.ask_user.placeholder.custom_response",
            "tui.connection_report.not_advertised",
            "tui.connection_test.default_subject",
            "tui.diff.badge.authorship_unverified",
            "tui.diff.badge.implicitly_detected",
            "tui.diff.badge.modified_internally_and_externally",
            "tui.diff.binary_file",
            "tui.diff.change_list",
            "tui.diff.content_not_backed_up",
            "tui.diff.eol.mixed",
            "tui.diff.eol.none",
            "tui.diff.external_tree",
            "tui.diff.file_moved",
            "tui.diff.load_error",
            "tui.diff.no_backup",
            "tui.diff.only_byte_representation_changed",
            "tui.diff.only_line_endings_changed",
            "tui.diff.select_file",
            "tui.diff.session_title",
            "tui.diff.skip_reason.binary_file",
            "tui.diff.skip_reason.file_too_large",
            "tui.diff.tab.all",
            "tui.diff.tab.turn",
            "tui.error.unknown",
            "tui.image_compression.default_title",
            "tui.main.session_json.no_active_session",
            "tui.mcp.connection_report.advertised_no_prompts",
            "tui.mcp.connection_report.advertised_no_tools",
            "tui.mcp.connection_report.connected_via",
            "tui.mcp.connection_report.header.advertised",
            "tui.mcp.connection_report.header.capability",
            "tui.mcp.connection_report.heading.capabilities",
            "tui.mcp.connection_report.heading.instructions",
            "tui.mcp.connection_report.heading.progressive_disclosure",
            "tui.mcp.connection_report.heading.prompts",
            "tui.mcp.connection_report.heading.tools",
            "tui.mcp.connection_report.identity_not_reported",
            "tui.mcp.connection_report.more_entries",
            "tui.mcp.connection_report.no_command_configured",
            "tui.mcp.connection_report.no_url_configured",
            "tui.mcp.connection_report.progressive_enabled",
            "tui.mcp.connection_report.prompt_loading_disabled",
            ("tui.mcp.connection_report.prompts_exposed", "tui.mcp.connection_report.prompts_exposed#plural"),
            ("tui.mcp.connection_report.tools_exposed", "tui.mcp.connection_report.tools_exposed#plural"),
            "tui.mcp.connection_test.error.cancelled",
            "tui.mcp.connection_test.error.configuration_conflict",
            "tui.mcp.connection_test.error.failed",
            "tui.mcp.connection_test.error.request_failed",
            "tui.mcp.connection_test.error.timed_out",
            "tui.mcp.connection_test.error.unreachable",
            "tui.mcp.connection_test.successful",
            "tui.model_config.api_style.chat_completions",
            "tui.model_config.api_style.responses",
            "tui.model_indicator.label.select",
            "tui.model_indicator.tooltip.configure",
            "tui.model_indicator.tooltip.details",
            "tui.model_indicator.tooltip.details_with_style",
            "tui.model_indicator.tooltip.locked.agent",
            "tui.model_indicator.tooltip.locked.generic",
            "tui.model_indicator.tooltip.locked.override",
            "tui.model_indicator.tooltip.select",
            "tui.model_indicator.tooltip.stream.off",
            "tui.model_indicator.tooltip.stream.on",
            "tui.model_indicator.tooltip.vision.off",
            "tui.model_indicator.tooltip.vision.on",
            ("tui.model_picker.hidden_profiles", "tui.model_picker.hidden_profiles#plural"),
            "tui.model_picker.manage",
            "tui.model_picker.title",
            "tui.notifications.button.test",
            "tui.notifications.desktop_popup",
            "tui.notifications.enable",
            "tui.notifications.section.delivery",
            "tui.notifications.section.events",
            "tui.notifications.section.general",
            "tui.notifications.sound",
            "tui.notifications.suppress_while_focused",
            "tui.rollback.debug_target.session_start",
            "tui.rollback.debug_target.turn",
            "tui.runtime_details.api_style.chat_completions",
            "tui.runtime_details.api_style.responses",
            "tui.runtime_details.inline_skills_source",
            "tui.session.profile_switch_indicator",
            "tui.session.working_directory_indicator",
            "tui.theme_picker.title",
            "tui.tool_view.copy_hint",
            "tui.tool_view.empty.input",
            "tui.tool_view.empty.output",
            "tui.tool_view.tab.input",
            "tui.tool_view.tab.output",
            "tui.workspace.change_directory",
        }
    )
    expected_ids.update(
        {
            "tui.chat.agent_fallback_label",
            "tui.chat.copy_agent_response_button",
            "tui.chat.copy_agent_response_tooltip",
            "tui.chat.retry_message",
            "tui.chat.think_prefix",
            "tui.app.quit_help",
            "tui.app.quit_help_title",
            "tui.compaction.copy.summary_tooltip",
            "tui.compaction.label_failed",
            "tui.compaction.label_failed_reason",
            "tui.compaction.label_interrupted",
            "tui.compaction.label_running",
            "tui.compaction.label_summarized",
            "tui.compaction.retry_notice",
            "tui.compaction.summary_title",
            "tui.editor.status.character_limit",
            "tui.editor.status.markdown_highlighting_paused",
            "tui.editor.status.paste_requires_insert",
            "tui.editor.status.ready",
            "tui.editor.status.selection_active",
            "tui.editor.vim.mode.insert",
            "tui.editor.vim.mode.normal",
            "tui.editor.vim.mode.visual",
            "tui.editor.vim.mode.visual_line",
            "tui.editor.vim.no_write_since_change",
            "tui.editor.vim.not_editor_command",
            "tui.model_guard.button.setup",
            "tui.model_guard.message",
            "tui.model_guard.title",
            "tui.tool_card.file_edit.bounded_preview_unchanged",
            "tui.tool_card.file_edit.full_diff_omitted",
            "tui.tool_card.file_edit.section.diff",
            "tui.tool_card.file_edit.section.diff_preview",
            "tui.terminal.launch.exited",
            "tui.terminal.launch.macos_failed",
            "tui.terminal.launch.unavailable",
            "tui.terminal.launch.unsupported",
            "tui.terminal.launch.windows_failed",
            "tui.terminal.shell.failed",
            "tui.terminal.shell.start_failed",
            "tui.tool_view.diff_prepare_failed",
            "tui.tool_view.empty",
            "tui.tool_view.image_unavailable",
            "tui.tool_view.preparing_diff",
            "tui.tool_view.section.output",
        }
    )
    expected_ids.update(
        {
            "settings.agent.default_profile.label",
            "settings.app.dev_mode.label",
            "settings.approval.default_mode.label",
            "settings.context.warn_threshold_pct.label",
            "settings.history.prompt.enabled.label",
            "settings.llm.retry.max_transient.label",
            "settings.log.raw_http_capture.label",
            "settings.memory.mcp.enabled.label",
            "settings.memory.writeback.idle_seconds.label",
            "settings.memory.writeback.on_session_end.label",
            "settings.model.profile.active.label",
            "settings.model.role.approval_judge.label",
            "settings.model.role.buddy_model_id.label",
            "settings.model.role.session_title.label",
            "settings.mutations.coordination.enabled.label",
            "settings.mutations.parallel_implicit_tools.label",
            "settings.mutations.snapshot.max_file_mb.label",
            "settings.mutations.snapshot.skip_binary.label",
            "settings.mutations.trace.fsatrace_path.label",
            "settings.mutations.trace.mode.label",
            "settings.notifications.delivery.desktop.label",
            "settings.notifications.delivery.sound.label",
            "settings.notifications.enabled.label",
            "settings.notifications.events.approval_required.label",
            "settings.notifications.events.ask_user.label",
            "settings.notifications.events.turn_complete.label",
            "settings.notifications.events.turn_error.label",
            "settings.notifications.suppress_when_focused.label",
            "settings.otel.enabled.label",
            "settings.otel.endpoint.label",
            "settings.otel.sensitive_data.label",
            "settings.project.config_enabled.label",
            "settings.project.hooks_enabled.label",
            "settings.pact.verify_command.label",
            "settings.rollback.snapshots_keep.label",
            "settings.routing.mode.label",
            "settings.routing.tiebreaker_model_profile.label",
            "settings.semantic_search.model_profile.label",
            "settings.session.title.auto.label",
            "settings.storage.session_root_dir.label",
            "settings.tools.ask_user.inline.label",
            "settings.tools.ask_user.timeout_seconds.label",
            "settings.tools.result.ceiling_tokens.label",
            "settings.ui.chat.file_snapshot_inline_chars.label",
            "settings.ui.editor.keymap.label",
            "settings.ui.locale.label",
            "settings.ui.theme.label",
            "settings.workspace.change_notice.enabled.label",
            "settings.workspace.change_notice.max_entries.label",
            "settings.workspace.mru_max_entries.label",
            "tui.binding.settings",
            "tui.commands.description.settings",
            "tui.commands.unknown_settings_tab",
            "tui.man.settings.body",
            "tui.settings.badge.dangerous",
            "tui.settings.badge.dotenv",
            "tui.settings.badge.env",
            "tui.settings.badge.pinned",
            "tui.settings.badge.project",
            "tui.settings.badge.sealed",
            "tui.settings.badge.this_session",
            "tui.settings.confirm.approval_auto",
            "tui.settings.confirm.button",
            "tui.settings.confirm.dangerous",
            "tui.settings.confirm.otel_sensitive_data",
            "tui.settings.confirm.raw_http_capture",
            "tui.settings.confirm.title",
            "tui.settings.dialog.title",
            "tui.settings.error.above_maximum",
            "tui.settings.error.below_minimum",
            "tui.settings.error.expected_finite_number",
            "tui.settings.error.expected_int",
            "tui.settings.error.expected_non_negative_int",
            "tui.settings.error.expected_number",
            "tui.settings.error.expected_text",
            "tui.settings.error.invalid",
            "tui.settings.error.not_a_choice",
            "tui.settings.error.required",
            "tui.settings.hint.agent.default_profile",
            "tui.settings.hint.approval.default_mode",
            "tui.settings.hint.history.prompt.enabled",
            "tui.settings.hint.llm.retry.max_transient",
            "tui.settings.hint.log.raw_http_capture",
            "tui.settings.hint.model.role.approval_judge",
            "tui.settings.hint.model.role.buddy_model_id",
            "tui.settings.hint.model.role.session_title",
            "tui.settings.hint.otel.enabled",
            "tui.settings.hint.otel.endpoint",
            "tui.settings.hint.otel.sensitive_data",
            ("tui.settings.hint.project.config_dormant", "tui.settings.hint.project.config_dormant#plural"),
            "tui.settings.hint.project.config_enabled",
            "tui.settings.hint.project.hooks_enabled",
            "tui.settings.hint.rollback.snapshots_keep",
            "tui.settings.hint.session.title.auto",
            "tui.settings.hint.storage.session_root_dir",
            "tui.settings.hint.tools.ask_user.inline",
            "tui.settings.hint.tools.ask_user.timeout_seconds",
            "tui.settings.hint.ui.locale",
            "tui.settings.hint.ui.theme",
            "tui.settings.hint.workspace.change_notice.enabled",
            "tui.settings.migrate.browse",
            "tui.settings.migrate.button.cancel",
            "tui.settings.migrate.button.close",
            "tui.settings.migrate.button.migrate",
            "tui.settings.migrate.copying",
            "tui.settings.migrate.description",
            "tui.settings.migrate.failed",
            "tui.settings.migrate.failed_item",
            "tui.settings.migrate.from",
            "tui.settings.migrate.need_paths",
            "tui.settings.migrate.nothing_to_copy",
            "tui.settings.migrate.restart_hint",
            "tui.settings.migrate.summary",
            "tui.settings.migrate.title",
            "tui.settings.migrate.to",
            "tui.settings.provenance.cli",
            "tui.settings.provenance.dotenv",
            "tui.settings.provenance.env",
            "tui.settings.provenance.project",
            "tui.settings.provenance.sealed",
            "tui.settings.provenance.session",
            "tui.settings.save_rejected",
            "tui.settings.section.agent",
            "tui.settings.section.appearance",
            "tui.settings.section.approval",
            "tui.settings.section.diagnostics",
            "tui.settings.section.input",
            "tui.settings.section.llm",
            "tui.settings.section.location",
            "tui.settings.section.model_roles",
            "tui.settings.section.notifications",
            "tui.settings.section.project_trust",
            "tui.settings.section.rollback",
            "tui.settings.section.session_titles",
            "tui.settings.section.telemetry",
            "tui.settings.section.tools",
            "tui.settings.section.workspace_notice",
            "tui.settings.select.active",
            "tui.settings.select.current",
            "tui.settings.select.default",
            "tui.settings.session_root.browse",
            "tui.settings.session_root.in_use",
            "tui.settings.session_root.invalid",
            "tui.settings.session_root.migrate",
            "tui.settings.session_root.written",
            "tui.settings.status.idle",
            "tui.settings.status.reload_after_turn",
            "tui.settings.status.reload_on_close",
            ("tui.settings.status.restart", "tui.settings.status.restart#plural"),
            "tui.settings.tab.general",
            "tui.settings.tab.models_agents",
            "tui.settings.tab.notifications",
            "tui.settings.tab.security",
            "tui.settings.tab.sessions",
            "tui.settings.tab.tools",
            "tui.settings.title",
            "tui.trajectory.bucket.idle",
            "tui.trajectory.bucket.model",
            "tui.trajectory.bucket.tools",
            "tui.trajectory.bucket.wait",
            "tui.trajectory.category.agent",
            "tui.trajectory.category.approval",
            "tui.trajectory.category.compaction",
            "tui.trajectory.category.hook",
            "tui.trajectory.category.model",
            "tui.trajectory.category.operation",
            "tui.trajectory.category.preparation",
            "tui.trajectory.category.retry",
            "tui.trajectory.category.tool",
            "tui.trajectory.category.wait",
            "tui.trajectory.charts_too_narrow",
            "tui.trajectory.coverage",
            "tui.trajectory.coverage_shares",
            "tui.trajectory.diagnostics.accounted_prefix",
            "tui.trajectory.diagnostics.after_sequence",
            "tui.trajectory.diagnostics.containment",
            ("tui.trajectory.diagnostics.corrupt", "tui.trajectory.diagnostics.corrupt#plural"),
            (
                "tui.trajectory.diagnostics.duration_mismatch_summary",
                "tui.trajectory.diagnostics.duration_mismatch_summary#plural",
            ),
            "tui.trajectory.diagnostics.gap",
            "tui.trajectory.diagnostics.healthy",
            ("tui.trajectory.diagnostics.hook_modes", "tui.trajectory.diagnostics.hook_modes#plural"),
            "tui.trajectory.diagnostics.intro",
            "tui.trajectory.diagnostics.operation",
            "tui.trajectory.diagnostics.operation.detached_hook",
            "tui.trajectory.diagnostics.operation.invalid_endpoints",
            "tui.trajectory.diagnostics.operation.missing_start",
            "tui.trajectory.diagnostics.operation.missing_terminal",
            "tui.trajectory.diagnostics.operation.nonunique",
            "tui.trajectory.diagnostics.operation.outside_coverage",
            "tui.trajectory.diagnostics.operation.rollback_start",
            "tui.trajectory.diagnostics.operation.rollback_terminal",
            "tui.trajectory.diagnostics.rollback",
            "tui.trajectory.diagnostics.sequence",
            (
                "tui.trajectory.diagnostics.side_call_empty_shells",
                "tui.trajectory.diagnostics.side_call_empty_shells#plural",
            ),
            "tui.trajectory.diagnostics.title",
            "tui.trajectory.diagnostics.torn_tail",
            (
                "tui.trajectory.diagnostics.unidentified_membership",
                "tui.trajectory.diagnostics.unidentified_membership#plural",
            ),
            "tui.trajectory.diagnostics.unresolved_metric",
            ("tui.trajectory.diagnostics.unsupported", "tui.trajectory.diagnostics.unsupported#plural"),
            "tui.trajectory.diagnostics.utilization_metric",
            "tui.trajectory.diagnostics.wall_metric",
            "tui.trajectory.elapsed_scope",
            "tui.trajectory.kpi.parallel",
            "tui.trajectory.kpi.split",
            "tui.trajectory.kpi.time",
            "tui.trajectory.metric.cp_compute",
            "tui.trajectory.metric.cp_response",
            "tui.trajectory.metric.elapsed",
            "tui.trajectory.metric.overlap",
            "tui.trajectory.metric.parallelism",
            "tui.trajectory.metric.usage",
            "tui.trajectory.metric.work",
            "tui.trajectory.no_turns",
            "tui.trajectory.precision.estimated",
            "tui.trajectory.precision.exact",
            "tui.trajectory.precision.missing",
            "tui.trajectory.precision.unresolved",
            "tui.trajectory.read_error",
            "tui.trajectory.session_info.copy_path",
            "tui.trajectory.session_info.events",
            ("tui.trajectory.session_info.files", "tui.trajectory.session_info.files#plural"),
            "tui.trajectory.session_info.first_message",
            "tui.trajectory.session_info.last_reply",
            "tui.trajectory.session_info.mutations",
            "tui.trajectory.session_info.on_disk",
            "tui.trajectory.session_info.open_failed",
            "tui.trajectory.session_info.open_folder",
            "tui.trajectory.session_info.open_unavailable",
            "tui.trajectory.session_info.path",
            "tui.trajectory.session_info.path_copied",
            "tui.trajectory.session_info.runtimes",
            "tui.trajectory.session_info.snapshots",
            "tui.trajectory.session_info.span",
            "tui.trajectory.session_info.sub_agents",
            "tui.trajectory.session_info.title",
            "tui.trajectory.session_info.turns",
            "tui.trajectory.tab.insights",
            "tui.trajectory.tab.overview",
            "tui.trajectory.tab.session_data",
            "tui.trajectory.tab.timeline",
            "tui.trajectory.time_ruler",
            "tui.trajectory.title",
            "tui.trajectory.turn",
            "tui.trajectory.unavailable",
            "tui.trajectory.utilization",
            "tui.trajectory.waterfall",
        }
    )
    expected_ids.update(
        {
            "settings.trajectory.verify_commands.label",
            "tui.trajectory.action.edit",
            "tui.trajectory.action.read",
            "tui.trajectory.action.search",
            "tui.trajectory.action.verify",
            "tui.trajectory.action_funnel.title",
            "tui.trajectory.change_verification.counts",
            "tui.trajectory.change_verification.files",
            "tui.trajectory.change_verification.state.after",
            "tui.trajectory.change_verification.state.net_zero",
            "tui.trajectory.change_verification.state.unverified",
            "tui.trajectory.change_verification.state.verified",
            "tui.trajectory.change_verification.title",
            "tui.trajectory.change_verification.truncated",
            "tui.trajectory.change_verification.unavailable",
            "tui.trajectory.failure_recovery.amplification",
            "tui.trajectory.failure_recovery.failures",
            "tui.trajectory.failure_recovery.median",
            "tui.trajectory.failure_recovery.repeated",
            "tui.trajectory.failure_recovery.title",
            "tui.trajectory.finding.approval_blocking_share.detail",
            "tui.trajectory.finding.approval_blocking_share.title",
            (
                "tui.trajectory.finding.context_carrying_load.detail",
                "tui.trajectory.finding.context_carrying_load.detail#plural",
            ),
            "tui.trajectory.finding.context_carrying_load.title",
            "tui.trajectory.finding.failed_attempt_critical_path.detail",
            "tui.trajectory.finding.failed_attempt_critical_path.title",
            (
                "tui.trajectory.finding.net_zero_churn.detail",
                "tui.trajectory.finding.net_zero_churn.detail#plural",
            ),
            "tui.trajectory.finding.net_zero_churn.title",
            (
                "tui.trajectory.finding.repeated_tool_fingerprint.detail",
                "tui.trajectory.finding.repeated_tool_fingerprint.detail#plural",
            ),
            "tui.trajectory.finding.repeated_tool_fingerprint.title",
            (
                "tui.trajectory.finding.retry_token_amplification.detail",
                "tui.trajectory.finding.retry_token_amplification.detail#plural",
            ),
            "tui.trajectory.finding.retry_token_amplification.title",
            (
                "tui.trajectory.finding.unverified_change.detail",
                "tui.trajectory.finding.unverified_change.detail#plural",
            ),
            "tui.trajectory.finding.unverified_change.title",
            "tui.trajectory.findings.none",
            "tui.trajectory.findings.title",
            "tui.trajectory.graph.cycle",
            "tui.trajectory.graph.legend",
            "tui.trajectory.graph.none",
            "tui.trajectory.graph.response",
            "tui.trajectory.graph.timeline_hint",
            "tui.trajectory.graph.title",
            "tui.trajectory.insights.column.approval",
            "tui.trajectory.insights.column.calls",
            "tui.trajectory.insights.column.connection_wait",
            "tui.trajectory.insights.column.cp_exclusive",
            "tui.trajectory.insights.column.duration_share",
            "tui.trajectory.insights.column.exit_codes",
            "tui.trajectory.insights.column.first_action",
            "tui.trajectory.insights.column.injected_tokens",
            "tui.trajectory.insights.column.loads",
            "tui.trajectory.insights.column.otel_cross_check",
            "tui.trajectory.insights.column.p50",
            "tui.trajectory.insights.column.p95",
            "tui.trajectory.insights.column.resource_reads",
            "tui.trajectory.insights.column.results",
            "tui.trajectory.insights.column.return_volume",
            "tui.trajectory.insights.column.revisions",
            "tui.trajectory.insights.column.script_runs",
            "tui.trajectory.insights.column.truncated_spill",
            "tui.trajectory.insights.column.turns",
            "tui.trajectory.insights.integrations.mcp",
            "tui.trajectory.insights.integrations.mcp_none",
            "tui.trajectory.insights.integrations.skills",
            "tui.trajectory.insights.integrations.skills_none",
            "tui.trajectory.insights.status.skill_changed",
            "tui.trajectory.insights.status.skill_not_found",
            "tui.trajectory.insights.status.unattributed",
            "tui.trajectory.insights.status.unclassified",
            "tui.trajectory.insights.tokens.carrying_assistant",
            "tui.trajectory.insights.tokens.carrying_explainer",
            "tui.trajectory.insights.tokens.carrying_item",
            "tui.trajectory.insights.tokens.carrying_load",
            "tui.trajectory.insights.tokens.carrying_row_detail",
            "tui.trajectory.insights.tokens.carrying_row_head",
            "tui.trajectory.insights.tokens.carrying_tool_result",
            "tui.trajectory.insights.tokens.carrying_user",
            "tui.trajectory.insights.tokens.input_share",
            "tui.trajectory.insights.tokens.no_turn_data",
            "tui.trajectory.insights.tokens.per_turn",
            "tui.trajectory.insights.tools.none",
            "tui.trajectory.insights.tools.title",
            "tui.trajectory.mcp_usage.none",
            "tui.trajectory.mcp_usage.title",
            "tui.trajectory.preparation_outcome.abandoned_no_target",
            "tui.trajectory.preparation_outcome.cancelled",
            "tui.trajectory.preparation_outcome.completed",
            "tui.trajectory.preparation_outcome.conflict",
            "tui.trajectory.preparation_outcome.dropped",
            "tui.trajectory.preparation_outcome.failed",
            "tui.trajectory.preparation_outcome.fresh_turn",
            "tui.trajectory.preparation_outcome.handoff",
            "tui.trajectory.preparation_outcome.image_rejected",
            "tui.trajectory.preparation_outcome.injected",
            "tui.trajectory.preparation_outcome.interrupted",
            "tui.trajectory.preparation_outcome.not_ready",
            "tui.trajectory.preparation_outcome.owner_changed",
            "tui.trajectory.preparation_outcome.preparation_failed",
            "tui.trajectory.preparation_outcome.rejected",
            "tui.trajectory.preparation_outcome.retry_turn",
            "tui.trajectory.preparation_outcome.superseded",
            "tui.trajectory.preparation_outcome.target_stale",
            "tui.trajectory.skill_usage.none",
            "tui.trajectory.skill_usage.title",
            "tui.trajectory.submission_latency.became_turn",
            "tui.trajectory.submission_latency.did_not_become",
            "tui.trajectory.submission_latency.injected",
            "tui.trajectory.submission_latency.intro",
            "tui.trajectory.submission_latency.none",
            "tui.trajectory.submission_latency.sample",
            (
                "tui.trajectory.submission_latency.stats",
                "tui.trajectory.submission_latency.stats#plural",
            ),
            "tui.trajectory.submission_latency.title",
            (
                "tui.trajectory.submission_latency.unresolved",
                "tui.trajectory.submission_latency.unresolved#plural",
            ),
            "tui.trajectory.timeline.graph_hint",
            "tui.trajectory.token_count",
            "tui.trajectory.token_usage.cache_creation",
            "tui.trajectory.token_usage.cache_hit",
            "tui.trajectory.token_usage.cache_read",
            "tui.trajectory.token_usage.input",
            "tui.trajectory.token_usage.output",
            "tui.trajectory.token_usage.reasoning",
            "tui.trajectory.token_usage.title",
            "tui.trajectory.tool_usage.more",
            "tui.trajectory.tool_usage.unattributed",
        }
    )
    assert len(expected_ids) == 1837
    assert {_entry_id(message) for message in pot if message.id} == expected_ids
    assert {_entry_id(message) for message in po if message.id} == expected_ids
    assert set(_effective_catalog_entries(po)) == expected_ids
    assert po.num_plurals == 1
    assert po.plural_expr == "0"
    with i18n.ZH_MO_PATH.open("rb") as stream:
        translations = gettext.GNUTranslations(stream)
    assert translations.gettext("missing.key") == "missing.key"
    assert "nplurals=1" in translations.info()["plural-forms"]
    assert not (i18n.REPO_ROOT / "locales" / "en" / "LC_MESSAGES" / "chrys.po").exists()
    assert not (i18n.REPO_ROOT / "locales" / "en" / "LC_MESSAGES" / "chrys.mo").exists()
    _assert_po_mo_semantically_consistent(
        source_root=i18n.SOURCE_ROOT,
        pot_path=i18n.POT_PATH,
        po_path=i18n.ZH_PO_PATH,
        mo_path=i18n.ZH_MO_PATH,
    )


def test_pseudo_output_guard_uses_real_paths(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path, "")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output = tmp_path / "outside"
    result = i18n.generate_pseudo_catalog(
        output / "nested" / os.pardir,
        source_root=source_root,
        repo_root=repo_root,
        location_root=source_root,
    )

    assert result.is_file()
