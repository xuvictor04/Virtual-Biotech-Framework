"""Envelope conformance suite (phase 4, F21: the fifth plugin kind) over the registered envelope plugins,
plus the kind's wiring: ``KINDS``, registry discovery and validation, a third-party envelope added without
a core change, and ``gateway/classify.py`` decoding through a binding's codec."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any, ClassVar, Mapping

import pytest

from vbt.datalayer.api import RawResult
from vbt.datalayer.gateway.classify import classify, decode, envelope_for, use_registry
from vbt.datalayer.plugins import KINDS
from vbt.datalayer.plugins.base import (
    CAPABILITIES,
    REQUIRED_METHODS,
    EnvelopeBase,
    EnvelopePlugin,
    ParsedResult,
    PluginBase,
    PluginError,
)
from vbt.datalayer.plugins.conformance import selected_plugins
from vbt.datalayer.plugins.conformance.envelope import *  # noqa: F401,F403  (collects E-1..E-5)
from vbt.datalayer.plugins.conformance.envelope import (
    GARBAGE,
    UNKNOWN_SHAPES,
    EnvelopeCase,
    EnvelopeCases,
    check_case,
)
from vbt.datalayer.plugins.envelopes import DEFAULT_ENVELOPE, codec_of, default_envelope, http_status
from vbt.datalayer.plugins.envelopes.jsonpath import JsonPathEnvelope
from vbt.datalayer.plugins.registry import PluginRegistry, discover, validate_plugin


class XmlHitsEnvelope(EnvelopeBase):
    """A third-party envelope: ``<hits total="N"><hit id=".."/></hits>`` and ``<error code="404">``."""

    name: ClassVar[str] = "dl_test_xml_hits"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"rows", "totals", "http_status"})

    def decode(self, raw_text: str | None, structured: Any, spec: Mapping[str, Any]) -> ParsedResult:
        try:
            root = ET.fromstring(raw_text or "")
        except ET.ParseError:
            return ParsedResult(unparsed=True, message=(raw_text or "")[:200] or None)
        if root.tag == "error":
            code = int(root.get("code", "0") or 0) or None
            return ParsedResult(found=False if code == 404 else None, message=root.text, http_status=code)
        if root.tag != "hits":
            return ParsedResult(unparsed=True)
        total = root.get("total")
        return ParsedResult(rows=[dict(h.attrib) for h in root.findall("hit")],
                            total=int(total) if total and total.isdigit() else None)

    @classmethod
    def conformance_cases(cls) -> Any:
        return EnvelopeCases(recorded=(
            EnvelopeCase("hits", '<hits total="7"><hit id="a"/><hit id="b"/></hits>', None, {}, rows=2, total=7),
            EnvelopeCase("not_found", '<error code="404">no such entry</error>', None, {}, found=False,
                         http_status=404, rows=None),
        ), spec={"rows": "hits"})


def test_envelope_is_the_fifth_kind() -> None:
    assert KINDS["envelope"] is EnvelopePlugin and set(CAPABILITIES) == set(KINDS)
    assert REQUIRED_METHODS["envelope"] == ("decode",)
    reg = discover(entry_points=False)
    assert reg.names("envelope") == [DEFAULT_ENVELOPE] and isinstance(reg.envelope(), JsonPathEnvelope)
    assert [p.name for p in selected_plugins("envelope", reg)] == [DEFAULT_ENVELOPE]
    assert reg.versions()[f"envelope/{DEFAULT_ENVELOPE}"] == "1.0"
    with pytest.raises(KeyError):
        reg.envelope("no_such_codec")
    assert PluginRegistry({k: v for k, v in KINDS.items() if k != "envelope"}).envelope() is default_envelope()


def test_validation_of_envelope_plugins() -> None:
    class NoDecode(PluginBase):
        kind = "envelope"
        name = "dl_bad_envelope"
        capabilities = frozenset()

    with pytest.raises(PluginError, match="decode"):
        validate_plugin(NoDecode())
    bad = type("BadCaps", (XmlHitsEnvelope,), {"capabilities": frozenset({"teleport"})})
    with pytest.raises(PluginError, match="unknown envelope capabilities"):
        validate_plugin(bad())


def test_a_third_party_envelope_needs_no_core_change() -> None:
    reg = discover(entry_points=False, extra=[XmlHitsEnvelope])
    plugin = reg.envelope("dl_test_xml_hits")
    for case in plugin.conformance_cases().recorded:
        check_case(plugin, case)
    for name, text, structured in UNKNOWN_SHAPES:
        got = plugin.decode(text, structured, {})
        assert got.unparsed and not got.rows, name
    for text, structured in GARBAGE:
        assert isinstance(plugin.decode(text, structured, {}), ParsedResult)


def _contract(codec: str | None, **result: Any) -> Any:
    res = SimpleNamespace(not_found_when=[], nested_errors=[], total=None, row_paths=["$.hits"], kind="rows",
                          codec=codec, codec_options={}, **result)
    binding = SimpleNamespace(result=res)
    return SimpleNamespace(binding=binding, generic=False, generic_spec=None, server="xmlsrv", tool="find",
                           tables={}, bound_table=None, identifier_args=[], args={})


def test_classify_decodes_with_the_binding_codec() -> None:
    reg = discover(entry_points=False, extra=[XmlHitsEnvelope])
    c = _contract("dl_test_xml_hits")
    nf = classify(RawResult('<error code="404">no such entry</error>', None, None, "ok"), c, None, registry=reg)
    assert nf.outcome == "not_found" and nf.parsed.http_status == 404
    ok = classify(RawResult('<hits total="2"><hit id="a"/><hit id="b"/></hits>', None, None, "ok"), c, None,
                  registry=reg)
    assert ok.outcome == "ok" and ok.parsed.total == 2 and len(ok.parsed.rows) == 2
    down = classify(RawResult('<error code="503">maintenance</error>', None, None, "ok"), c, None, registry=reg)
    assert down.outcome == "source_error" and down.error.subkind == "http_status"
    # the registry set for the gateway is used when none is passed
    use_registry(reg)
    try:
        assert envelope_for("dl_test_xml_hits") is reg.envelope("dl_test_xml_hits")
        name, parsed = decode(RawResult('<hits total="1"><hit id="z"/></hits>', None, None, "ok"), c)
        assert name == "dl_test_xml_hits" and parsed.rows == [{"id": "z"}]
    finally:
        use_registry(None)


def test_codec_of_reads_the_overlay_result_spec() -> None:
    from vbt.datalayer.descriptor.overlay import ResultSpec, TotalSpec

    spec = ResultSpec(rows="$.trials", total=TotalSpec(path="$.total_count"), not_found_when=["$.error =~ 'x'"],
                      nested_errors=["$.details.error"])
    name, options = codec_of(spec)
    assert name == DEFAULT_ENVELOPE
    assert options == {"rows": ["$.trials"], "total": "$.total_count", "not_found_when": ["$.error =~ 'x'"],
                       "errors": ["$.details.error"]}
    got = default_envelope().decode('{"total_count": 3, "trials": [{}, {}, {}], "details": {"error": "boom"}}', None,
                                    options)
    assert got.total == 3 and len(got.rows) == 3 and got.errors == ("boom",)


@pytest.mark.parametrize("text, status", [
    ("Failed to retrieve data: status code: 404", 404), ("HTTP 503 from upstream", 503),
    ("Server error '502 Bad Gateway' for url 'https://x'", 502), ("upstream HTTP/1.1 500", 500),
    ("no status here (2024 rows)", None), ("value 404 of 1000", None),
])
def test_http_status_from_text(text: str, status: int | None) -> None:
    assert http_status(None, text) == status


def test_http_status_from_payload_keys() -> None:
    assert http_status({"status": 404}) == 404 and http_status({"statusCode": "503"}) == 503
    assert http_status({"status": "ok"}) is None and http_status({"error": "status code: 429"}) == 429
