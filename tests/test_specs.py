from helpers import ROOT, orders_specs
from spec_run_graph.specs.diff import diff_hashes
from spec_run_graph.specs.loader import digest, discover
from spec_run_graph.specs.openapi import schema_brief


def test_discover_folder():
    files = discover([ROOT / "examples" / "orders"])
    assert sorted(f.name for f in files) == ["openapi.yaml", "prd.md"]


def test_openapi_parsing():
    oas = next(s for s in orders_specs() if s.spec_type == "OPENAPI")
    assert (oas.name, oas.version) == ("Orders API", "1.0.0")
    idx = oas.endpoint_index()
    assert set(idx) == {
        "GET /v1/products",
        "POST /v1/orders",
        "GET /v1/orders/{orderId}",
        "DELETE /v1/orders/{orderId}",
        "POST /v1/orders/{orderId}/confirm",
    }
    create = idx["POST /v1/orders"]
    assert create.requires_auth and create.tag == "orders"
    assert {"201", "400", "401", "422"} <= set(create.response_schemas)
    # $refs are resolved
    assert create.response_schemas["201"]["schema"]["properties"]["data"]["properties"]["status"]["enum"][0] == "DRAFT"
    assert idx["GET /v1/orders/{orderId}"].parameters[0]["name"] == "orderId"


def test_digest_is_compact():
    oas, prd = orders_specs()
    text = digest(oas)
    assert "POST /v1/orders  [orders] createOrder" in text
    assert "qty*: integer 1..100" in text
    assert "[id: PRD-ORD-43]" in digest(prd)


def test_prd_sections():
    prd = next(s for s in orders_specs() if s.spec_type == "REQUIREMENTS_DOC")
    ids = [s.req_id for s in prd.sections if s.req_id]
    assert ids[:3] == ["PRD-ORD-40", "PRD-ORD-42", "PRD-ORD-43"]
    assert prd.version.startswith("sha-")


def test_schema_brief_cycles():
    assert schema_brief({"$ref_cycle": "#/components/schemas/Node"}) == "Node"
    assert schema_brief({"type": "array", "items": {"type": "string"}, "maxItems": 3}) == "[string] ..3"


def test_v2_diff_detects_changed_endpoint():
    v1 = next(s for s in orders_specs() if s.spec_type == "OPENAPI")
    v2 = next(s for s in orders_specs("orders_v2") if s.spec_type == "OPENAPI")
    ch = diff_hashes(
        v1.name,
        v1.version,
        v2.version,
        {e.key: e.contract_hash for e in v1.endpoints},
        {e.key: e.contract_hash for e in v2.endpoints},
    )
    assert ch.changed == ["POST /v1/orders"] and not ch.added and not ch.removed
    assert ch.affected == {"POST /v1/orders"}
