"""Redfish protocol behaviour every BMC shares (DMTF DSP0266), checked on all
BMCs the way a generic Redfish client relies on it."""
from conftest import ROOT, at, members, redfish_message_id


def test_service_root_is_unauthenticated(lab, bmc_name):
    b = lab.bmc(bmc_name)
    r = b.request("GET", "/redfish", auth=False)
    assert r.status_code == 200 and r.json() == {"v1": f"{ROOT}/"}
    r = b.request("GET", ROOT, auth=False)
    assert r.status_code == 200
    assert r.headers.get("OData-Version") == "4.0"
    sr = r.json()
    assert sr["Vendor"] == "NVIDIA", "clients pick vendor quirks from Vendor"
    for k in ("Chassis", "Managers", "SessionService"):
        assert at(sr, k, "@odata.id"), f"service root has no {k} link"


def test_resources_need_credentials(lab, bmc_name):
    b = lab.bmc(bmc_name)
    r = b.request("GET", f"{ROOT}/Chassis", auth=False)
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate"), "401 without WWW-Authenticate"
    b.password = "wrong"
    assert b.request("GET", f"{ROOT}/Chassis").status_code == 401


def test_session_lifecycle(lab, bmc_name):
    b = lab.bmc(bmc_name)
    bad = b.request("POST", f"{ROOT}/SessionService/Sessions", {"UserName": lab.user, "Password": "wrong"}, auth=False)
    assert bad.status_code == 401

    r = b.login()
    loc = r.headers.get("Location")
    assert loc and r.json()["@odata.id"] == loc
    assert loc in members(b.get(f"{ROOT}/SessionService/Sessions"))
    assert b.get(loc)["UserName"] == lab.user
    b.write("DELETE", loc)
    assert b.request("GET", f"{ROOT}/Chassis").status_code == 401, "token accepted after logout"


def test_manager_reset_ends_sessions(lab, bmc_name):
    """A BMC reset ends every session; the managed host keeps running."""
    b = lab.bmc(bmc_name)
    b.login()
    mgr = members(b.get(f"{ROOT}/Managers"))[0]
    target = at(b.get(mgr), "Actions", "#Manager.Reset", "target")
    assert target, f"{mgr} advertises no Manager.Reset"
    b.write("POST", target, {"ResetType": "GracefulRestart"})
    assert b.request("GET", f"{ROOT}/Chassis").status_code == 401, "session survived Manager.Reset"


def test_errors_are_redfish_messages(lab, bmc_name):
    b = lab.bmc(bmc_name)
    r = b.request("GET", f"{ROOT}/Chassis/NoSuchChassis")
    assert r.status_code == 404
    assert redfish_message_id(r).endswith(".ResourceNotFound")

    r = b.request("POST", f"{ROOT}/SessionService/Sessions", auth=False, data=b"{not json",
                  headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert redfish_message_id(r).endswith(".MalformedJSON")

    # A method the resource does not support is refused, not treated as a missing resource.
    r = b.request("DELETE", f"{ROOT}/Chassis")
    assert r.status_code == 405
    assert redfish_message_id(r), f"405 body is not a Redfish error: {r.text[:200]}"


def _links(v, top=True):
    """Every {"@odata.id": ...} nested in v except the resource's own."""
    if isinstance(v, dict):
        if not top and isinstance(v.get("@odata.id"), str):
            yield v["@odata.id"].split("#")[0].rstrip("/")
        for c in v.values():
            yield from _links(c, False)
    elif isinstance(v, list):
        for c in v:
            yield from _links(c, False)


def test_every_link_resolves(lab, bmc_name):
    """Crawl everything reachable from the service root, as DMTF's
    Redfish-Service-Validator does: every link resolves, every resource names
    itself by the URI it was fetched from and has a type, every collection
    counts its members correctly."""
    b = lab.bmc(bmc_name)
    seen, queue, problems = {ROOT}, [ROOT], []
    while queue and len(seen) < 5000:
        path = queue.pop(0)
        r = b.request("GET", path)
        if r.status_code != 200:
            problems.append(f"GET {path}: {r.status_code}")
            continue
        res = r.json()
        if res.get("@odata.id", "").rstrip("/") != path:
            problems.append(f"{path}: @odata.id {res.get('@odata.id')!r}")
        typ = res.get("@odata.type", "")
        if not typ.startswith("#"):
            problems.append(f"{path}: @odata.type {typ!r}")
        if typ.endswith("Collection"):
            if res.get("Members@odata.count") != len(res.get("Members", [])):
                problems.append(f"{path}: Members@odata.count {res.get('Members@odata.count')} != {len(res['Members'])}")
        elif path != ROOT and not res.get("Id"):
            problems.append(f"{path}: no Id")
        for link in _links(res):
            # Sessions come and go while crawling; actions are POST-only.
            if link not in seen and link.startswith(ROOT) and "/Sessions/" not in link and "/Actions/" not in link:
                seen.add(link)
                queue.append(link)
    assert not problems, "\n".join(problems)
