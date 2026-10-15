"""Flight and train option retrieval, including the non-live labelling contract."""
import pytest

from tools import _estimate_flight_options, _local_airport_table, _match_local_airport, get_flight_options, get_train_options


def _generated_stub(*_args, **_kwargs):
    return ([{"airline": "IndiGo", "flight_number": "6E 101", "departure_time": "09:30",
              "arrival_time": "11:00", "duration_minutes": 90, "stops": "non-stop",
              "aircraft": "A320", "cabin": "Economy", "indicative_fare_inr": 5500}], "Stub")


def test_local_airport_table_resolves_major_cities_and_iata_codes():
    assert _match_local_airport("Pune")["iata"] == "PNQ"
    assert _match_local_airport("udaipur")["iata"] == "UDR"
    assert _match_local_airport("Dubai")["iata"] == "DXB"
    # A bare IATA code resolves directly.
    assert _match_local_airport("DEL")["city"] == "New Delhi"
    assert _match_local_airport("Nowhereville") is None
    assert len(_local_airport_table()) > 20


def test_great_circle_estimate_claims_no_airline_or_schedule():
    pune = _match_local_airport("Pune")
    udaipur = _match_local_airport("Udaipur")
    rows = _estimate_flight_options("Pune", "Udaipur", pune, udaipur)
    assert rows, "Pune-Udaipur is a real route and should produce an estimate"
    row = rows[0]
    # The deterministic fallback must never invent a carrier or flight number.
    assert row["airline"] is None
    assert row["flight_number"] is None
    assert row["departure_time"] is None
    assert 40 < row["duration_minutes"] < 240
    assert 500 < row["distance_km"] < 900


def test_flight_options_fall_back_when_no_aviation_key(monkeypatch):
    import tools

    monkeypatch.setattr(tools, "_aviationstack_key", lambda: "")
    monkeypatch.setattr(tools, "generate_flight_options", _generated_stub)
    result = get_flight_options("Pune", "Udaipur")
    assert result["mode"] == "flights"
    assert result["status"] == "generated"
    assert result["live"] is False
    assert result["flights"]
    # The estimate is still computed from local coordinates.
    assert result["flight_estimate"]
    assert "not verified" in result["notice"]


def test_flight_options_report_honestly_when_nothing_is_available(monkeypatch):
    import tools

    monkeypatch.setattr(tools, "_aviationstack_key", lambda: "")
    monkeypatch.setattr(tools, "generate_flight_options", lambda *a, **k: ([], "unavailable"))
    monkeypatch.setattr(tools, "_resolve_airport", lambda name: None)
    result = get_flight_options("Nowhereville", "Elsewhere")
    assert result["status"] == "unavailable"
    assert result["flights"] == []
    assert "could not" in result["notice"].lower()


def test_train_options_are_trains_only(monkeypatch):
    import tools

    monkeypatch.setattr(tools, "_railradar_key", lambda: "stub")
    monkeypatch.setattr(tools, "_railradar_trains", lambda *a, **k: {
        "status": "ok", "trains": [{"train": {"number": "19668", "name": "Humsafar"},
                                     "from": {"departure": "08:15"}, "to": {"arrival": "03:10"},
                                     "live": {"type": "on-time"}}]})
    result = get_train_options("Jaipur", "Udaipur")
    assert result["mode"] == "trains"
    assert result["live"] is True
    assert result["trains"][0]["number"] == "19668"
    # No road or air alternatives leak into a train request.
    assert "options" in result and result["options"] == []
    assert result["notice"]


def test_train_options_explain_when_the_feed_is_unconfigured(monkeypatch):
    import tools

    monkeypatch.setattr(tools, "_railradar_key", lambda: "")
    result = get_train_options("Jaipur", "Udaipur")
    assert result["status"] == "not_configured"
    assert result["trains"] == []
    assert "not configured" in result["notice"].lower()


def test_train_options_do_not_claim_a_timetable_when_lookup_fails(monkeypatch):
    import tools

    monkeypatch.setattr(tools, "_railradar_key", lambda: "stub")
    monkeypatch.setattr(tools, "_railradar_trains", lambda *a, **k: {"status": "unavailable", "trains": []})
    result = get_train_options("Jaipur", "Udaipur")
    assert result["status"] == "unavailable"
    assert result["trains"] == []
    assert "not confirmed" in result["notice"].lower() or "could not" in result["notice"].lower()


def test_generated_flight_payload_is_validated_and_bounded():
    from prompting import _normalize_generated_flights

    assert _normalize_generated_flights({"flights": [{"airline": "A"}]}) == []
    assert _normalize_generated_flights(None) == []
    assert _normalize_generated_flights({"flights": "nope"}) == []
    # Implausible durations and fares are rejected rather than shown.
    bad = [{"airline": "A", "flight_number": "1", "departure_time": "1", "arrival_time": "2",
            "duration_minutes": 99999, "stops": "s", "aircraft": "a", "cabin": "c",
            "indicative_fare_inr": 100}]
    assert _normalize_generated_flights({"flights": bad}) == []
    # The list is capped so a verbose model cannot flood the UI.
    good = [{"airline": "A", "flight_number": str(i), "departure_time": "1", "arrival_time": "2",
             "duration_minutes": 60, "stops": "s", "aircraft": "a", "cabin": "c",
             "indicative_fare_inr": 100} for i in range(50)]
    assert len(_normalize_generated_flights({"flights": good})) == 8
