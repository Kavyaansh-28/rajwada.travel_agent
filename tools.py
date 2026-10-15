"""Safe allow-listed tools for the Rajasthan travel agent — expanded with real data."""
from __future__ import annotations
import os, requests
from functools import lru_cache
from urllib.parse import quote
from retrieval import retrieve_knowledge
from ingestion import load_documents
from budget import calculate_budget
from planner import compare_routes, validate_itinerary
# Imported at module level so tests can patch tools.generate_flight_options, and so
# a broken prompting module fails loudly at import instead of mid-request.
from prompting import generate_flight_options


# ── Helpers ───────────────────────────────────────────────────────────────────

def _docs():
    """Cached document accessor."""
    if not hasattr(_docs, "_cache"):
        _docs._cache = load_documents()
    return _docs._cache


def _filter_docs(category=None, destination=None, extra_fields=None):
    """Filter documents by category and/or destination."""
    results = []
    for d in _docs():
        if category and d.get("category", "").lower() != category.lower():
            continue
        if destination and destination.lower() not in d.get("destination", "").lower():
            continue
        if extra_fields:
            if not all(d.get(k) for k in extra_fields):
                continue
        results.append(d)
    return results


# ── Hotels ────────────────────────────────────────────────────────────────────

def search_hotels(city: str, tier: str = "all", top_k: int = 3) -> dict:
    """
    Return seeded hotel reference records for a city. Values are illustrative and are not live listings.
    """
    if os.getenv("GOOGLE_PLACES_API_KEY"):
        live = search_places("hotels", city)
        if live.get("source") == "Google Places":
            places = live.get("results", [])[:top_k]
            hostel_search = search_places("hostels and backpacker hostels", city)
            hostels = hostel_search.get("results", [])[:top_k] if hostel_search.get("source") == "Google Places" else []
            return {"source": "Google Places live search", "city": city, "hotels": [{"name": p.get("displayName", {}).get("text", "Hotel"), "address": p.get("formattedAddress"), "text": "Place listing from Google. Room availability, price and booking terms are not provided."} for p in places], "hostels": [{"name": p.get("displayName", {}).get("text", "Hostel"), "address": p.get("formattedAddress"), "text": "Hostel place listing. Dorm/room availability and prices are not provided."} for p in hostels], "notice": "Live place listings only; room availability and prices are not provided."}
    hotels = _filter_docs(category="hotel", destination=city)
    hostels = [h for h in hotels if "zostel" in h.get("name", "").lower() or "hostel" in h.get("name", "").lower()]
    hotels = [h for h in hotels if h not in hostels]
    if not hotels:
        # fallback guidance
        return {
            "source": "planning guidance",
            "city": city,
            "hotels": [],
            "guidance": _hotel_guidance(city, tier),
            "notice": "No specific hotel data available for this city yet; general guidance shown."
        }

    if tier != "all":
        filtered = [h for h in hotels if h.get("tier", "").lower() == tier.lower()]
        if filtered:
            hotels = filtered

    # Stable name ordering; local fixture ratings are not live-verified.
    hotels = sorted(hotels, key=lambda h: h.get("name", "").casefold())[:top_k]

    formatted = []
    for h in hotels:
        formatted.append({
            "name": h.get("name"),
            "tier": h.get("tier"),
            "price_per_night": f"₹{h.get('price_per_night_inr', 0):,}",
            "address": h.get("address"),
            "area": h.get("area"),
            "amenities": h.get("amenities", []),
            "check_in": h.get("check_in"),
            "check_out": h.get("check_out"),
            "dietary_options": h.get("dietary_options"),
            "travel_times": h.get("travel_time_to_attractions", {}),
            "text": "Illustrative property reference. Verify property identity, amenities, address, price and availability directly.",
        })

    return {
        "source": "illustrative local reference data",
        "city": city,
        "hotels": formatted,
        "hostels": [{"name": h.get("name"), "address": h.get("address"), "tier": h.get("tier"), "text": "Illustrative hostel reference; verify current details and availability directly."} for h in hostels[:top_k]],
        "guidance": _hotel_guidance(city, tier),
        "notice": "Illustrative local reference data only. Prices, ratings, amenities, addresses and availability are not live-verified; verify directly before relying on them."
    }


def _hotel_guidance(city: str, tier: str) -> str:
    tiers = {
        "budget": "For hotels, compare guesthouses and budget properties by location, recent reviews, cooling/heating, and cancellation terms. Hostels are listed separately; compare dorm/private-room options and rules directly.",
        "mid-range": "For mid-range: prioritize well-reviewed hotels in the main tourist area — compare breakfast inclusion, pickup convenience, and recent reviews. Heritage-style properties offer great value.",
        "luxury": "For luxury: heritage palaces and boutique properties offer unique experiences beyond standard 5-stars. Compare by location, inclusions, and current reviews — not brand name alone.",
        "all": "Compare stays across tiers by location, recent reviews, and inclusions. Consider heritage properties for a uniquely Rajasthani experience.",
    }
    city_note = ""
    if city.lower() in {"ranthambore", "sawai madhopur"}:
        city_note = " For Ranthambore: confirm safari zone and reporting point before booking — proximity matters enormously."
    elif city.lower() == "jaisalmer":
        city_note = " In Jaisalmer: staying inside or near the fort is atmospheric; Sam Road hotels are better for desert access."
    return tiers.get(tier, tiers["all"]) + city_note


# ── Restaurants ───────────────────────────────────────────────────────────────

def search_restaurants(city: str, cuisine: str = None, dietary: str = None, top_k: int = 3) -> dict:
    """
    Return seeded restaurant reference records. Values are illustrative and are not live listings.
    """
    if os.getenv("GOOGLE_PLACES_API_KEY"):
        live = search_places("restaurants", city)
        if live.get("source") == "Google Places":
            places = live.get("results", [])[:top_k]
            return {"source": "Google Places live search", "city": city, "results": [{"name": p.get("displayName", {}).get("text", "Restaurant"), "address": p.get("formattedAddress"), "text": "Current place listing. Check today's opening hours, menu and dietary preparation directly."} for p in places], "notice": "Live place listings only; hours, menus and dietary preparation are not verified."}
    restaurants = _filter_docs(category="restaurant", destination=city)

    if cuisine:
        restaurants = [r for r in restaurants if cuisine.lower() in r.get("cuisine", "").lower() + r.get("text", "").lower()]
    if dietary:
        restaurants = [r for r in restaurants if dietary.lower() in r.get("dietary", "").lower() + r.get("text", "").lower()]

    restaurants = sorted(restaurants, key=lambda r: r.get("name", "").casefold())[:top_k]

    if not restaurants:
        return {
            "source": "planning guidance",
            "city": city,
            "results": [],
            "notice": f"No specific restaurant data for {city}. Seek a well-reviewed local thali restaurant near your stay. Confirm dietary needs directly."
        }

    formatted = []
    for r in restaurants:
        formatted.append({
            "name": r.get("name"),
            "cuisine": r.get("cuisine"),
            "dietary": r.get("dietary"),
            "price_level": r.get("price_level"),
            "avg_meal_inr": r.get("avg_meal_inr"),
            "address": r.get("address"),
            "hours": r.get("hours"),
            "famous_for": r.get("famous_for", []),
            "text": "Illustrative venue reference. Verify menus, opening hours, prices, venue details and dietary preparation directly.",
        })

    return {
        "source": "illustrative local reference data",
        "city": city,
        "results": formatted,
        "notice": "Illustrative local reference data only. Ratings, menus, hours and prices are not live-verified. Confirm current details and dietary preparation directly."
    }


# ── Attractions ───────────────────────────────────────────────────────────────

def search_attractions(city: str, category: str = None, top_k: int = 4) -> dict:
    """
    Return locally curated attraction guidance; time-sensitive details require verification.
    """
    if os.getenv("GOOGLE_PLACES_API_KEY"):
        live = search_places("tourist attractions", city)
        if live.get("source") == "Google Places":
            places = live.get("results", [])[:top_k]
            return {"source": "Google Places live search", "city": city, "attractions": [{"name": p.get("displayName", {}).get("text", "Place"), "address": p.get("formattedAddress"), "status": "Place listing only; access, fees and operating hours are not verified."} for p in places], "notice": "Live place listings only; confirm official access details before visiting."}
    attractions = [d for d in _docs() if d.get("destination", "").lower() == city.lower()
                   and d.get("category") not in {"hotel", "restaurant", "accommodation", "general", "planning guidance"}]

    if category:
        attractions = [a for a in attractions if category.lower() in a.get("category", "").lower()]

    attractions = sorted(attractions, key=lambda a: (a.get("title") or a.get("name", "")).casefold())[:top_k]

    formatted = []
    for a in attractions:
        formatted.append({
            "name": a.get("title") or a.get("name"),
            "category": a.get("category"),
            "entry_fee_indian": f"₹{a.get('entry_fee_inr', 0)}",
            "entry_fee_foreigner": f"₹{a.get('entry_fee_foreigner_inr', 0)}",
            "timings": a.get("timings"),
            "best_time": a.get("best_time"),
            "duration_hours": a.get("duration_hours"),
            "highlights": a.get("highlights", []),
            "tips": a.get("tips", []),
            "status": "Verify current access and operating status with an official source.",
        })

    return {
        "source": "curated local reference data",
        "city": city,
        "attractions": formatted,
        "notice": "Locally curated reference data. Entry fees, timings and access are not live-verified; check current official guidance."
    }


# ── Cab / Transport ───────────────────────────────────────────────────────────

INTERCITY_ROUTES = {
    ("Jaipur", "Udaipur"): {"distance_km": 393, "road_hours": 6, "train_hours": 7, "cab_inr": 3500, "bus_inr": 400, "train_inr": 350},
    ("Jaipur", "Jodhpur"): {"distance_km": 345, "road_hours": 5, "train_hours": 4.5, "cab_inr": 3000, "bus_inr": 350, "train_inr": 280},
    ("Jaipur", "Jaisalmer"): {"distance_km": 575, "road_hours": 8, "train_hours": 10, "cab_inr": 5000, "bus_inr": 550, "train_inr": 400},
    ("Jaipur", "Pushkar"): {"distance_km": 146, "road_hours": 2.5, "train_hours": None, "cab_inr": 1500, "bus_inr": 150, "train_inr": None},
    ("Jaipur", "Ranthambore"): {"distance_km": 180, "road_hours": 3, "train_hours": 2, "cab_inr": 1800, "bus_inr": None, "train_inr": 200},
    ("Jaipur", "Mount Abu"): {"distance_km": 490, "road_hours": 7.5, "train_hours": 8, "cab_inr": 4500, "bus_inr": 500, "train_inr": 380},
    ("Udaipur", "Jodhpur"): {"distance_km": 270, "road_hours": 4.5, "train_hours": 6, "cab_inr": 2800, "bus_inr": 300, "train_inr": 260},
    ("Udaipur", "Mount Abu"): {"distance_km": 185, "road_hours": 3, "train_hours": None, "cab_inr": 2000, "bus_inr": 200, "train_inr": None},
    ("Jodhpur", "Jaisalmer"): {"distance_km": 285, "road_hours": 4.5, "train_hours": 5.5, "cab_inr": 2800, "bus_inr": 280, "train_inr": 250},
    ("Jodhpur", "Bikaner"): {"distance_km": 250, "road_hours": 4, "train_hours": 5, "cab_inr": 2500, "bus_inr": 250, "train_inr": 220},
    ("Ranthambore", "Jaipur"): {"distance_km": 180, "road_hours": 3, "train_hours": 2, "cab_inr": 1800, "bus_inr": None, "train_inr": 200},
    ("Ranthambore", "Agra"): {"distance_km": 250, "road_hours": 4, "train_hours": 3.5, "cab_inr": 2500, "bus_inr": None, "train_inr": 280},
}

def get_cab_rates(from_city: str, to_city: str) -> dict:
    """Return intercity transport options and costs."""
    key = (from_city, to_city)
    rev_key = (to_city, from_city)
    route = INTERCITY_ROUTES.get(key) or INTERCITY_ROUTES.get(rev_key)

    if not route:
        return {
            "from": from_city, "to": to_city,
            "message": f"No specific route data for {from_city} → {to_city}. Approximate private cab: ₹15-18/km. Verify current options.",
            "source": "planning guidance"
        }

    options = []
    if route.get("train_inr"):
        options.append({"mode": "Train", "duration": f"{route['train_hours']} hrs", "cost": f"₹{route['train_inr']}–{route['train_inr']*2} (sleeper to AC)", "book_at": "irctc.co.in"})
    options.append({"mode": "Private Cab (AC)", "duration": f"{route['road_hours']} hrs", "cost": f"₹{route['cab_inr']:,}", "book_at": "Hotel desk, Ola, local operators"})
    if route.get("bus_inr"):
        options.append({"mode": "Rajasthan State Bus / Volvo", "duration": f"{route['road_hours']+0.5} hrs", "cost": f"₹{route['bus_inr']}–{int(route['bus_inr']*2.5)}", "book_at": "rsrtc.rajasthan.gov.in or bus stand"})

    return {
        "from": from_city, "to": to_city,
        "distance_km": route["distance_km"],
        "options": options,
        "recommendation": "Train is most comfortable and punctual for longer routes. Cab offers flexibility for multi-stop days.",
        "notice": "Prices are estimates — verify current fares before booking.",
        "source": "local transport knowledge"
    }

def _railradar_key() -> str:
    """Accept either the raw token or a pasted Authorization: Bearer header."""
    value = os.getenv("IRCTC_API_KEY", "").strip()
    if value.lower().startswith("authorization:"):
        value = value.split(":", 1)[1].strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if not value or "your_api_key" in value.lower() or "…" in value:
        return ""
    return value


def _railradar_get(path: str, params: dict | None = None) -> dict:
    key = _railradar_key()
    base = os.getenv("IRCTC_API_BASE_URL", "https://api.railradar.in/v1").strip().rstrip("/")
    if not key or not base:
        return {"status": "not_configured"}
    try:
        response = requests.get(
            f"{base}/{path.lstrip('/')}",
            params=params or {},
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("success") is not True:
            return {"status": "provider_error"}
        return {"status": "ok", "data": payload.get("data"), "meta": payload.get("meta", {})}
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        return {"status": "provider_http_error", "http_status": status}
    except (requests.RequestException, ValueError):
        return {"status": "unavailable"}


@lru_cache(maxsize=128)
def _railradar_station_search(query: str) -> tuple[dict, ...]:
    result = _railradar_get("lookup/search/stations", {"q": query, "limit": 10})
    rows = result.get("data") if result.get("status") == "ok" else None
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict))


def _railradar_station(query: str) -> dict | None:
    rows = _railradar_station_search(query.strip())
    if not rows:
        return None
    wanted = query.casefold().strip()
    def score(row: dict) -> tuple[int, str]:
        city = str(row.get("city", "")).casefold().strip()
        name = str(row.get("name", "")).casefold().strip()
        code = str(row.get("code", "")).casefold().strip()
        points = 5 if city == wanted else 4 if name == wanted else 3 if code == wanted else 2 if city.startswith(wanted) else 1
        return points, code
    best = max(enumerate(rows), key=lambda item: (score(item[1])[0], -item[0]))[1]
    return best if best.get("code") and score(best)[0] > 1 else None


def _railradar_trains(from_city: str, to_city: str, travel_date: str | None = None) -> dict:
    """Find scheduled trains (with live departure overlay) using RailRadar."""
    origin = _railradar_station(from_city)
    destination = _railradar_station(to_city)
    if not origin or not destination:
        return {"status": "station_not_found", "trains": []}
    params = {"byCity": "true", "live": "true"}
    if travel_date:
        params["date"] = travel_date
    result = _railradar_get(
        f"trains/between/{quote(str(origin['code']), safe='')}/{quote(str(destination['code']), safe='')}",
        params,
    )
    if result.get("status") != "ok":
        return {**result, "from": origin, "to": destination, "trains": []}
    data = result.get("data") or {}
    trains = data.get("trains", []) if isinstance(data, dict) else []
    return {
        "status": "ok", "source": "RailRadar", "from": data.get("from", origin),
        "to": data.get("to", destination), "trains": [row for row in trains if isinstance(row, dict)],
        "date": travel_date, "trace_id": (result.get("meta") or {}).get("traceId"),
    }


def get_transport_options(from_city: str, to_city: str, travel_date: str | None = None) -> dict:
    """Return local road estimates plus live RailRadar train schedules when configured."""
    def title(value: str) -> str:
        return " ".join(word.capitalize() for word in value.strip().split())

    origin, destination = title(from_city), title(to_city)
    route = get_cab_rates(origin, destination)
    rail = _railradar_trains(origin, destination, travel_date) if _railradar_key() else {"status": "not_configured", "trains": []}
    live_trains = []
    for row in rail.get("trains", [])[:8]:
        train = row.get("train") or {}
        departure = row.get("from") or {}
        arrival = row.get("to") or {}
        live = row.get("live") or {}
        live_trains.append({
            "number": train.get("number"), "name": train.get("name"), "type": train.get("type"),
            "departure": departure.get("departure"), "arrival": arrival.get("arrival"),
            "duration_minutes": row.get("duration"), "date": travel_date,
            "status": live.get("type"), "delay_minutes": live.get("delayMinutes"),
            "distance_km": row.get("distance"),
        })
    local_options = list(route.get("options", []))
    if live_trains:
        local_options = [option for option in local_options if option.get("mode") != "Train"]
    if live_trains:
        notice = "Live train timetable/status from RailRadar; this is not ticket availability or a booking quote. Road fares are local planning estimates."
    elif rail.get("status") == "not_configured":
        notice = route.get("notice") or "Train API is not configured; schedules and availability must be checked with the operator."
    else:
        notice = "Live train lookup did not return results; schedules and ticket availability are not confirmed. Road fares are planning estimates."
    return {
        "from": origin,
        "to": destination,
        "options": local_options,
        "live_trains": live_trains,
        "rail_status": rail.get("status", "not_configured"),
        "rail_source": "RailRadar" if live_trains else None,
        "flight": {"status": "provider_required", "message": "Flight schedules and fares are not connected yet; check a flight booking provider for live options."},
        "notice": notice,
        "distance_km": route.get("distance_km"),
        "recommendation": route.get("recommendation"),
        "source": "RailRadar + local estimates" if live_trains else route.get("source", "planning guidance"),
        "live_schedules": bool(live_trains),
    }


def get_train_options(from_city: str, to_city: str, travel_date: str | None = None) -> dict:
    """Return trains only: live RailRadar schedules, else local timetable estimates.

    Never mixes in flights, buses or cabs — a train request returns trains.
    """
    def title(value: str) -> str:
        return " ".join(word.capitalize() for word in value.strip().split())

    origin, destination = title(from_city), title(to_city)
    # "options" is always present and always empty: a train request must not carry
    # road or air alternatives into the response.
    base = {"from": origin, "to": destination, "mode": "trains", "travel_date": travel_date,
            "options": [], "trains": []}
    if not _railradar_key():
        return {**base, "status": "not_configured", "live": False,
                "notice": "The live train feed (RailRadar) is not configured, so I have no verified timetable for "
                          "this route. Check the operator or a booking site for current schedules."}

    rail = _railradar_trains(origin, destination, travel_date)
    trains = []
    for row in (rail.get("trains") or [])[:10]:
        train = row.get("train") or {}
        departure = row.get("from") or {}
        arrival = row.get("to") or {}
        live = row.get("live") or {}
        trains.append({
            "number": train.get("number"), "name": train.get("name"), "type": train.get("type"),
            "departure": departure.get("departure"), "arrival": arrival.get("arrival"),
            "duration_minutes": row.get("duration"), "date": travel_date,
            "status": live.get("type"), "delay_minutes": live.get("delayMinutes"),
            "distance_km": row.get("distance"),
        })

    if rail.get("status") == "ok" and trains:
        return {**base, "status": "ok", "live": True, "trains": trains,
                "source": "RailRadar live timetable",
                "rail_status": rail.get("status"),
                "notice": "Live timetable and running status from RailRadar. This is not ticket availability, seat "
                          "confirmation or a fare quote — book through the operator."}

    reason = {
        "station_not_found": "I could not match both cities to railway stations, so I have no live timetable for "
                             "this route.",
        "provider_http_error": "The live train feed rejected the request.",
        "unavailable": "The live train feed could not be reached.",
        "provider_error": "The live train feed returned an error.",
    }.get(rail.get("status"), "The live train feed did not return a timetable for this route.")
    return {**base, "status": rail.get("status", "unavailable"), "live": False, "trains": [],
            "rail_status": rail.get("status"),
            "notice": f"{reason} Check the operator or a booking site for current schedules and fares."}


# ── Accommodation (legacy wrapper) ───────────────────────────────────────────

def search_accommodation(city: str, style: str = "balanced") -> dict:
    """Legacy wrapper — delegates to search_hotels with tier mapping."""
    tier_map = {"budget": "budget", "balanced": "mid-range", "premium": "luxury"}
    tier = tier_map.get(style.lower(), "all")
    return search_hotels(city, tier=tier)


# ── Ranking ──────────────────────────────────────────────────────────────────

def rank_recommendations(items: list, preferences: str = None) -> list:
    """Rank a list of items against user preferences."""
    if not preferences:
        return items
    pref_words = [w for w in preferences.lower().split() if len(w) > 2]
    ranked = []
    for item in items:
        text = " ".join(str(v) for v in item.values()).lower()
        score = sum(1 for w in pref_words if w in text)
        ranked.append({**item, "match_score": score, "why": "Matches your stated preferences."})
    return sorted(ranked, key=lambda x: x["match_score"], reverse=True)


# ── Live Places (Google) ─────────────────────────────────────────────────────

def search_places(query: str, city: str = None) -> dict:
    """Try Google Places API; fall back to local data if unconfigured."""
    key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not key:
        local = [d for d in _docs() if
                 (not city or d.get("destination", "").lower() == city.lower()) and
                 query.lower() in (d.get("title", "") + d.get("text", "")).lower()][:5]
        return {"source": "local fallback", "results": [{k: d.get(k) for k in ("title", "name", "destination", "category", "text", "source_type", "source_url")} for d in local], "notice": "Google Places not configured; showing illustrative local reference data."}
    try:
        r = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": "places.displayName,places.formattedAddress"},
            json={"textQuery": f"{query} {city or 'Rajasthan'}"},
            timeout=8
        )
        r.raise_for_status()
        return {"source": "Google Places", "results": r.json().get("places", [])}
    except requests.RequestException as e:
        local = [d for d in _docs() if (not city or d.get("destination", "").lower() == city.lower()) and query.lower() in (d.get("title", "") + d.get("text", "")).lower()][:5]
        safe = [{k: d.get(k) for k in ("title", "name", "destination", "category", "text", "source_type", "source_url")} for d in local]
        return {"source": "local fallback", "results": safe, "notice": f"Live search unavailable ({e.__class__.__name__}); showing illustrative local reference data."}


# ── Live aviation data (Aviationstack) ───────────────────────────────────────

def _aviationstack_base() -> str:
    """Aviationstack base URL; overridable via AVIATIONSTACK_API_BASE_URL."""
    return os.getenv("AVIATIONSTACK_API_BASE_URL", "https://api.aviationstack.com/v1").strip().rstrip("/")


def search_aviation_airports(city: str) -> dict:
    """Look up airports by city/name through Aviationstack; never fabricate live data."""
    key = os.getenv("AVIATIONSTACK_API_KEY", "").strip()
    if not key:
        return {"status": "not_configured", "source": "Aviationstack"}
    try:
        response = requests.get(
            f"{_aviationstack_base()}/airports",
            params={"access_key": key, "search": city, "limit": 10}, timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            return {"status": "provider_error", "source": "Aviationstack"}
        airports = payload.get("data", [])
        return {"status": "ok", "source": "Aviationstack", "airports": [
            {k: item.get(k) for k in ("airport_name", "iata_code", "icao_code", "city", "country_name", "latitude", "longitude")}
            for item in airports if isinstance(item, dict)
        ]}
    except requests.RequestException as exc:
        return {"status": "unavailable", "source": "Aviationstack", "error_type": type(exc).__name__}


def search_aviation_flights(origin_iata: str, destination_iata: str) -> dict:
    """Return Aviationstack real-time flight status for an airport pair."""
    key = os.getenv("AVIATIONSTACK_API_KEY", "").strip()
    if not key:
        return {"status": "not_configured", "source": "Aviationstack"}
    try:
        response = requests.get(
            f"{_aviationstack_base()}/flights",
            params={"access_key": key, "dep_iata": origin_iata, "arr_iata": destination_iata, "limit": 10}, timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            return {"status": "provider_error", "source": "Aviationstack"}
        flights = payload.get("data", [])
        return {"status": "ok", "source": "Aviationstack", "flights": [
            {"flight_date": item.get("flight_date"), "flight_status": item.get("flight_status"),
             "flight": item.get("flight", {}).get("iata"), "airline": item.get("airline", {}).get("name"),
             "departure": {k: item.get("departure", {}).get(k) for k in ("airport", "iata", "scheduled", "estimated", "actual")},
             "arrival": {k: item.get("arrival", {}).get(k) for k in ("airport", "iata", "scheduled", "estimated", "actual")}}
            for item in flights if isinstance(item, dict)
        ]}
    except requests.RequestException as exc:
        return {"status": "unavailable", "source": "Aviationstack", "error_type": type(exc).__name__}


def _aviationstack_key() -> str:
    return os.getenv("AVIATIONSTACK_API_KEY", "").strip()


def _pick_airport(payload, preferred_name: str | None) -> dict | None:
    """Choose the best IATA match for a city, preferring an exact name/city hit."""
    airports = payload.get("airports") or []
    if not airports:
        return None
    wanted = (preferred_name or "").casefold().strip()
    if not wanted:
        return next((a for a in airports if a.get("iata_code")), None)
    for field in ("airport_name", "city"):
        for airport in airports:
            if str(airport.get(field) or "").casefold().strip() == wanted:
                return airport
    for airport in airports:
        if str(airport.get("city") or "").casefold().startswith(wanted):
            return airport
    return next((a for a in airports if a.get("iata_code")), None)


def _local_airport_table() -> list[dict]:
    """Airport reference rows from data/airports.json (IATA codes and coordinates)."""
    if not hasattr(_local_airport_table, "_cache"):
        import json
        from pathlib import Path
        path = Path(__file__).parent / "data" / "airports.json"
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = []
        _local_airport_table._cache = [row for row in rows if isinstance(row, dict) and row.get("iata")]
    return _local_airport_table._cache


def _match_local_airport(name: str) -> dict | None:
    """Resolve a city or airport name against the local table.

    IATA codes and coordinates are public reference facts, so this stays usable
    when the live aviation feed is unconfigured or plan-restricted.
    """
    raw = str(name or "").strip()
    wanted = raw.casefold()
    if not wanted:
        return None
    rows = _local_airport_table()
    # A bare three-letter uppercase token is an IATA code, not a city name. This is
    # checked against the original casing because the comparison form is casefolded.
    if raw.isupper() and len(raw) == 3:
        return next((row for row in rows if row["iata"].casefold() == wanted), None)
    for row in rows:
        if str(row.get("city", "")).casefold() == wanted or str(row.get("name", "")).casefold() == wanted:
            return row
    for row in rows:
        if str(row.get("city", "")).casefold().startswith(wanted):
            return row
        if wanted in str(row.get("name", "")).casefold():
            return row
    return None


def _resolve_airport(name: str) -> dict | None:
    """Look up one airport by city/name; prefer the live feed, fall back to data/."""
    payload = search_aviation_airports(name)
    airport = _pick_airport(payload, name) if payload.get("status") == "ok" else None
    if not airport or not airport.get("iata_code"):
        local = _match_local_airport(name)
        if not local:
            return None
        airport = {"airport_name": local.get("name"), "iata_code": local.get("iata"),
                   "city": local.get("city"), "country_name": local.get("country"),
                   "latitude": local.get("latitude"), "longitude": local.get("longitude")}
    try:
        latitude, longitude = float(airport["latitude"]), float(airport["longitude"])
    except (KeyError, TypeError, ValueError):
        latitude = longitude = None
    return {"iata": airport["iata_code"], "name": airport.get("airport_name"),
            "city": airport.get("city"), "country": airport.get("country_name"),
            "latitude": latitude, "longitude": longitude, "resolved_from": "local reference"}


def _estimate_flight_options(origin: str, destination: str, origin_airport: dict | None,
                             destination_airport: dict | None) -> list[dict]:
    """Deterministic non-stop estimate from great-circle distance between the airports.

    This invents no airline, flight number or schedule. It only states what the
    geometry and a standard cruise speed imply, which is the same class of
    planning assumption as the cab and budget estimates.
    """
    try:
        lat1, lon1 = float(origin_airport["latitude"]), float(origin_airport["longitude"])
        lat2, lon2 = float(destination_airport["latitude"]), float(destination_airport["longitude"])
    except (KeyError, TypeError, ValueError):
        return []
    from math import asin, cos, radians, sin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    haversine = 2 * asin(sqrt(sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2))
    distance_km = haversine * 6371
    if not 50 <= distance_km <= 6000:
        return []
    # ~800 km/h block speed including climb/descent, plus taxi time.
    minutes = int(round(distance_km / 800 * 60 + 35))
    return [{
        "airline": None, "flight_number": None,
        "departure_time": None, "arrival_time": None,
        "duration_minutes": minutes,
        "stops": "Direct (estimated)", "aircraft": None, "cabin": None,
        "indicative_fare_inr": int(round((1800 + distance_km * 4.2) / 100) * 100),
        "distance_km": round(distance_km),
    }]


def get_flight_options(origin: str, destination: str, travel_date: str | None = None) -> dict:
    """Return flights only: live Aviationstack data first, then clearly-labelled fallbacks.

    Never mixes in trains, buses or cabs — a flight request returns flights. The
    fallback chain is unconditional, so a restricted or missing aviation key still
    produces options: live -> generated representative rows -> great-circle
    estimate -> honest "not available".
    """

    def title(value: str) -> str:
        return " ".join(word.capitalize() for word in str(value).strip().split())

    origin_name, destination_name = title(origin), title(destination)
    base = {"from": origin_name, "to": destination_name, "mode": "flights", "travel_date": travel_date}
    unverified = ("These are representative planning options only — flight numbers, timings and fares are not "
                  "verified, and they are not a booking quote or a seat-availability claim. Confirm with the "
                  "airline or a booking provider before travelling.")

    # _resolve_airport already falls back to data/airports.json, so coordinates and
    # IATA codes are available even with no Aviationstack key or plan access. Only
    # the *live* flight call requires the key.
    origin_airport = _resolve_airport(origin_name)
    destination_airport = _resolve_airport(destination_name)
    origin_iata = origin_airport["iata"] if origin_airport else None
    destination_iata = destination_airport["iata"] if destination_airport else None

    if origin_iata and destination_iata and _aviationstack_key():
        live = search_aviation_flights(origin_iata, destination_iata)
        if live.get("status") == "ok" and live.get("flights"):
            return {**base, "status": "ok", "source": "Aviationstack live flight data", "live": True,
                    "origin_iata": origin_iata, "destination_iata": destination_iata,
                    "origin_airport": origin_airport, "destination_airport": destination_airport,
                    "flights": live["flights"],
                    "notice": "Live status from Aviationstack for today's operations. Fares and seat availability "
                              "are not included; confirm with the airline before booking."}

    estimate = _estimate_flight_options(origin_name, destination_name, origin_airport, destination_airport)
    generated, provider = generate_flight_options(origin_name, destination_name, travel_date,
                                                  origin_iata, destination_iata)
    if generated:
        reason = (f"No live flight was returned for {origin_iata} → {destination_iata}."
                  if origin_iata and destination_iata and _aviationstack_key() else
                  "No live flight feed is available for this route.")
        return {**base, "status": "generated", "source": f"AI-generated representative schedule ({provider})",
                "live": False, "origin_iata": origin_iata, "destination_iata": destination_iata,
                "origin_airport": origin_airport, "destination_airport": destination_airport,
                "flights": generated, "flight_estimate": estimate or None,
                "notice": f"{reason} {unverified}"}
    if estimate:
        return {**base, "status": "estimated", "source": "Great-circle route estimate", "live": False,
                "origin_iata": origin_iata, "destination_iata": destination_iata,
                "origin_airport": origin_airport, "destination_airport": destination_airport,
                "flights": estimate,
                "notice": "Computed from the great-circle distance between the two airports. No airline, flight "
                          "number or schedule is claimed — check the airline for actual departures."}
    return {**base, "status": "unavailable", "live": False, "flights": [],
            "origin_iata": origin_iata, "destination_iata": destination_iata,
            "notice": "I could not retrieve, generate or estimate flights for this route. Check the airline or a "
                      "booking provider directly."}


# ── Budget ────────────────────────────────────────────────────────────────────

def estimate_travel_costs(*args, **kwargs):
    return calculate_budget(*args, **kwargs)
