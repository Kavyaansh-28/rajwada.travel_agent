"""Full-featured Rajasthan travel agent — like a real human travel agent."""
from __future__ import annotations
import re
from datetime import date
from retrieval import retrieve_knowledge
from planner import plan_itinerary, compare_routes, validate_itinerary, CITY_STOPS
from tools import (
    search_restaurants, search_hotels, search_attractions,
    get_cab_rates, get_transport_options, rank_recommendations, search_accommodation, search_places,
    search_aviation_airports, search_aviation_flights
)


HINDI_CITY_NAMES = {
    "Jaipur": "जयपुर", "Udaipur": "उदयपुर", "Jodhpur": "जोधपुर", "Jaisalmer": "जैसलमेर",
    "Pushkar": "पुष्कर", "Ranthambore": "रणथंभौर", "Sawai Madhopur": "सवाई माधोपुर",
    "Mount Abu": "माउंट आबू", "Ajmer": "अजमेर", "Bikaner": "बीकानेर", "Bundi": "बूंदी",
    "Chittorgarh": "चित्तौड़गढ़", "Kumbhalgarh": "कुंभलगढ़", "Bharatpur": "भरतपुर",
    "Shekhawati": "शेखावाटी", "Alwar": "अलवर", "Barmer": "बाड़मेर", "Osian": "ओसियान",
    "Nathdwara": "नाथद्वारा", "Kota": "कोटा", "Dungarpur": "डूंगरपुर", "Jhalawar": "झालावाड़",
    "Sikar": "सीकर", "Tonk": "टोंक", "Karauli": "करौली", "MountAbu": "माउंट आबू",
}


def detect_response_language(message: str, requested: str | None = None) -> str:
    """Prefer Hindi when the traveler writes Hindi; keep the explicit Hindi UI choice."""
    if requested == "hindi_devanagari":
        return requested
    if re.search(r"[\u0900-\u097f]", message):
        return "hindi_devanagari"
    # Small, conservative Hinglish signal: avoid switching on a single ambiguous
    # word such as "hai" in an otherwise English sentence.
    tokens = set(re.findall(r"[a-z]+", message.lower()))
    hindi_markers = {"aur", "hai", "hain", "mujhe", "hum", "karo", "karna", "kitna", "kripya", "mein", "ke", "liye", "nahi", "paisa", "raha", "sasta", "se", "safar", "thoda", "yatra", "chahiye", "banao", "batao", "din"}
    hits = tokens & hindi_markers
    return "hindi_devanagari" if len(hits) >= 2 or (hits and len(tokens) <= 4) else "english"


def expand_multilingual_query(message: str, cities: list[str]) -> str:
    """Add transparent English travel-domain terms so Hindi queries can hit the English corpus."""
    lower = message.casefold()
    concepts = {
        "visit attractions sightseeing places": ("देखें", "देखना", "घूमने", "घूमना", "दर्शनीय", "जगह", "ghoom", "dekh"),
        "food restaurants cuisine": ("खाना", "खाने", "भोजन", "रेस्टोरेंट", "khana", "khane"),
        "hotel stay accommodation": ("रहने", "ठहरने", "रुकने", "hotel", "rehne", "thaharne"),
        "bus train transport route": ("बस", "ट्रेन", "किराया", "रूट", "bus", "train", "safar"),
        "budget cost price": ("बजट", "कीमत", "खर्च", "बजट", "budget", "kharcha", "paisa"),
        "wildlife safari park": ("सफारी", "वन्यजीव", "बाघ", "safari", "tiger"),
    }
    additions = [terms for terms, markers in concepts.items() if any(marker in lower for marker in markers)]
    return " ".join([message, *cities, *additions])


# ── Transport mode detection ────────────────────────────────────────────────
# Terms are matched on word boundaries. Plain substring matching is unsafe in this
# codebase's languages: "सबसे" contains "बस" (bus) and "airport" contains "air",
# so a Hindi "सबसे अच्छी उड़ान" (best flight) used to register as a bus request.

TRANSPORT_MODE_TERMS = {
    "flights": ("flight", "flights", "fly", "flying", "airline", "airlines", "plane", "airport",
                "direct flight", "nonstop", "non-stop",
                "उड़ान", "उड़ानें", "उडान", "हवाई", "हवाईजहाज", "एयरपोर्ट", "विमान", "जहाज"),
    "trains": ("train", "trains", "railway", "rail", "ट्रेन", "रेल", "रेलवे"),
    "buses": ("bus", "buses", "coach", "बस"),
    "cabs": ("cab", "cabs", "taxi", "auto rickshaw", "कैब", "ऑटो", "कार"),
}

# Words that mean "show me transport options for a route". This is deliberately
# narrower than TRANSPORT_MODE_TERMS: "which airport is nearest to Ranthambore"
# names an airport but asks about arrival points, not about flights, so it must
# not be captured as a flight-booking request.
TRANSPORT_INTENT_TERMS = (
    "flight", "flights", "fly", "flying", "airline", "airlines", "plane", "planes", "airways",
    "nonstop", "non-stop", "train", "trains", "railway", "bus", "buses", "coach",
    "cab", "cabs", "taxi", "how to get", "how do i get", "how to reach", "how to travel",
    "getting from", "travel from", "best way to get", "कैसे पहुँच", "कैसे जाएं", "कैसे जाऊं", "किराया",
    "उड़ान", "उड़ानें", "उडान", "हवाईजहाज", "विमान", "जहाज", "ट्रेन", "रेल", "रेलवे", "बस", "कैब", "ऑटो",
)

_TERM_BOUNDARY = r"[A-Za-z0-9\u0900-\u097F]"
# Devanagari combining marks (matras, viramas, anusvara) may trail a term without
# making it part of a longer word: "ट्रेनें" is the plural of "ट्रेन". Only
# consonants and independent vowels actually start a new word.
_TERM_NOT_BEFORE = r"[A-Za-z0-9\u0904-\u0939\u0958-\u095F\u0978-\u097F]"


def _term_present(lower_text: str, term: str) -> bool:
    """True when `term` appears as a standalone token, not inside a longer word."""
    pattern = rf"(?<!{_TERM_BOUNDARY}){re.escape(term)}(?!{_TERM_NOT_BEFORE})"
    return re.search(pattern, lower_text) is not None


_PLACE_SUFFIXES = {"की", "का", "के", "को", "में", "से", "और", "लिए", "वाले", "वाली", "वाला", "तक", "है", "हैं"}

# Words that can precede "to" without being an origin city ("I want to go to…").
_ROUTE_STOPWORDS = {
    "i", "we", "you", "it", "they", "he", "she", "my", "our", "this", "that", "want", "wanna", "need",
    "like", "plan", "going", "go", "goes", "to", "from", "prefer", "travel", "travelling", "traveling",
    "flying", "fly", "and", "or", "but", "then", "also", "now", "please", "just", "how", "when", "where",
    "best", "cheapest", "fastest", "morning", "evening", "night", "today", "tomorrow", "next",
    "budget", "trip", "cheaper", "cost", "price", "distance", "journey", "route", "schedule",
}

# Short conversational turns carry no task. They must never inherit the previous
# turn's routed intent — "hi" after a flight question was answering with trains.
SMALL_TALK = re.compile(
    r"^\s*(?:(?:hi+|hey+|hello+|yo+|namaste|नमस्ते|नमस्कार)(?:\s+there)?"
    r"|(?:ok(?:ay)?|k|kk|yes|yeah|yep|yup|no|nope|nah|sure|right|correct|exactly|fine|great|nice|cool|good"
    r"|thanks?|thank\s*you|thx|cheers|bye|goodbye|see\s*you|got\s*it|got\s*it|understood|noted"
    r"|done|perfect|wonderful|शुक्रिया|धन्यवाद|ठीक\s*है|हाँ|हां|नहीं)"
    r"[\s,.!?।]*)+$",
    re.I)


def _all_mode_terms() -> list[str]:
    return [term for terms in TRANSPORT_MODE_TERMS.values() for term in terms]


def _is_mode_word(token: str) -> bool:
    lowered = token.lower()
    return any(_term_present(lowered, term) for term in _all_mode_terms())


def _place_head(chunk: str | None) -> str | None:
    """Place words at the START of a chunk, stopping at the first mode word.

    Used for destinations, where the mode word trails the city
    ("उदयपुर ट्रेनें बताएं" → "उदयपुर").
    """
    if not chunk:
        return None
    words = chunk.split()
    kept = []
    for word in words:
        if _is_mode_word(word):
            break
        kept.append(word)
    while kept and kept[0].lower().strip(".,।") in _ROUTE_STOPWORDS:
        kept.pop(0)
    while kept and kept[-1].lower().strip(".,।") in _PLACE_SUFFIXES:
        kept.pop()
    return " ".join(kept).strip(" ,।?") or None


def _place_tail(chunk: str | None) -> str | None:
    """Place words at the END of a chunk, dropping everything up to the last mode word.

    Used for origins, where request words lead the city
    ("direct flights delhi" → "delhi").
    """
    if not chunk:
        return None
    words = chunk.split()
    cut = 0
    for index, word in enumerate(words):
        if _is_mode_word(word):
            cut = index + 1
    kept = words[cut:]
    while kept and kept[0].lower().strip(".,।") in _ROUTE_STOPWORDS:
        kept.pop(0)
    while kept and kept[-1].lower().strip(".,।") in _PLACE_SUFFIXES:
        kept.pop()
    return " ".join(kept).strip(" ,।?") or None


def detect_transport_mode(text: str) -> str:
    """Return the single transport mode the traveler asked for, or 'any'.

    A request naming one mode is answered with that mode only. Naming several, or
    none, falls back to 'any' so the full comparison table is still offered.
    """
    lower = text.lower()
    named = [mode for mode, terms in TRANSPORT_MODE_TERMS.items() if any(_term_present(lower, term) for term in terms)]
    return named[0] if len(named) == 1 else "any"


def parse_route_intent(text: str) -> tuple[str | None, str | None]:
    """Extract an explicit 'from A to B' / 'A to B' / 'between A and B' route."""
    english = re.search(
        r"\bfrom\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s-]{0,35}?)\s+(?:to|→)\s+"
        r"([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s-]{0,35}?)(?=[,.!?]|\s+(?:for|on|please|और|का|की|को)\b|$)",
        text, re.I)
    if english:
        return _place_tail(english.group(1)), _place_head(english.group(2))
    between = re.search(
        r"\bbetween\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s-]{0,30}?)\s+and\s+"
        r"([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s-]{0,30}?)(?=[,.!?]|\s+(?:for|on|please)\b|$)",
        text, re.I)
    if between:
        origin, destination = _place_tail(between.group(1)), _place_head(between.group(2))
        if origin and destination:
            return origin, destination
    # "flights delhi to jaipur" — the origin is stated without a "from".
    bare = re.search(
        r"(?:^|\s)([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]{0,24}?)\s+to\s+"
        r"([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s-]{0,35}?)(?=[,.!?]|\s+(?:for|on|please)\b|$)",
        text)
    if bare:
        origin, destination = _place_tail(bare.group(1)), _place_head(bare.group(2))
        if origin and destination and len(origin.split()) <= 3:
            return origin, destination
    hindi = re.search(
        r"([\u0900-\u097F][\u0900-\u097F\s]{1,24}?)\s+से\s+(?:उड़ान|ट्रेन|बस|कैब)?\s*"
        r"([\u0900-\u097F][\u0900-\u097F\s]{1,24}?)(?=[,.!?]|\s+(?:की|का|के|को|और|में)\b|$)",
        text)
    if hindi:
        return _place_tail(hindi.group(1)), _place_head(hindi.group(2))
    return None, None


class RajasthanAgent:
    def __init__(self):
        self.max_calls = 16

    def _cities(self, text: str, existing=None) -> list:
        lowered = text.lower()
        aliases = {
            "Jaipur": ["जयपुर"], "Udaipur": ["उदयपुर"], "Jodhpur": ["जोधपुर"],
            "Jaisalmer": ["जैसलमेर"], "Pushkar": ["पुष्कर"], "Ranthambore": ["रणथंभौर", "रणथम्बोर"],
            "Sawai Madhopur": ["सवाई माधोपुर"], "Mount Abu": ["माउंट आबू"], "Bikaner": ["बीकानेर"], "Ajmer": ["अजमेर"],
        }
        found_at = []
        for city in CITY_STOPS:
            positions = [lowered.find(alias.lower()) for alias in [city, *aliases.get(city, [])] if lowered.find(alias.lower()) >= 0]
            if positions:
                found_at.append((city, min(positions)))
        found = [city for city, _ in sorted(found_at, key=lambda item: item[1])]
        return found or (existing or [])

    def _travelers(self, text: str, existing=1) -> int:
        normalized = text.translate(str.maketrans("०१२३४५६७८९", "0123456789")).lower()
        m = re.search(r"(\d+)\s*(?:people|persons?|pax|travell?ers?|adults?|of us|members?|log|लोग|व्यक्ति)", normalized)
        if m:
            return max(1, int(m.group(1)))
        if any(term in normalized for term in ("two", "couple", "2 people", "do log", "दो लोग", "दो व्यक्ति")):
            return 2
        if any(term in normalized for term in ("family", "parivar", "परिवार")):
            return 4
        if "solo" in text.lower() or "alone" in text.lower() or "myself" in text.lower():
            return 1
        return existing or 2

    @staticmethod
    def _days_from_dates(text: str) -> int | None:
        """Infer the trip length from a natural-language date range."""
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "जनवरी": 1, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
            "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10,
            "नवंबर": 11, "दिसंबर": 12,
        }
        normalized = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
        pattern = re.compile(r"(\d{1,2})\s+([A-Za-z]+|[\u0900-\u097f]+)\s+(\d{4})", re.I)
        found = []
        for day, month, year in pattern.findall(normalized):
            month_number = month_names.get(month.lower(), month_names.get(month))
            if month_number:
                try:
                    found.append(date(int(year), month_number, int(day)))
                except ValueError:
                    continue
        if len(found) < 2:
            return None
        duration = (found[1] - found[0]).days
        return duration if 1 <= duration <= 30 else None

    @staticmethod
    def _travel_date_from_text(text: str) -> str | None:
        """Extract one explicit travel date (used for dated transport searches)."""
        normalized = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
        months = {name: number for number, names in enumerate((
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ), 1) for name in names}
        months.update({name: number for number, name in enumerate((
            "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "सितम्बर", "अक्टूबर", "नवंबर", "दिसंबर"
        ), 1)})
        match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]+|[\u0900-\u097f]+)\s+(\d{4})\b", normalized, re.I)
        if not match:
            return None
        month = months.get(match.group(2).lower(), months.get(match.group(2)))
        if not month:
            return None
        try:
            return date(int(match.group(3)), month, int(match.group(1))).isoformat()
        except ValueError:
            return None

    def run(self, message: str, context: dict = None) -> dict:
        """Interpret with the model, answer locally when possible, and use grounded model fallback otherwise."""
        from prompting import interpret_request, synthesize_answer, synthesize_answer_groq, synthesize_answer_gemini

        context = dict(context or {})
        language = detect_response_language(message, context.get("response_language"))
        context["response_language"] = language
        interpretation = interpret_request(message, context)
        if interpretation:
            context["model_interpretation"] = interpretation
            # Model extraction informs the homemade planner, but explicit UI
            # context and locally parsed entities retain priority.
            for target, source in (
                ("days", "days"), ("travelers", "travelers"), ("budget", "budget_inr"),
                ("style", "travel_style"), ("dietary", "dietary"), ("origin_city", "origin_city"),
            ):
                if context.get(target) in (None, "", 0) and interpretation.get(source) not in (None, ""):
                    context[target] = interpretation[source]
            if not context.get("cities"):
                context["cities"] = interpretation.get("cities", [])
            if not context.get("interests") and interpretation.get("interests"):
                context["interests"] = interpretation["interests"]
            if "question_only" not in context:
                context["question_only"] = bool(interpretation.get("standalone_question"))
        result = self._run_deterministic(message, context)
        if interpretation:
            result.setdefault("trace", []).insert(0, {
                "tool": "llm_request_interpretation", "status": "complete",
                "summary": f"Intent: {interpretation['intent']}; extracted {len(interpretation.get('cities', []))} city/cities."
            })
        if result.pop("answer_is_final", False):
            result["prompting"] = {"strategy": "curated_travel_faq", "status": "complete"}
            return result
        has_local_answer = any(result.get(key) for key in ("plan", "stays", "restaurants", "hotels_by_city", "attractions", "places", "transport_options", "flight_options", "train_options"))
        # The model interprets every request, but local planner/tool results
        # remain the answer whenever they can answer the request.
        if has_local_answer:
            strategy = "local_planner" if result.get("plan") else "local_tool" if any(
                result.get(key) for key in ("transport_options", "flight_options", "train_options", "stays", "restaurants", "hotels_by_city", "attractions", "places")
            ) else "local_answer"
            result["prompting"] = {"strategy": strategy, "status": "complete"}
            return result
        # Retrieval score alone is not proof that a passage answers the question.
        # Keep passages as context for the GPT fallback instead of pasting loosely
        # related attractions (e.g. Bundi places in response to an airport query).
        if result.get("context", {}).get("needs_trip_details"):
            result["prompting"] = {"strategy": "planner_clarification", "status": "complete"}
            return result
        answer, status = synthesize_answer(message, result, language)
        primary_status = status
        answer_provider = "OpenAI"
        groq_status = "not_attempted"
        gemini_status = "not_attempted"
        if not answer:
            answer, groq_status = synthesize_answer_groq(message, result, language)
            if answer:
                answer_provider = "Groq"
                status = f"fallback_after_{primary_status}"
            else:
                answer, gemini_status = synthesize_answer_gemini(message, result, language)
                if answer:
                    answer_provider = "Gemini"
                    status = f"fallback_after_{primary_status}_and_groq_{groq_status}"
                else:
                    status = f"openai_{primary_status}__groq_{groq_status}__gemini_{gemini_status}"
        result["prompting"] = {
            "strategy": "all_provider_failure" if not answer else "grounded_json_synthesis" if answer_provider == "OpenAI" else f"{answer_provider.lower()}_fallback",
            "status": status,
            "primary_status": primary_status,
            "groq_status": groq_status,
            "fallback_status": gemini_status,
        }
        if answer:
            result["answer"] = answer
            result.setdefault("trace", []).append({
                "tool": f"{answer_provider.lower()}_fallback" if answer_provider != "OpenAI" else "llm_synthesis",
                "status": "complete", "summary": f"Generated from planner output and retrieved evidence using {answer_provider}."
            })
        else:
            if primary_status == "provider_http_429_credit_balance_exhausted":
                result["answer"] = (
                    "OpenAI has no available API credits, and the configured fallback providers could not complete the answer. Restore OpenAI billing/credits or retry when a fallback provider is available. I won’t guess about live travel details."
                    if language == "english" else
                    "OpenAI API में क्रेडिट उपलब्ध नहीं हैं और configured fallback सेवा भी उत्तर नहीं दे पाई। OpenAI क्रेडिट बहाल करें या fallback सेवा उपलब्ध होने पर फिर कोशिश करें। मैं लाइव यात्रा जानकारी का अनुमान नहीं लगाऊँगा।"
                )
            elif language == "hindi_devanagari":
                result["answer"] = (
                    "इस प्रश्न का भरोसेमंद उत्तर मेरे स्थानीय यात्रा स्रोतों में नहीं मिला। मैं मौजूदा समय-सारणी, कीमत या उपलब्धता की पुष्टि नहीं कर सकता—कृपया आधिकारिक स्रोत देखें। आप शहर या यात्रा का संदर्भ बताएँ, तो मैं उपलब्ध जानकारी खोज सकता हूँ।"
                    if result.get("context", {}).get("question_only") else self._hindi_fallback(result)
                )
            else:
                result["answer"] = (
                    "I couldn’t verify an answer from the local travel sources, and the AI answer service is temporarily unavailable. "
                    "I won’t guess about live schedules, airport proximity, prices, or availability. Please retry when the AI service is available."
                    if result.get("context", {}).get("question_only") else result.get("answer", "I couldn’t complete that request just now.")
                )
        return result

    @staticmethod
    def _hindi_fallback(result: dict) -> str:
        plan = result.get("plan")
        if plan:
            route = " → ".join(HINDI_CITY_NAMES.get(city, city) for city in plan.get("cities", []))
            days = len(plan.get("itinerary", []))
            budget = plan.get("budget", {})
            total = budget.get("group_total")
            people = result.get("context", {}).get("travelers", 2)
            amount = f"₹{total:,}" if isinstance(total, (int, float)) else ""
            return f"{route} के लिए {days} दिन की यात्रा योजना तैयार है। यह योजना {people} यात्रियों के लिए है। अनुमानित कुल बजट {amount} है। विस्तृत दिनवार कार्यक्रम नीचे दिया है। किराये, समय-सारणी और उपलब्धता यात्रा से पहले जाँच लें।"
        stays = result.get("stays", {})
        if stays:
            names = [hotel.get("name") for hotel in stays.get("hotels", []) if hotel.get("name")]
            city = stays.get("city", "इस शहर")
            return f"{city} में ठहरने के विकल्प: {', '.join(names) if names else 'उपयुक्त होटल और हॉस्टल की तुलना करें'}। कीमत और उपलब्धता सीधे जाँचें।"
        places = result.get("restaurants", [])
        if places:
            return "भोजन के विकल्प: " + "、".join(place.get("name", "स्थानीय विकल्प") for place in places[:5]) + "। खुलने का समय और भोजन संबंधी जानकारी सीधे जाँचें।"
        places = result.get("attractions", result.get("places", []))
        if places:
            return "देखने योग्य जगहें: " + "、".join(place.get("name", place.get("title", "स्थानीय जगह")) for place in places[:5]) + "। खुलने का समय और प्रवेश नियम आधिकारिक स्रोत से जाँचें।"
        return "मैंने आपकी यात्रा संबंधी जानकारी समझ ली है। कृपया शहर, यात्रा की अवधि और बजट बताइए, ताकि मैं आपके लिए सही सुझाव दे सकूँ।"

    # ── Mode-specific transport handlers ─────────────────────────────────────
    # A traveler asking for flights gets flights. Asking for trains gets trains.
    # Generated or estimated rows always travel with an explicit non-live label.

    @staticmethod
    def _flight_row_text(flight: dict) -> str:
        parts = []
        label = " ".join(x for x in (flight.get("airline"), flight.get("flight_number")) if x)
        if label:
            parts.append(f"**{label}**")
        if flight.get("departure_time") or flight.get("arrival_time"):
            parts.append(f"{flight.get('departure_time') or '?'} → {flight.get('arrival_time') or '?'}")
        if flight.get("duration_minutes"):
            parts.append(f"{int(flight['duration_minutes'])} min")
        if flight.get("stops"):
            parts.append(str(flight["stops"]))
        if flight.get("aircraft"):
            parts.append(str(flight["aircraft"]))
        if flight.get("cabin"):
            parts.append(str(flight["cabin"]))
        if flight.get("flight_status"):
            parts.append(f"status: {flight['flight_status']}")
        if flight.get("indicative_fare_inr"):
            parts.append(f"indicative ₹{int(flight['indicative_fare_inr']):,}")
        return " — ".join(parts) or "Option details unavailable"

    # Curated, source-linked answers for questions a provider API cannot answer
    # well (e.g. "which airport is nearest to Ranthambore"). These are general
    # published facts, not live availability, and are checked before provider calls.
    _CURATED_AIRPORT_ANSWERS = {
        "Ranthambore": (
            "Ranthambore के लिए जयपुर एयरपोर्ट (JAI) सबसे नज़दीकी प्रमुख हवाई अड्डा है—राजस्थान पर्यटन के अनुसार यह "
            "सवाई माधोपुर से लगभग 150–170 किमी दूर है। वहाँ से सड़क मार्ग से जाएँ या सवाई माधोपुर तक ट्रेन लेकर "
            "स्थानीय टैक्सी लें। रणथंभौर पार्क सवाई माधोपुर से लगभग 14 किमी दूर है। बुकिंग से पहले मौजूदा उड़ान, "
            "ट्रेन और सड़क यात्रा की जानकारी जाँचें। "
            "[राजस्थान पर्यटन](https://www.tourism.rajasthan.gov.in/sawaimadhopur.html)",
            "For Ranthambore, Jaipur Airport (JAI) is the nearest major airport. Rajasthan Tourism places it about "
            "150–170 km from Sawai Madhopur. From Jaipur, you can continue by road or take a train to Sawai Madhopur "
            "and arrange a local transfer; the park is about 14 km from town. Check current flight, train and road "
            "schedules before booking. "
            "[Rajasthan Tourism](https://www.tourism.rajasthan.gov.in/sawaimadhopur.html)",
        ),
        "Pushkar": (
            "पुष्कर के लिए जयपुर का सांगानेर/जयपुर एयरपोर्ट सबसे नज़दीकी हवाई अड्डा है—राजस्थान पर्यटन के अनुसार "
            "लगभग 146 किमी दूर। पुष्कर से 14 किमी दूर अजमेर रेलवे स्टेशन भी एक उपयोगी विकल्प है। उड़ानों और आगे के "
            "सड़क परिवहन की उपलब्धता यात्रा से पहले जाँचें। "
            "[राजस्थान पर्यटन](https://www.tourism.rajasthan.gov.in/pushkar.html)",
            "For Pushkar, Jaipur’s Sanganer Airport (JAI) is the nearest airport listed by Rajasthan Tourism, about "
            "146 km away. Ajmer railway station, around 14 km from Pushkar, is another useful arrival point. Check "
            "current flight routes and onward transport before booking. "
            "[Rajasthan Tourism](https://www.tourism.rajasthan.gov.in/pushkar.html)",
        ),
    }

    _CURATED_BUS_ANSWERS = {
        "Pushkar": (
            "हाँ—राजस्थान पर्यटन के अनुसार पुष्कर से जयपुर, जोधपुर और अजमेर के लिए नियमित बस सेवाएँ हैं। कौन-सा बस "
            "स्टॉप उपयोग करना है, उसकी पुष्टि यात्रा की तारीख के लिए स्थानीय रूप से करें; स्टॉप, ऑपरेटर और समय-सारणी "
            "बदल सकते हैं। [राजस्थान पर्यटन](https://www.tourism.rajasthan.gov.in/pushkar.html)",
            "Yes. Rajasthan Tourism says Pushkar has regular bus connections with Jaipur, Jodhpur and Ajmer. "
            "Confirm the exact stop, operator and departure time for your travel date, since these can change. "
            "[Rajasthan Tourism](https://www.tourism.rajasthan.gov.in/pushkar.html)",
        ),
    }

    @staticmethod
    def _curated_transport_answer(cities, lower, transport_mode, context) -> str | None:
        """Return a curated source-linked answer when one matches the question.

        An airport question is about arrival points for a city, not about flights
        on a route, so "nearest airport" wording is checked before mode narrowing.
        Each table entry is (hindi, english).
        """
        hindi = context.get("response_language") == "hindi_devanagari"
        airport_question = any(_term_present(lower, term) for term in
                               ("airport", "एयरपोर्ट", "हवाई अड्डा", "हवाईजहाज", "एयरलाइंस"))
        if airport_question and len(cities) < 2:
            index = 0 if hindi else 1
            for city in reversed(cities):
                if city in ("Ranthambore", "Sawai Madhopur"):
                    return RajasthanAgent._CURATED_AIRPORT_ANSWERS["Ranthambore"][index]
                if city == "Pushkar":
                    return RajasthanAgent._CURATED_AIRPORT_ANSWERS["Pushkar"][index]
        if transport_mode in ("buses", "any"):
            bus_question = any(_term_present(lower, term) for term in ("bus", "bus stop", "bus stand", "बस"))
            if bus_question and "Pushkar" in cities:
                return RajasthanAgent._CURATED_BUS_ANSWERS["Pushkar"][0 if hindi else 1]
        return None

    def _flight_answer(self, route_from, route_to, context, trace, evidence) -> dict:
        """Flights only — live Aviationstack first, then labelled generated rows."""
        from tools import get_flight_options
        flight_info = get_flight_options(str(route_from), str(route_to), context.get("travel_date"))
        trace.append({"tool": "flight_options", "status": "complete",
                      "summary": f"{flight_info.get('status')} · {flight_info.get('source', 'no source')}"})
        flights = flight_info.get("flights") or []
        title = f"## Flights from {flight_info.get('from', route_from)} to {flight_info.get('to', route_to)}"
        if context.get("travel_date"):
            title += f" on {context['travel_date']}"

        if flights:
            sections = [title, "\n".join(f"• {self._flight_row_text(f)}" for f in flights[:8])]
            if flight_info.get("flight_estimate"):
                estimate = flight_info["flight_estimate"][0]
                sections.append(
                    f"**Computed route estimate:** about {estimate['duration_minutes']} min non-stop "
                    f"({estimate.get('distance_km')} km great-circle), indicative ₹{estimate['indicative_fare_inr']:,}."
                )
            sections.append(f"*{flight_info.get('notice', '')}*")
            sections.append(f"_Source: {flight_info.get('source', 'unknown')}_")
            answer = "\n\n".join(x for x in sections if x)
        else:
            answer = "\n\n".join(x for x in [
                title,
                f"I couldn't get flight options for this route. {flight_info.get('notice', '')}",
            ] if x)

        result = {"answer": answer, "context": context, "trace": trace, "evidence": evidence,
                  "transport_options": flight_info, "flight_options": flight_info}
        if flight_info.get("live"):
            result["answer_is_final"] = True
        return result

    def _train_answer(self, route_from, route_to, context, trace, evidence) -> dict:
        """Trains only — live RailRadar timetable, else an honest 'not verified'."""
        from tools import get_train_options
        train_info = get_train_options(str(route_from), str(route_to), context.get("travel_date"))
        trace.append({"tool": "train_options", "status": "complete",
                      "summary": f"{train_info.get('status')} · {train_info.get('source', 'RailRadar')}"})
        trains = train_info.get("trains") or []
        title = f"## Trains from {train_info.get('from', route_from)} to {train_info.get('to', route_to)}"
        if context.get("travel_date"):
            title += f" on {context['travel_date']}"

        if trains:
            rows = []
            for t in trains:
                parts = []
                label = " ".join(x for x in (t.get("name"), t.get("number")) if x) or "Train"
                parts.append(f"**{label}**")
                if t.get("departure") or t.get("arrival"):
                    parts.append(f"{t.get('departure') or '?'} → {t.get('arrival') or '?'}")
                if t.get("duration_minutes"):
                    parts.append(f"{int(t['duration_minutes'])} min")
                if t.get("type"):
                    parts.append(str(t["type"]))
                if t.get("status"):
                    parts.append(f"live status: {t['status']}")
                if isinstance(t.get("delay_minutes"), (int, float)):
                    parts.append(f"delay {t['delay_minutes']} min")
                rows.append(f"• {' — '.join(parts)}")
            sections = [title, "\n".join(rows), f"*{train_info.get('notice', '')}*",
                        f"_Source: {train_info.get('source', 'RailRadar')}_"]
            answer = "\n\n".join(x for x in sections if x)
        else:
            answer = "\n\n".join(x for x in [title, train_info.get("notice", "")] if x)

        return {"answer": answer, "context": context, "trace": trace, "evidence": evidence,
                "transport_options": train_info, "train_options": train_info, "answer_is_final": True}

    def _single_mode_answer(self, mode, route_from, route_to, context, trace, evidence) -> dict:
        """Buses or cabs only — local planning estimates for the requested mode."""
        from tools import get_cab_rates
        route = get_cab_rates(str(route_from), str(route_to))
        trace.append({"tool": f"{mode}_estimate", "status": "complete",
                      "summary": route.get("source", "local transport knowledge")})
        wanted = "Bus" if mode == "buses" else "Cab"
        options = [o for o in (route.get("options") or []) if o.get("mode", "").lower().startswith(wanted.lower())]
        title = f"## {'Buses' if mode == 'buses' else 'Cabs'} from {route_from} to {route_to}"
        if not options:
            answer = "\n\n".join(x for x in [
                title,
                f"I don't have a {mode} estimate for this route. Approximate private cab rate is ₹15–18/km — "
                "confirm current fares and schedules with the operator.",
            ] if x)
        else:
            rows = [f"• **{o['mode']}**: {o['duration']}, {o['cost']} — Book at: {o['book_at']}" for o in options]
            answer = "\n\n".join(x for x in [title, "\n".join(rows), f"*{route.get('notice', '')}*"] if x)
        payload = {"from": route_from, "to": route_to, "mode": mode, "options": options,
                   "distance_km": route.get("distance_km"), "source": route.get("source")}
        return {"answer": answer, "context": context, "trace": trace, "evidence": evidence,
                "transport_options": payload, "answer_is_final": True}

    def _transport_answer(self, route_from, route_to, context, trace, evidence, lower, call) -> dict:
        """No single mode named — compare trains, buses and cabs together."""
        from tools import get_transport_options
        route_info = call("live_transport_options", get_transport_options, str(route_from), str(route_to),
                          context.get("travel_date"))
        if route_info and (route_info.get("options") or route_info.get("live_trains")):
            options_text = "\n".join(
                f"• **{o['mode']}**: {o['duration']}, {o['cost']} — Book at: {o['book_at']}"
                for o in route_info["options"]
            )
            train_text = "\n".join(
                f"• **{t.get('name') or 'Train'} ({t.get('number') or 'number unavailable'})** — "
                f"{t.get('departure') or 'time unavailable'} → {t.get('arrival') or 'time unavailable'}"
                + (f"; {t['duration_minutes']} min" if t.get("duration_minutes") else "")
                + (f"; live status: {t['status']}" if t.get("status") else "")
                + (f"; delay {t['delay_minutes']} min" if isinstance(t.get("delay_minutes"), (int, float)) else "")
                for t in route_info.get("live_trains", [])
            )
            sections = [f"## Getting from {route_from} to {route_to}"]
            if context.get("travel_date"):
                sections[0] += f" on {context['travel_date']}"
            if train_text:
                sections += ["**RailRadar train schedules**\n" + train_text]
            if options_text:
                sections += ["**Other transport estimates**\n" + options_text]
            if any(_term_present(lower, term) for term in ("flight", "fly", "airline")):
                sections += ["**Flights:** ask specifically for flights and I will return flights only."]
            if route_info.get("recommendation"):
                sections += [f"**Recommendation:** {route_info['recommendation']}"]
            sections += [f"*{route_info.get('notice', '')}*"]
            return {"answer": "\n\n".join(sections), "context": context, "trace": trace,
                    "evidence": evidence, "transport_options": route_info}
        # A transport request is still a transport request when providers return no
        # rows. Never reinterpret it as itinerary intake just because a city was named.
        if route_info:
            date_label = f" for {context['travel_date']}" if context.get("travel_date") else ""
            flight_note = (" Say *flights* and I will return flights only."
                           if any(_term_present(lower, term) for term in ("flight", "fly", "airline")) else "")
            answer = (f"I couldn't retrieve transport options for {route_from} to {route_to}{date_label}. "
                      f"{route_info.get('notice', 'Check the operator for current schedules and availability.')}{flight_note}")
            return {"answer": answer, "context": context, "trace": trace, "evidence": evidence,
                    "transport_options": route_info}
        date_label = f" for {context['travel_date']}" if context.get("travel_date") else ""
        answer = (f"I couldn't reach the transport service for {route_from} to {route_to}{date_label}. "
                  "Please retry later or confirm schedules directly with the rail/air operators.")
        return {"answer": answer, "context": context, "trace": trace, "evidence": evidence}

    def _run_deterministic(self, message: str, context: dict = None) -> dict:
        context = dict(context or {})
        trace = []
        text = message.strip()
        context["response_language"] = detect_response_language(text, context.get("response_language"))
        calls = 0

        def call(name, fn, *args, **kwargs):
            nonlocal calls
            if calls >= self.max_calls:
                return None
            calls += 1
            trace.append({"tool": name, "status": "running"})
            try:
                result = fn(*args, **kwargs)
                trace[-1]["status"] = "complete"
                trace[-1]["summary"] = f"OK — {type(result).__name__}"
                return result
            except Exception as exc:
                trace[-1]["status"] = "failed"
                # Third-party exceptions can contain user inputs, request URLs,
                # or credentials. Keep the visible trace useful but non-sensitive.
                trace[-1]["summary"] = f"Tool failed ({type(exc).__name__}); no details exposed."
                return None

        # ── Small talk ──────────────────────────────────────────────────────────
        # Guard before any intent routing: a greeting or "ok" must not inherit the
        # previous turn's task and re-answer it.
        if SMALL_TALK.match(text):
            route = [c for c in ([context.get("origin_city")] + list(context.get("cities") or [])) if c]
            route_text = " → ".join(str(c) for c in dict.fromkeys(route))
            if context.get("response_language") == "hindi_devanagari":
                answer = ("नमस्ते! मैं आपका राजस्थान यात्रा सहायक हूँ।"
                          + (f" आपकी यात्रा {route_text} के लिए है।" if route_text else "")
                          + " बताइए आप क्या जानना चाहते हैं — योजना, ठहरना, खाना, उड़ान या ट्रेन।")
            else:
                answer = ("Hello! I'm your Rajasthan travel assistant."
                          + (f" Your current trip is {route_text}." if route_text else "")
                          + " Tell me what you'd like — a plan, a stay, food, flights or trains.")
            # No retrieval runs for small talk, so evidence is empty by construction.
            # Marked final so it is never handed to model synthesis, which would
            # otherwise replace the greeting with a generic fallback message.
            return {"answer": answer, "context": context, "trace": trace, "evidence": [],
                    "answer_is_final": True}

        # ── Parse intent ───────────────────────────────────────────────────────
        cities = self._cities(text, context.get("cities"))
        origin_city = str(context.get("origin_city", "")).strip().casefold()
        if origin_city:
            cities = [city for city in cities if city.casefold() != origin_city]
        if not cities:
            cities = context.get("cities") or []
        context["cities"] = cities

        normalized = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
        supplied_travel_date = context.get("travel_date")
        if isinstance(supplied_travel_date, str):
            try:
                supplied_travel_date = date.fromisoformat(supplied_travel_date).isoformat()
            except ValueError:
                supplied_travel_date = None
        else:
            supplied_travel_date = None
        context["travel_date"] = supplied_travel_date or self._travel_date_from_text(text)
        day_match = re.search(r"(\d+)\s*(?:day|days|night|nights|din|रात|दिन)", normalized.lower())
        if day_match:
            context["days"] = int(day_match.group(1))
        else:
            hindi_number = re.search(r"\b(ek|do|teen|char|chaar|paanch|panch|chhe|cheh|saat|aath|nau|das)\s+(?:din|days?)\b|([एकदोतीनचारपाँचछहसातआठनौदस]+)\s*दिन", normalized.lower())
            number_words = {"ek": 1, "एक": 1, "do": 2, "दो": 2, "teen": 3, "तीन": 3, "char": 4, "chaar": 4, "चार": 4, "paanch": 5, "panch": 5, "पाँच": 5, "छह": 6, "saat": 7, "सात": 7, "aath": 8, "आठ": 8, "nau": 9, "नौ": 9, "das": 10, "दस": 10}
            if hindi_number:
                context["days"] = number_words.get((hindi_number.group(1) or hindi_number.group(2)).lower(), context.get("days"))
        if not day_match and not context.get("days"):
            inferred_days = self._days_from_dates(text)
            if inferred_days:
                context["days"] = inferred_days

        money = re.search(
            r"(?:₹|rs\.?|inr\s?|budget\s*(?:of|is|under)?\s*|बजट\s*(?:है|का)?\s*)(\d[\d,]*)"
            r"|(\d[\d,]*)\s*(?:budget|बजट|rupees?|रुपये?)",
            normalized.lower(),
        )
        if money:
            amount = money.group(1) or money.group(2)
            context["budget"] = int(amount.replace(",", ""))

        for style in ("budget", "balanced", "premium", "luxury", "sasta", "सस्ता", "कम बजट"):
            if style in text.lower():
                context["style"] = "premium" if style == "luxury" else "budget" if style in {"sasta", "सस्ता", "कम बजट"} else style

        dietary = next((x for x in ["non-vegetarian", "jain", "vegan", "vegetarian"] if x in text.lower()), context.get("dietary"))
        if dietary:
            context["dietary"] = dietary

        travelers = self._travelers(text, context.get("travelers", 2))
        context["travelers"] = travelers

        # ── Detect intents ─────────────────────────────────────────────────────
        lower = text.lower()
        interpreted_intent = (context.get("model_interpretation") or {}).get("intent")
        model_classified = interpreted_intent in {"question", "itinerary", "attractions", "lodging", "food", "transport", "places"}
        hotel_intent = interpreted_intent == "lodging" if model_classified else any(x in lower for x in ["hotel", "stay", "accommodation", "hostel", "resort", "palace", "where to stay", "place to stay", "रहने", "रुकने", "ठहरने"])
        restaurant_intent = interpreted_intent == "food" if model_classified else any(x in lower for x in ["restaurant", "eat", "food", "cafe", "dining", "thali", "where to eat", "best food", "khana", "khaana", "खाना", "भोजन", "रेस्टोरेंट"])
        attraction_intent = interpreted_intent == "attractions" if model_classified else any(x in lower for x in ["attraction", "visit", "see", "fort", "temple", "palace", "things to do", "sightseeing", "places to visit", "ghoom", "ghum", "देखना", "घूम", "जगह", "दर्शनीय"])
        explicit_transport_terms = any(_term_present(lower, term) for term in TRANSPORT_INTENT_TERMS)
        cab_intent = interpreted_intent == "transport" or explicit_transport_terms
        transport_mode = detect_transport_mode(text)
        # Duration words alone (e.g. "a 2 day stay") describe the lodging
        # request; they must not turn it into an itinerary intake flow.
        explicit_plan_request = any(x in lower for x in ("plan", "itinerary", "make me", "create", "build", "prepare", "बनाओ", "योजना बनाओ"))
        legacy_plan_language = any(x in lower for x in ["plan", "itinerary", "trip", "route", "schedule", "guide me", "organize", "arrange", "cheaper", "budget", "add", "modify", "change", "update", "yatra", "safar", "ghoomna", "ghumna", "banao", "batao", "यात्रा", "सफर", "घूमना", "योजना", "बनाओ"])
        plan_intent = (
            explicit_plan_request or (interpreted_intent == "itinerary" and not explicit_transport_terms)
            if model_classified else explicit_plan_request or (legacy_plan_language and not explicit_transport_terms)
        )
        review_intent = any(x in lower for x in ["review", "rating", "recommend", "best", "top", "which hotel", "which restaurant"])
        places_intent = interpreted_intent == "places" if model_classified else any(x in lower for x in ["nearby", "near me", "google places", "search places", "places around", "find places"])

        # ── RAG: retrieve background knowledge ─────────────────────────────────
        interpreted = context.get("model_interpretation") or {}
        query = expand_multilingual_query(" ".join([text, *interpreted.get("interests", []), *interpreted.get("constraints", [])]), cities)
        evidence = call("retrieve_knowledge", retrieve_knowledge, query,
                        destination=cities[0] if len(cities) == 1 else None, top_k=5) or []

        # ── Question-only handling ──────────────────────────────────────────────
        # Curated, source-linked answers are checked first for both question-only
        # turns and transport turns, so a "which airport serves X" question keeps
        # its vetted answer instead of being rerouted as a flight search.
        if not plan_intent:
            curated = self._curated_transport_answer(cities, lower, transport_mode, context)
            if curated:
                return {"answer": curated, "context": context, "trace": trace, "evidence": evidence,
                        "answer_is_final": True}

        if context.get("question_only") and not plan_intent and not cab_intent:
            aviation_airports = None
            if cities and any(_term_present(lower, term)
                              for term in ("airport", "एयरपोर्ट", "हवाई अड्डा")):
                aviation_airports = call("aviationstack_airport_lookup", search_aviation_airports, cities[-1])
            if aviation_airports and aviation_airports.get("status") == "ok":
                rows = "\n".join(
                    f"• **{a.get('airport_name') or 'Airport'}** ({a.get('iata_code') or 'code unavailable'}) — {a.get('city') or ''}, {a.get('country_name') or ''}"
                    for a in aviation_airports.get("airports", [])[:5])
                answer = (f"### Airports serving {cities[-1]}\n\n{rows}\n\n"
                          "*Airport listings from Aviationstack. Confirm current routes and schedules with the airline.*")
            elif evidence:
                answer = ("मेरे स्थानीय स्रोतों में इस सवाल का भरोसेमंद जवाब नहीं मिला। "
                          "आप शहर और अपनी ज़रूरत—जैसे बस, ट्रेन, ठहरने या घूमने की जानकारी—बताएँ, "
                          "तो मैं सही जानकारी खोजने की कोशिश करूँगा।"
                          if context.get("response_language") == "hindi_devanagari" else
                          "I couldn’t find a reliable local source for that exact question. "
                          "Tell me the city and whether you mean transport, places to visit, food or a stay, "
                          "and I’ll narrow it down.")
            else:
                answer = ("मैं यात्रा, घूमने की जगहों, ठहरने और खाने के बारे में मदद कर सकता हूँ। आप क्या जानना चाहेंगे?"
                          if context.get("response_language") == "hindi_devanagari" else
                          "I can help with travel, places to visit, stays and food. What would you like to know?")
            # Only a vetted answer is final. A "no local source found" reply must
            # stay open so the grounded model synthesis pass can try to help.
            final = bool(aviation_airports and aviation_airports.get("status") == "ok")
            return {"answer": answer, "context": context, "trace": trace, "evidence": evidence,
                    **({"aviation_airports": aviation_airports} if aviation_airports else {}),
                    "answer_is_final": final}

        # ── No city detected for planning ──────────────────────────────────────
        if not cities and plan_intent:
            context["needs_trip_details"] = True
            return {
                "answer": (
                    "I'd love to plan your Rajasthan trip! To get started, I need a few details:\n\n"
                    "**Which cities would you like to visit?** (Jaipur, Udaipur, Jodhpur, Jaisalmer, Ranthambore, Pushkar, Mount Abu, etc.)\n\n"
                    "**How many days do you have?**\n\n"
                    "**How many people are travelling?**\n\n"
                    "**What's your budget?** (per person or total group)\n\n"
                    "**Any preferences?** (wildlife safari, heritage forts, food travel, romantic trip, family-friendly, adventure, etc.)\n\n"
                    "Once I have these, I'll create a day-by-day plan with clearly labelled local reference data, estimates, and details that need confirmation."
                ),
                "context": context, "trace": trace, "evidence": evidence
            }

        # ── Optional place lookup ───────────────────────────────────────────────
        if places_intent:
            query_text = re.sub(r"\b(nearby|near me|google places|search places|places around|find places)\b", "", text, flags=re.I).strip() or "travel attractions"
            places_data = call("search_places", search_places, query_text, cities[0] if cities else None) or {}
            source = places_data.get("source", "local fallback")
            rows = places_data.get("results", [])
            if source == "Google Places" and rows:
                lines = [f"- **{p.get('displayName', {}).get('text', 'Place')}** — {p.get('formattedAddress', 'Address unavailable')}" for p in rows[:5]]
            else:
                lines = [f"- **{p.get('title', p.get('name', 'Local reference'))}** — {p.get('text', '')}" for p in rows[:5]]
            answer = f"Place search source: **{source}**.\n\n" + ("\n".join(lines) if lines else "No matching place records were found.")
            if places_data.get("notice"):
                answer += f"\n\n_{places_data['notice']}_"
            return {"answer": answer, "context": context, "trace": trace, "evidence": evidence,
                    "places": rows, "places_source": source}

        # ── Transport query (mode-specific) ────────────────────────────────────
        # An explicit "from A to B" wins over inferred cities so a non-Rajasthan
        # origin such as Pune is preserved.
        parsed_from, parsed_to = parse_route_intent(text)
        route_from = parsed_from or (cities[0] if len(cities) >= 2 else (
            context.get("origin_city") or (context.get("model_interpretation") or {}).get("origin_city")))
        route_to = parsed_to or (cities[1] if len(cities) >= 2 else (cities[-1] if cities else None))
        # A mode-specific request is answered with that mode.
        if cab_intent and not plan_intent and route_from and route_to and str(route_from).casefold() != str(route_to).casefold():
            if transport_mode == "flights":
                return self._flight_answer(route_from, route_to, context, trace, evidence)
            if transport_mode == "trains":
                return self._train_answer(route_from, route_to, context, trace, evidence)
            if transport_mode == "buses":
                return self._single_mode_answer("buses", route_from, route_to, context, trace, evidence)
            if transport_mode == "cabs":
                return self._single_mode_answer("cabs", route_from, route_to, context, trace, evidence)
            return self._transport_answer(route_from, route_to, context, trace, evidence, lower, call)

        # ── Hotel-only query ───────────────────────────────────────────────────
        if hotel_intent and cities and not plan_intent:
            tier_map = {"budget": "budget", "balanced": "mid-range", "premium": "luxury"}
            tier = tier_map.get(context.get("style", "balanced"), "all")
            hotels_data = call("search_hotels", search_hotels, cities[0], tier=tier, top_k=3)
            hotels = hotels_data.get("hotels", []) if hotels_data else []

            if hotels:
                hotel_lines = []
                for h in hotels:
                    hotel_lines.append(
                        f"### 🏨 {h['name']} ({h.get('tier', '').title()})\n"
                        f"**Indicative reference price:** {h['price_per_night']}/night\n"
                        
                        
                        
                        f"**Amenities:** {', '.join(h.get('amenities', [])[:5])}\n"
                        f"**Dietary options:** Confirm directly with the property.\n"
                        
                        + (f"**Travel times:** {', '.join(f'{k}: {v}' for k,v in list(h.get('travel_times', {}).items())[:3])}" if h.get('travel_times') else "")
                    )
                answer = (
                    f"## Hotel reference ideas for {cities[0]} ({tier.title()} tier)\n\n"
                    + "\n\n---\n\n".join(hotel_lines)
                    + f"\n\n*{hotels_data.get('guidance', '')}*"
                    + f"\n\n⚠️ *{hotels_data.get('notice', '')}*"
                )
            else:
                answer = (
                    f"I don't have specific hotel listings for {cities[0]} yet, but here's my guidance:\n\n"
                    f"{hotels_data.get('guidance', 'Choose a well-reviewed stay near your main attractions.')}\n\n"
                    f"Search on Booking.com or MakeMyTrip with filters for your preferred area."
                )
            return {"answer": answer, "context": context, "trace": trace, "evidence": evidence,
                    "stays": {"source": hotels_data.get("source", "planning guidance"), "city": cities[0],
                              "guidance": hotels_data.get("guidance", ""), "hotels": hotels,
                              "notice": hotels_data.get("notice", "")}}

        # ── Restaurant-only query ──────────────────────────────────────────────
        if restaurant_intent and cities and not plan_intent:
            rest_data = call("search_restaurants", search_restaurants, cities[0], dietary=dietary, top_k=4)
            restaurants = rest_data.get("results", []) if rest_data else []

            if restaurants:
                rest_lines = []
                for r in restaurants:
                    famous = ", ".join(r.get("famous_for", [])[:4])
                    rest_lines.append(
                        f"### 🍽️ {r['name']}\n"
                        f"**Cuisine:** {r.get('cuisine')} · **Avg meal:** ₹{r.get('avg_meal_inr')}/person\n"
                        
                        
                        
                        "**Dietary:** Confirm directly with the venue.\n"
                        
                        
                    )
                answer = (
                    f"## Restaurant reference ideas for {cities[0]}\n\n"
                    + "\n\n---\n\n".join(rest_lines)
                    + f"\n\n⚠️ *{rest_data.get('notice', '')}*"
                )
            else:
                answer = f"No specific restaurant data for {cities[0]}. Look for a Rajasthani thali near your stay — most areas have excellent local options. Confirm dietary requirements directly."
            return {"answer": answer, "context": context, "trace": trace, "evidence": evidence, "restaurants": restaurants}

        # ── Attraction query ───────────────────────────────────────────────────
        if attraction_intent and cities and not plan_intent:
            attr_data = call("search_attractions", search_attractions, cities[0], top_k=5)
            attractions = attr_data.get("attractions", []) if attr_data else []

            if attractions:
                attr_lines = []
                for a in attractions:
                    highlights = ", ".join(a.get("highlights", [])[:3])
                    tips = a.get("tips", [])
                    tips_text = "\n".join(f"  💡 {t}" for t in tips[:3]) if tips else ""
                    attr_lines.append(
                        f"### 🏛️ {a['name']} ({a.get('category', '').title()})\n"
                        f"**Entry:** {a['entry_fee_indian']} (Indians) / {a['entry_fee_foreigner']} (foreigners)\n"
                        f"**Timings:** {a.get('timings')} · **Best time:** {a.get('best_time')}\n"
                        f"**Indicative duration:** {a.get('duration_hours')} hours\n"
                        f"**Highlights:** {highlights}\n"
                        + (tips_text + "\n" if tips_text else "")
                    )
                answer = (
                    f"## Attraction ideas in {cities[0]}\n\n"
                    + "\n\n---\n\n".join(attr_lines)
                    + f"\n\n⚠️ *{attr_data.get('notice', '')}*"
                )
            else:
                answer = f"I have general attraction knowledge for {cities[0]}. What specifically are you looking for — forts, temples, wildlife, or experiences?"
            return {"answer": answer, "context": context, "trace": trace, "evidence": evidence, "attractions": attractions}

        # ── Full itinerary planning ────────────────────────────────────────────
        # A city mention alone used to trigger an itinerary, even for factual
        # questions. When GPT has classified the request, only the itinerary
        # intent (or an old-parser fallback) may enter the planner.
        explicit_non_planning_intent = hotel_intent or restaurant_intent or attraction_intent or places_intent or cab_intent
        should_plan = plan_intent or (bool(cities) and interpreted_intent in {None, "other"} and not explicit_non_planning_intent)
        if should_plan:
            if any(word in lower for word in ("cheaper", "budget", "sasta", "सस्ता", "कम बजट")):
                context["style"] = "budget"

            if not context.get("days"):
                context["needs_trip_details"] = True
                return {
                    "answer": (
                        f"Great choice — **{' → '.join(cities)}** is a wonderful route!\n\n"
                        f"How many days do you have? And how many people are travelling?\n\n"
                        f"A budget or travel style will help me tune the hotels, restaurants and activities to exactly what you need."
                    ),
                    "context": context, "trace": trace, "evidence": evidence
                }

            style = context.get("style", "balanced")

            # Full planning
            result = call("plan_itinerary", plan_itinerary,
                          cities, context["days"], travelers=travelers,
                          style=style, total_budget=context.get("budget"),
                          interests=" ".join([text, *(context.get("interests") or interpreted.get("interests", [])), *interpreted.get("constraints", [])])) or {}

            validation = call("validate_itinerary", validate_itinerary, result.get("itinerary", [])) or {}
            route_info = call("compare_routes", compare_routes, result.get("cities", cities)) or {}

            # Get hotels for each unique city
            hotels_by_city = {}
            hostels_by_city = {}
            place_sources = []
            for city in result.get("cities", cities):
                tier_map = {"budget": "budget", "balanced": "mid-range", "premium": "luxury"}
                tier = tier_map.get(style, "all")
                h_data = call(f"hotels_{city}", search_hotels, city, tier=tier, top_k=2)
                if h_data and h_data.get("source"):
                    place_sources.append(h_data["source"])
                if h_data and h_data.get("hotels"):
                    hotels_by_city[city] = h_data["hotels"]
                if h_data and h_data.get("hostels"):
                    hostels_by_city[city] = h_data["hostels"]

            # Keep food and activity suggestions attached to each itinerary city.
            restaurants_by_city, attractions_by_city = {}, {}
            for city in result.get("cities", cities):
                rest_data = call(f"restaurants_{city}", search_restaurants, city, dietary=dietary, top_k=3) or {}
                attr_data = call(f"attractions_{city}", search_attractions, city, top_k=3) or {}
                place_sources.extend([data.get("source", "") for data in (rest_data, attr_data)])
                if rest_data.get("results"):
                    restaurants_by_city[city] = rest_data["results"]
                if attr_data.get("attractions"):
                    attractions_by_city[city] = attr_data["attractions"]
            restaurants = restaurants_by_city.get((result.get("cities") or cities)[0], [])
            attractions = attractions_by_city.get((result.get("cities") or cities)[0], [])

            # Build rich answer
            budget = result.get("budget", {})
            planned_cities = result.get("cities", cities)
            route_str = " → ".join(planned_cities)
            hindi = context.get("response_language") == "hindi_devanagari"
            style_label = {"budget": "बजट", "balanced": "संतुलित", "premium": "प्रीमियम"}.get(style, style.title())
            # The itinerary body stays as curated English reference text, but the
            # summary must match the traveler's language or a Hindi trip opens in English.
            route_label = " → ".join(HINDI_CITY_NAMES.get(city, city) for city in planned_cities) if hindi else route_str
            if hindi:
                intro_parts = [
                    f"मैंने **{route_label}** के लिए **{style_label}** राजस्थान यात्रा योजना तैयार की है।",
                    "दिनवार कार्यक्रम और बजट स्थानीय योजना मान्यताओं से गणना किए गए हैं। होटल और स्थान "
                    "रिकॉर्ड केवल स illustrative संदर्भ हैं, लाइव खोज परिणाम नहीं।",
                ]
                if route_info.get("transfers"):
                    intro_parts.append(f"आपके मार्ग में **{route_info['transfers']}** अंतर-शहर स्थानांतरण हैं।")
                if budget.get("per_person"):
                    intro_parts.append(
                        f"\n**अनुमानित लागत: ₹{budget['per_person']:,} प्रति व्यक्ति** "
                        f"({travelers} लोगों के लिए ₹{budget['group_total']:,})। इसमें ठहरना, खाना, दर्शनीय स्थल, "
                        "परिवहन और आकस्मिक खर्च शामिल हैं।")
                if budget.get("over_budget"):
                    intro_parts.append("\n⚠️ **बजट चेतावनी:** आपके बताए बजट में यह यात्रा सीमित है। कम शहर, "
                                      "बजट होटल, कैब के बजाय ट्रेन और कम भुगतान अनुभव चुनें।")
                if validation.get("warnings"):
                    intro_parts.append("\n⚠️ **व्यावहारिक सुझाव:** " + " | ".join(validation["warnings"]))
                intro_parts.append(
                    "\nपूरा दिनवार कार्यक्रम, होटल सुझाव, रेस्तराँ और बजट विवरण नीचे दिया है। "
                    "आप *'सस्ता बनाएं'*, *'शाकाहारी विकल्प जोड़ें'* या *'जयपुर से उदयपुर कैसे पहुँचें'* कह सकते हैं।")
            else:
                intro_parts = [
                    f"I've built a comprehensive **{style.title()} Rajasthan plan** for **{route_str}**.",
                    f"The itinerary and budget are computed from local planning assumptions. Optional property and venue records are illustrative references, not live search results.",
                ]

                if route_info.get("transfers"):
                    intro_parts.append(f"Your route has **{route_info['transfers']} intercity transfer(s)**.")
                if budget.get("per_person"):
                    intro_parts.append(
                        f"\n**Estimated cost: ₹{budget['per_person']:,} per person** (₹{budget['group_total']:,} for {travelers} people). "
                        f"This covers accommodation, food, attractions, transport and a contingency buffer."
                    )
                if budget.get("over_budget"):
                    intro_parts.append(f"\n⚠️ **Budget alert:** Your stated budget looks tight for these cities and nights. Best levers: fewer cities, budget stays, trains over cabs, fewer paid experiences.")
                if validation.get("warnings"):
                    intro_parts.append("\n⚠️ **Practical notes:** " + " | ".join(validation["warnings"]))

                intro_parts.append(
                    "\nThe full day-by-day plan, hotel recommendations, restaurants and budget breakdown are shown below. "
                    "You can say things like *'make it cheaper'*, *'add a vegetarian option'*, *'what's the best hotel in Jodhpur'*, or *'how do I get from Jaipur to Udaipur'*."
                )

            answer = " ".join(intro_parts)

            context.update({"cities": result.get("cities", cities), "last_plan": result})

            return {
                "answer": answer,
                "context": context,
                "trace": trace,
                "evidence": evidence,
                "plan": result,
                "restaurants": restaurants,
                "restaurants_by_city": restaurants_by_city,
                "hotels_by_city": hotels_by_city,
                "hostels_by_city": hostels_by_city,
                "attractions_by_city": attractions_by_city,
                "places_live": any("Google Places" in source for source in place_sources),
                "attractions": attractions,
                "stays": {
                    "guidance": result.get("stay_guidance"),
                    "notice": "Prices change seasonally — verify and book directly."
                }
            }

        # ── Fallback ───────────────────────────────────────────────────────────
        return {
            "answer": (
                "I'm your Rajasthan travel expert — I can help with:\n\n"
                "• **Full trip planning** — day-by-day itinerary with local guidance, reference records and planning estimates\n"
                "• **Stay guidance** — compare locations, amenities and reference data; verify current prices and availability\n"
                "• **Restaurant picks** — local favourites with what to order and current hours\n"
                "• **Things to do** — attractions with indicative guidance; verify current fees, access and hours\n"
                "• **Transport costs** — cab, train and bus options between cities\n"
                "• **Budget planning** — detailed cost breakdown across tiers\n\n"
                "Tell me a city or destination and what you'd like, and I'll get to work!"
            ),
            "context": context, "trace": trace, "evidence": evidence
        }
