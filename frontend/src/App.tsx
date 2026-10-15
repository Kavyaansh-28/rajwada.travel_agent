import { useState, useRef, useEffect, useCallback, type ReactNode } from "react"

// ─── Types ─────────────────────────────────────────────────────────────────

type Lang = "en" | "hi"
type Theme = "light" | "dark"

type IconName =
  | "arch" | "arrow" | "attach" | "bookmark" | "calendar" | "check"
  | "chevron" | "clock" | "close" | "compass" | "download" | "food"
  | "location" | "menu" | "mic" | "moon" | "more" | "plus" | "route"
  | "settings" | "share" | "shield" | "spark" | "sun" | "sunrise"
  | "users" | "star" | "cab" | "hotel" | "export" | "translate" | "edit"
  | "trash" | "collapse" | "loading" | "send" | "wallet" | "tag" | "plane"

type ItineraryDay = {
  day: number
  city: string
  morning: string
  afternoon: string
  evening: string
  transport: string
  meal: string
  note: string
}

type Budget = {
  per_person: number
  group_total: number
  breakdown_per_person: Record<string, number>
  over_budget: boolean
  note: string
}

type FlightRow = {
  airline?: string
  flight_number?: string
  departure_time?: string
  arrival_time?: string
  duration_minutes?: number
  stops?: string
  aircraft?: string
  cabin?: string
  indicative_fare_inr?: number
  flight_status?: string
}

type TrainRow = {
  number?: string
  name?: string
  type?: string
  departure?: string
  arrival?: string
  duration_minutes?: number
  status?: string
  delay_minutes?: number
}

type TransportOptions = {
  from?: string
  to?: string
  mode?: string
  status?: string
  source?: string | null
  live?: boolean
  notice?: string
  distance_km?: number
  origin_iata?: string | null
  destination_iata?: string | null
  flights?: FlightRow[]
  trains?: TrainRow[]
  live_trains?: TrainRow[]
  options?: { mode?: string; duration?: string; cost?: string; book_at?: string }[]
}

type AgentResult = {
  answer: string
  context: Record<string, unknown>
  plan?: {
    itinerary: ItineraryDay[]
    budget: Budget
    cities: string[]
    stay_guidance: string
  }
  restaurants?: { name?: string; title?: string; text?: string }[]
  restaurants_by_city?: Record<string, { name?: string; title?: string; text?: string; address?: string }[]>
  hotels_by_city?: Record<string, { name?: string; tier?: string; text?: string; address?: string; price_per_night?: string }[]>
  hostels_by_city?: Record<string, { name?: string; tier?: string; text?: string; address?: string }[]>
  attractions?: { name?: string; address?: string; status?: string }[]
  attractions_by_city?: Record<string, { name?: string; address?: string; status?: string }[]>
  places_live?: boolean
  evidence?: { title: string; text: string }[]
  trace?: { tool: string; status: string }[]
  prompting?: { strategy: string; status: string }
  stays?: { style?: string; guidance?: string; notice?: string }
  transport_options?: TransportOptions
  flight_options?: TransportOptions
  train_options?: TransportOptions
}

async function readJsonResponse<T>(response: Response): Promise<T> {
  const body = await response.text()
  if (!body.trim()) {
    throw new Error(`The server returned an empty response (HTTP ${response.status}). Check that the planner API is running and try again.`)
  }
  try {
    return JSON.parse(body) as T
  } catch {
    throw new Error(`The server returned an unreadable response (HTTP ${response.status}). Check the planner API and try again.`)
  }
}

type ChatMessage = {
  id: string
  role: "user" | "assistant"
  text: string
  result?: AgentResult
  timestamp: Date
}

type Journey = {
  id: string
  title: string
  route?: string
  messages: ChatMessage[]
  lastActive: Date
  stage?: "origin" | "destination" | "details" | "planning"
  origin?: string
  destination?: string
  routeCities?: string[]
  initialPrompt?: string
  details?: string
  context?: Record<string, unknown>
}

const CITY_ALIASES: Record<string, string[]> = {
  Jaipur: ["Jaipur", "जयपुर"], Udaipur: ["Udaipur", "उदयपुर"], Jodhpur: ["Jodhpur", "जोधपुर"],
  Jaisalmer: ["Jaisalmer", "जैसलमेर"], Pushkar: ["Pushkar", "पुष्कर"],
  Ranthambore: ["Ranthambore", "रणथंभौर", "रणथम्बोर"], "Sawai Madhopur": ["Sawai Madhopur", "सवाई माधोपुर"],
  "Mount Abu": ["Mount Abu", "माउंट आबू"], Ajmer: ["Ajmer", "अजमेर"], Bikaner: ["Bikaner", "बीकानेर"],
}

function extractRouteCities(text: string): string[] {
  const hits: { city: string; index: number }[] = []
  for (const [city, aliases] of Object.entries(CITY_ALIASES)) {
    const positions = aliases.flatMap((alias) => {
      const escaped = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      const match = new RegExp(escaped, "iu").exec(text)
      return match ? [match.index] : []
    })
    if (positions.length) hits.push({ city, index: Math.min(...positions) })
  }
  return hits.sort((a, b) => a.index - b.index).map((hit) => hit.city)
}

// A bare short message is treated as a city name only when it really looks like
// one. Conversational filler ("ok got it!", "thanks!") must never become an
// origin, or the journey title and trip map fill with junk.
const NON_CITY_WORDS = /^(?:ok|okay|okey|k|kk|yes|yeah|yep|yup|yup|no|nope|nah|sure|right|correct|exactly|fine|great|nice|cool|good|awesome|thanks|thank|thx|cheers|bye|goodbye|hi|hello|hey|yo|namaste|done|perfect|wonderful|got|it|understood|noted|please|help|next|more|stop|wait|and|then|also|but|so|the|a|an|is|are|am|do|does|did|can|could|would|will|what|which|where|when|why|how|much|many|book|booked|booking|hotel|hotels|hostel|stay|stays|flight|flights|train|trains|bus|buses|cab|cabs|taxi|food|restaurant|restaurants|budget|itinerary|plan|plans|trip|ticket|tickets|cheaper|cheap|best|near|nearby|में|का|की|के|को|है|हैं|और|लिए|ठीक|शुक्रिया|धन्यवाद)$/iu

function looksLikeCityName(text: string): boolean {
  const trimmed = text.trim()
  if (!trimmed || trimmed.length > 40) return false
  // Multi-word input is only trusted when it arrived through an explicit
  // "I'm from X" phrasing, which extractOrigin handles separately.
  if (/\s/.test(trimmed)) return false
  if (!/^[\p{L}][\p{L}.'’-]*$/u.test(trimmed)) return false
  if (NON_CITY_WORDS.test(trimmed)) return false
  if (hasTravelTask(trimmed)) return false
  return true
}

function extractOrigin(text: string): string | null {
  if (isQuestionLike(text) || isAcknowledgement(text) || isGreeting(text)) return null
  const english = text.match(/\b(?:i am coming from|i'm coming from|coming from|travelling from|traveling from|starting from|departing from|i am from|i'm from|im from|based in|living in|live in)\s+([\p{L}][\p{L}\s-]{0,35}?)(?=,|\s+(?:and|i|to|on|for|need|want|plan)\b|[.!?]|$)/iu)
  const romanHindi = text.match(/\b(?:mai|main|hum)\s+([\p{L}][\p{L}\s-]{0,28}?)\s+se\b/iu)
  const devanagari = text.match(/([\p{Script=Devanagari}][\p{Script=Devanagari}\s]{0,24}?)\s+से\b/u)
  const candidate = (english?.[1] ?? romanHindi?.[1] ?? devanagari?.[1])?.trim()
  if (candidate) {
    const cleaned = candidate.replace(/^(?:i am|i'm|main|mai|hum|मैं|हम)\s+/iu, "").replace(/[.!?,]+$/u, "").trim()
    if (cleaned && cleaned.split(/\s+/).length <= 4 && !NON_CITY_WORDS.test(cleaned.toLowerCase())) return cleaned
  }
  return looksLikeCityName(text) ? text.trim() : null
}

function isGreeting(text: string): boolean {
  return /^(?:(?:hi+|hello+|hey+)(?:\s+there)?|good\s+(?:morning|afternoon|evening)|namaste|नमस्ते|नमस्कार|हाय|हेलो)[!.\s,]*$/iu.test(text.trim())
}

function isQuestionLike(text: string): boolean {
  return text.includes("?") || /^(?:(?:which|what|where|when|why|how|can|could|do|does|is|are|will|would|nearest|closest|distance|tell me|explain|show me)\b|(?:कौन|क्या|कहाँ|कहां|कैसे|कब|बताइए|बताओ)(?:\s|$))/iu.test(text.trim())
}

function isAcknowledgement(text: string): boolean {
  return /^(?:ok(?:ay)?|yes|no|sure|right|great|cool|thanks?|thank you|got it|alright|ठीक है|हाँ|हां|नहीं|धन्यवाद)[!.\s,]*$/iu.test(text.trim())
}

function hasTravelTask(text: string): boolean {
  return /\b(?:hotel|hostel|stay|accommodation|food|eat|restaurant|airport|bus|train|flight|visit|see|things to do|attraction|recommend|suggest|plan|trip|itinerary|budget|cheap|cheaper|cost|route|weather|activity|activities|book|ghoom|ghumna|jana|jaana|batao|banao)\b|(?:होटल|रहने|खाना|भोजन|घूम|यात्रा|योजना|बजट|किराया|हवाई अड्डा|बस|ट्रेन)/iu.test(text)
}

function inferTripDays(text: string): number | undefined {
  const normalized = text.replace(/[०-९]/g, (digit) => String("०१२३४५६७८९".indexOf(digit)))
  const explicit = normalized.match(/\b(\d{1,2})\s*(?:days?|nights?|din|दिन|रात|रातें)\b/iu)
  if (explicit) return Number(explicit[1])
  const words: Record<string, number> = { ek: 1, एक: 1, do: 2, दो: 2, teen: 3, तीन: 3, char: 4, chaar: 4, चार: 4, paanch: 5, पाँच: 5, पांच: 5, chhe: 6, छह: 6, saat: 7, सात: 7, aath: 8, आठ: 8, nau: 9, नौ: 9, das: 10, दस: 10 }
  const word = normalized.match(/\b(ek|do|teen|char|chaar|paanch|chhe|saat|aath|nau|das|एक|दो|तीन|चार|पाँच|पांच|छह|सात|आठ|नौ|दस)\s+(?:din|दिन)\b/iu)
  if (word) return words[word[1].toLowerCase()]
  const dates = [...normalized.matchAll(/(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{4})/giu)]
  if (dates.length < 2) return undefined
  const months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
  const hindiMonths = ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
  const dateValue = (match: RegExpMatchArray) => {
    const month = match[2].toLowerCase()
    const index = months.findIndex((name) => name.startsWith(month))
    const monthIndex = index >= 0 ? index : hindiMonths.indexOf(match[2])
    return Date.UTC(Number(match[3]), monthIndex, Number(match[1]))
  }
  const first = dateValue(dates[0])
  const last = dateValue(dates[1])
  const days = Math.round((last - first) / 86400000)
  return days > 0 && days <= 30 ? days : undefined
}

function inferTravelDates(text: string): { start_date: string; end_date: string } | undefined {
  const normalized = text.replace(/[०-९]/g, (digit) => String("०१२३४५६७८९".indexOf(digit)))
  const dates = [...normalized.matchAll(/(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{4})/giu)]
  if (dates.length < 2) return undefined
  const months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
  const hindiMonths = ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
  const format = (match: RegExpMatchArray) => {
    const month = match[2].toLowerCase()
    const index = months.findIndex((name) => name.startsWith(month))
    const monthIndex = index >= 0 ? index : hindiMonths.indexOf(match[2])
    return new Date(Date.UTC(Number(match[3]), monthIndex, Number(match[1]))).toISOString().slice(0, 10)
  }
  return { start_date: format(dates[0]), end_date: format(dates[1]) }
}

function inferTravelDate(text: string): string | undefined {
  const normalized = text.replace(/[०-९]/g, (digit) => String("०१२३४५६७८९".indexOf(digit)))
  const match = normalized.match(/\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{4})\b/iu)
  if (!match) return undefined
  const months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
  const hindiMonths = ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "सितम्बर", "अक्टूबर", "नवंबर", "दिसंबर"]
  const monthName = match[2].toLowerCase()
  const monthIndex = months.findIndex((name) => name.startsWith(monthName))
  const index = monthIndex >= 0 ? monthIndex : hindiMonths.indexOf(match[2])
  if (index < 0) return undefined
  const date = new Date(Date.UTC(Number(match[3]), index, Number(match[1])))
  if (date.getUTCFullYear() !== Number(match[3]) || date.getUTCMonth() !== index || date.getUTCDate() !== Number(match[1])) return undefined
  return date.toISOString().slice(0, 10)
}

function inferBudget(text: string): number | undefined {
  const normalized = text.replace(/[०-९]/g, (digit) => String("०१२३४५६७८९".indexOf(digit)))
  const patterns = [/(?:₹|rs\.?|inr)\s*([0-9][0-9,]{2,})/iu, /\b(?:budget|बजट)\s*(?:of|is|under|है|का)?\s*([0-9][0-9,]{2,})/iu, /([0-9][0-9,]{2,})\s*(?:budget|बजट|rupees?|रुपये?)/iu]
  for (const pattern of patterns) {
    const match = normalized.match(pattern)
    if (match) return Number(match[1].replace(/,/g, ""))
  }
  return undefined
}

function inferTravelers(text: string): number | undefined {
  const match = text.replace(/[०-९]/g, (digit) => String("०१२३४५६७८९".indexOf(digit))).match(/\b(\d+)\s*(?:people|persons?|travell?ers?|adults?|log|लोग|व्यक्ति)\b/iu)
  return match ? Math.max(1, Math.min(20, Number(match[1]))) : undefined
}

function restoreJourney(raw: Journey): Journey {
  const originalOrigin = raw.origin ?? ""
  const malformedOrigin = originalOrigin.length > 45 || /\b(?:coming from|need to be|plan my trip)\b/i.test(originalOrigin) || isQuestionLike(originalOrigin) || isGreeting(originalOrigin) || isAcknowledgement(originalOrigin)
  const origin = malformedOrigin ? extractOrigin(originalOrigin) ?? "" : originalOrigin
  const initialPrompt = raw.initialPrompt ?? (malformedOrigin ? originalOrigin : undefined)
  const routeCities = raw.routeCities ?? extractRouteCities(raw.destination ?? "")
  const context = { ...(raw.context ?? {}) }
  const days = inferTripDays(initialPrompt ?? "")
  const dates = inferTravelDates(initialPrompt ?? "")
  if (days && !context.days) context.days = days
  if (dates) Object.assign(context, dates)
  return {
    ...raw,
    origin,
    initialPrompt,
    routeCities,
    ...(routeCities.length ? { destination: routeCities.join(" → ") } : {}),
    title: routeCities.length ? `${origin || "Trip"} → ${routeCities.join(" → ")}` : malformedOrigin ? origin ? `From ${origin}` : "Rajasthan trip planning" : raw.title,
    // Legacy journeys may still carry intake-stage markers. They are preserved
    // for display only; every new message now goes straight to open chat.
    stage: "planning",
    context,
    lastActive: new Date(raw.lastActive),
    messages: raw.messages.map((message) => ({ ...message, timestamp: new Date(message.timestamp) })),
  }
}

function languageCopy(lang: Lang) {
  return lang === "hi" ? {
    askOrigin: "आप कहाँ से यात्रा शुरू कर रहे हैं? मैं पहले राजस्थान पहुँचने के विकल्प दिखाऊँगा।",
    askCity: (origin: string) => `समझ गया—आप ${origin} से आ रहे हैं। राजस्थान में किन शहरों या जगहों पर जाना चाहेंगे? उदाहरण: उदयपुर, जयपुर।`,
    originRetry: "मैं आपके शुरुआती शहर को समझ नहीं पाया। कृपया सिर्फ शहर का नाम लिखें, जैसे पुणे या दिल्ली।",
    cityRetry: "कृपया राजस्थान का शहर या यात्रा-क्रम लिखें, जैसे उदयपुर और फिर जयपुर।",
    askDetails: "कितने दिन की यात्रा है, कितने लोग जा रहे हैं, और आपका बजट या रुचियाँ क्या हैं?",
    routeUnavailable: "यात्रा विकल्प सेवा उपलब्ध नहीं है। कृपया बैकएंड API को नवीनतम कोड के साथ रीस्टार्ट करें।",
    liveNotice: "ट्रेन का लाइव समय/स्थिति RailRadar से हो सकती है; यह टिकट उपलब्धता या किराया नहीं है। बस और कैब के दाम अनुमान हैं।",
    noEstimate: "इस मार्ग का अनुमान अभी उपलब्ध नहीं है।",
    routeTitle: "यात्रा विकल्प", train: "ट्रेन", bus: "बस", cab: "निजी कैब", flight: "उड़ान",
  } : {
    askOrigin: "What city are you coming from? I’ll first show ways to reach Rajasthan.",
    askCity: (origin: string) => `Got it—you’re coming from ${origin}. Which Rajasthan cities or places would you like to visit? For example: Udaipur, Jaipur.`,
    originRetry: "I couldn’t identify your starting city. Please enter just the city name, such as Pune or Delhi.",
    cityRetry: "Please name a Rajasthan city or route, such as Udaipur and then Jaipur.",
    askDetails: "How many days are you travelling, how many people, and what budget or interests should I plan around?",
    routeUnavailable: "The route options service is unavailable. Please restart the backend API with the updated project code.",
    liveNotice: "Train schedules/status may be live from RailRadar; this is not ticket availability or fare data. Bus and cab prices are estimates.",
    noEstimate: "No estimate is available for this mode on this route.",
    routeTitle: "Travel options", train: "Train", bus: "Bus", cab: "Private cab", flight: "Flight",
  }
}

// ─── i18n ──────────────────────────────────────────────────────────────────

const t: Record<string, Record<Lang, string>> = {
  appName:       { en: "Rajwada Travel Agent", hi: "राजवाडा ट्रैवल एजेंट" },
  tagline:       { en: "Your thoughtful Rajasthan companion", hi: "आपका विचारशील राजस्थान साथी" },
  newTrip:       { en: "+ Naya Safar", hi: "+ नया सफ़र" },
  yourJourneys:  { en: "Chat history", hi: "पुरानी चैट" },
  savedPlaces:   { en: "Saved places", hi: "सहेजी जगहें" },
  shareItin:     { en: "Share Itinerary", hi: "यात्रा शेयर करें" },
  exportPDF:     { en: "Export PDF", hi: "PDF निर्यात" },
  bookCab:       { en: "Book Cab", hi: "कैब बुक करें" },
  placeholder:   { en: "Ask anything about your Rajasthan trip…", hi: "राजस्थान यात्रा के बारे में कुछ भी पूछें…" },
  disclaimer:    { en: "AI guidance can be imperfect. Confirm permits, schedules and availability.", hi: "AI मार्गदर्शन अपूर्ण हो सकता है। परमिट, समय और उपलब्धता की पुष्टि करें।" },
  morning:       { en: "Morning", hi: "सुबह" },
  afternoon:     { en: "Afternoon", hi: "दोपहर" },
  evening:       { en: "Evening", hi: "शाम" },
  dayLabel:      { en: "Day", hi: "दिन" },
  budgetTitle:   { en: "Where your budget goes", hi: "बजट का विवरण" },
  budgetEst:     { en: "Transparent estimate", hi: "पारदर्शी अनुमान" },
  checkCabRates: { en: "Check Cab Rates", hi: "कैब दरें देखें" },
  checkFlights:  { en: "Find Flights", hi: "उड़ानें देखें" },
  viewHotel:     { en: "View Palace Hotels", hi: "पैलेस होटल देखें" },
  budgetTier:    { en: "Budget", hi: "बजट" },
  midTier:       { en: "Mid-Range", hi: "मध्यम श्रेणी" },
  luxuryTier:    { en: "Royal / Luxury", hi: "शाही / लक्जरी" },
  perPerson:     { en: "per person", hi: "प्रति व्यक्ति" },
  evidenceTitle: { en: "Why this plan", hi: "यह योजना क्यों" },
  searching:     { en: "Planning your itinerary…", hi: "आपकी यात्रा योजना बना रहे हैं…" },
  tripSnapshot:  { en: "Trip snapshot", hi: "यात्रा का सारांश" },
  duration:      { en: "Duration", hi: "अवधि" },
  days:          { en: "days", hi: "दिन" },
  estTotal:      { en: "Estimated total", hi: "अनुमानित कुल" },
  groupTotal:    { en: "Group total", hi: "समूह कुल" },
  yourBudget:    { en: "Your budget", hi: "आपका बजट" },
  remaining:     { en: "Remaining buffer", hi: "शेष बजट" },
  planningEst:   { en: "Planning estimate", hi: "योजना अनुमान" },
  flightOptions: { en: "Flight options", hi: "उड़ान विकल्प" },
  trainOptions:  { en: "Train options", hi: "ट्रेन विकल्प" },
  busOptions:    { en: "Bus options", hi: "बस विकल्प" },
  cabOptions:    { en: "Cab options", hi: "कैब विकल्प" },
  routeOptions:  { en: "Travel options", hi: "यात्रा विकल्प" },
  liveData:      { en: "Live data", hi: "लाइव डेटा" },
  estimateData:  { en: "Estimate · verify directly", hi: "अनुमान · सीधे जाँचें" },
  travelersLabel:{ en: "Travellers", hi: "यात्री" },
}

function tr(key: string, lang: Lang): string {
  return t[key]?.[lang] ?? t[key]?.en ?? key
}

// ─── Icon Paths ─────────────────────────────────────────────────────────────

const iconPaths: Record<IconName, ReactNode> = {
  arch: (<><path d="M5 20V11a7 7 0 0 1 14 0v9" /><path d="M9 20v-9a3 3 0 0 1 6 0v9M3 20h18" /></>),
  arrow: (<><path d="m5 12 14-7-4 14-3-6-7-1Z" /><path d="m12 13 7-8" /></>),
  attach: <path d="m20 11-8.4 8.4a5 5 0 0 1-7.1-7.1l9.2-9.2a3.5 3.5 0 1 1 5 5l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" />,
  bookmark: <path d="M6 4.5A1.5 1.5 0 0 1 7.5 3h9A1.5 1.5 0 0 1 18 4.5V21l-6-4-6 4V4.5Z" />,
  calendar: (<><path d="M5 4h14a2 2 0 0 1 2 2v14H3V6a2 2 0 0 1 2-2ZM8 2v4m8-4v4M3 9h18" /></>),
  check: <path d="m5 12 4 4L19 6" />,
  chevron: <path d="m8 10 4 4 4-4" />,
  clock: (<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>),
  close: <path d="m6 6 12 12M18 6 6 18" />,
  compass: (<><circle cx="12" cy="12" r="9" /><path d="m15 9-2 4-4 2 2-4 4-2Z" /></>),
  download: (<><path d="M12 3v12m-4-4 4 4 4-4M4 19h16" /></>),
  food: (<><path d="M7 3v8m-3-8v5a3 3 0 0 0 6 0V3M7 11v10M16 3v18m0-18c3 2 4 5 4 8h-4" /></>),
  location: (<><path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z" /><circle cx="12" cy="10" r="2.5" /></>),
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  mic: (<><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0m-7 7v3" /></>),
  moon: <path d="M20 15.5A8 8 0 0 1 8.5 4 8 8 0 1 0 20 15.5Z" />,
  more: (<><circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" /></>),
  plus: <path d="M12 5v14M5 12h14" />,
  route: (<><circle cx="6" cy="18" r="2" /><circle cx="18" cy="6" r="2" /><path d="M8 18h3a3 3 0 0 0 3-3V9a3 3 0 0 1 3-3" /></>),
  settings: (<><circle cx="12" cy="12" r="3" /><path d="M19 13.5v-3l-2-.7-.7-1.7.9-2-2.2-2.2-2 .9-1.7-.7-.7-2h-3l-.7 2-1.7.7-2-.9L1 6.1l.9 2-.7 1.7-2 .7v3l2 .7.7 1.7-.9 2L3.2 20l2-.9 1.7.7.7 2h3l.7-2 1.7-.7 2 .9 2.2-2.2-.9-2 .7-1.7 2-.6Z" transform="scale(.86) translate(2 2)" /></>),
  share: (<><circle cx="18" cy="5" r="2" /><circle cx="6" cy="12" r="2" /><circle cx="18" cy="19" r="2" /><path d="m8 11 8-5m-8 7 8 5" /></>),
  shield: (<><path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6l-8-3Z" /><path d="m8.5 12 2 2 5-5" /></>),
  spark: <path d="M12 2c.8 5.5 3 7.7 8 8-5 .7-7.2 3-8 8-.8-5-3-7.3-8-8 5-.3 7.2-2.5 8-8Z" />,
  sun: (<><circle cx="12" cy="12" r="3.5" /><path d="M12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M19 5l-2 2M7 17l-2 2" /></>),
  sunrise: (<><path d="M4 18h16M6 14h12M12 4v3m-6 4-2-2m14 2 2-2M8 14a4 4 0 0 1 8 0" /></>),
  users: (<><circle cx="9" cy="8" r="3" /><path d="M3 20v-2a6 6 0 0 1 12 0v2M16 5a3 3 0 0 1 0 6m1 3a5 5 0 0 1 4 5" /></>),
  star: (<><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" /></>),
  cab: (<><path d="M5 17H3a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h11l4 4v4a2 2 0 0 1-2 2h-1" /><circle cx="7" cy="17" r="2" /><circle cx="15" cy="17" r="2" /><path d="M9 11V7" /></>),
  hotel: (<><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></>),
  export: (<><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></>),
  translate: (<><path d="M5 8l6 6" /><path d="M4 14l6-6 2-3" /><path d="M2 5h12" /><path d="M7 2h1" /><path d="M22 22l-5-10-5 10" /><path d="M14 18h6" /></>),
  edit: (<><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" /></>),
  trash: (<><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></>),
  collapse: <path d="M9 18l6-6-6-6" />,
  loading: <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48 2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48 2.83-2.83" />,
  send: <path d="m5 12 14-7-4 14-3-6-7-1Z" />,
  wallet: (<><rect x="2" y="5" width="20" height="14" rx="2" /><path d="M16 14h2" /></>),
  tag: (<><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z" /><line x1="7" y1="7" x2="7.01" y2="7" /></>),
  plane: (<><path d="M17.8 19.2 16 11l3.5-3.5a2.1 2.1 0 0 0-3-3L13 8 4.8 6.2a.5.5 0 0 0-.5.8l3.5 3.5-2 2-2.5-.5a.5.5 0 0 0-.5.8L5 16l1.2 2.2a.5.5 0 0 0 .8-.5l-.5-2.5 2-2 3.5 3.5a.5.5 0 0 0 .8-.5Z" /></>),
}

function Icon({ name, size = 18, className = "" }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={`icon ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {iconPaths[name]}
    </svg>
  )
}

// ─── Brand Mark ─────────────────────────────────────────────────────────────

function BrandMark({ small = false }: { small?: boolean }) {
  return (
    <span className={`brand-mark ${small ? "small" : ""}`} aria-hidden="true">
      <span className="brand-sun"><Icon name="spark" size={small ? 13 : 16} /></span>
      <span className="brand-arch"><Icon name="arch" size={small ? 12 : 14} /></span>
    </span>
  )
}

// Render the small, predictable Markdown subset returned by the planner.
// This intentionally avoids injecting HTML from model-generated text.
function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g)
  return <>{parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>
    if (part.startsWith("*") && part.endsWith("*")) return <em key={index}>{part.slice(1, -1)}</em>
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>
    return part
  })}</>
}

function MessageBody({ text }: { text: string }) {
  const blocks = text.trim().split(/\n\s*\n/).filter(Boolean)
  return <div className="message-body">{blocks.map((block, index) => {
    const lines = block.split("\n").map((line) => line.trim()).filter(Boolean)
    if (lines.every((line) => /^[-*•]\s+/.test(line))) {
      return <ul key={index}>{lines.map((line, item) => <li key={item}><InlineMarkdown text={line.replace(/^[-*•]\s+/, "")} /></li>)}</ul>
    }
    return lines.map((line, item) => {
      const heading = line.match(/^(#{1,3})\s+(.+)$/)
      if (heading) {
        const content = <InlineMarkdown text={heading[2]} />
        if (heading[1].length === 1) return <h2 key={`${index}-${item}`}>{content}</h2>
        if (heading[1].length === 2) return <h3 key={`${index}-${item}`}>{content}</h3>
        return <h4 key={`${index}-${item}`}>{content}</h4>
      }
      const bullet = line.match(/^[-*•]\s+(.+)$/)
      // The bullet row is a 2-column grid. InlineMarkdown returns a fragment whose
      // segments would each become their own implicit grid cell and collapse to one
      // character per line, so the content is wrapped in a single element.
      return <p key={`${index}-${item}`} className={bullet ? "message-bullet" : undefined}>{bullet ? <><span aria-hidden="true">✦</span><span className="message-bullet-text"><InlineMarkdown text={bullet[1]} /></span></> : <InlineMarkdown text={line} />}</p>
    })
  })}</div>
}

// ─── Status Badges ──────────────────────────────────────────────────────────

function StatusBadge({ type, children }: { type: "local" | "estimate" | "confirm" | "success" | "live" | "warning"; children: ReactNode }) {
  return (
    <span className={`status-badge ${type}`}>
      <span className="badge-dot" />
      {children}
    </span>
  )
}

// ─── Loading Dots ───────────────────────────────────────────────────────────

function LoadingDots() {
  return (
    <div className="loading-dots" aria-label="Thinking…">
      <span /><span /><span />
    </div>
  )
}

// ─── Static Journey Data ─────────────────────────────────────────────────────

function createJourney(lang: Lang = "en"): Journey {
  return {
    id: `trip-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: "New Rajasthan trip",
    messages: [{ id: `welcome-${Date.now()}`, role: "assistant", text: lang === "hi"
      ? "नमस्ते! मैं आपका राजस्थान यात्रा सहायक हूँ। 😊 आप कुछ भी पूछ सकते हैं—जगहें, रास्ते, ट्रेन, उड़ान, खाना या ठहरने के बारे में।"
      : "Hey there! I’m your Rajasthan travel assistant. 😊 Ask me anything—places, routes, trains, flights, food, or stays.", timestamp: new Date() }],
    lastActive: new Date(),
    stage: "planning",
  }
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────

function Sidebar({
  open, onClose, lang, onLangChange, theme, onThemeChange, activeJourneyId, journeys, savedPlaces, cities, origin, apiStatus, onJourneySelect, onNewJourney, onSavedPlace
}: {
  open: boolean
  onClose: () => void
  lang: Lang
  onLangChange: (l: Lang) => void
  theme: Theme
  onThemeChange: (t: Theme) => void
  activeJourneyId: string
  journeys: Journey[]
  savedPlaces: string[]
  cities?: string[]
  origin?: string
  apiStatus: { llm_enabled?: boolean; llm_provider?: string; places_enabled?: boolean; gemini_fallback_enabled?: boolean; gemini_fallback_model?: string; groq_fallback_enabled?: boolean; groq_fallback_model?: string; railradar_enabled?: boolean } | null
  onJourneySelect: (id: string) => void
  onNewJourney: () => void
  onSavedPlace: (place: string) => void
}) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  return (
    <>
      {open && <button className="drawer-backdrop" aria-label="Close navigation" onClick={onClose} />}
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Navigation">
        <button className="mobile-close" onClick={onClose} aria-label="Close menu">
          <Icon name="close" />
        </button>

        <div className="brand">
          <BrandMark />
          <div>
            <strong>Rajwada Travel Agent</strong>
            <span>Your thoughtful Rajasthan companion</span>
          </div>
        </div>

        <button className="new-trip" onClick={onNewJourney} id="new-journey-btn">
          <Icon name="plus" size={16} />
          <span>{tr("newTrip", lang)}</span>
        </button>

        <nav className="sidebar-scroll" aria-label="Journey navigation">
          <div className="sidebar-map"><span className="nav-label">Trip map</span><RouteMap cities={cities} origin={origin} /></div>
          <div className="nav-label">{tr("yourJourneys", lang)}</div>
          {journeys.map((j) => (
            <button
              key={j.id}
              className={`journey ${j.id === activeJourneyId ? "selected" : ""}`}
              onClick={() => { onJourneySelect(j.id); onClose() }}
              aria-current={j.id === activeJourneyId ? "page" : undefined}
            >
              <Icon name="route" size={14} />
              <span>
                <strong>{j.title}</strong>
                {j.route && <small>{j.route}</small>}
              </span>
              <span className="item-more"><Icon name="more" size={14} /></span>
            </button>
          ))}

          <div className="nav-label saved-label">{tr("savedPlaces", lang)}</div>
          {savedPlaces.map((place) => (
            <button className="saved" key={place} onClick={() => { onSavedPlace(place); onClose() }}>
              <Icon name="bookmark" size={13} />
              <span>{place}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="local-badge">
            <span className="pulse-dot" />
            {apiStatus?.llm_enabled ? `${apiStatus.llm_provider ?? "AI model"} configured` : "Local planner ready"}
          </div>
          <small className="provider-status">Places: {apiStatus?.places_enabled ? "live search on" : "add Google Places key for live listings"}</small>
          <small className="provider-status">AI fallback: {[apiStatus?.groq_fallback_enabled ? apiStatus.groq_fallback_model ?? "Groq" : null, apiStatus?.gemini_fallback_enabled ? apiStatus.gemini_fallback_model ?? "Gemini" : null].filter(Boolean).join(" → ") || "not configured"}</small>

          <div className="lang-theme-row">
            <button
              className={`lang-btn ${lang === "en" ? "active" : ""}`}
              onClick={() => onLangChange("en")}
              aria-label="Switch to English"
              id="lang-en-btn"
            >EN</button>
            <button
              className={`lang-btn ${lang === "hi" ? "active" : ""}`}
              onClick={() => onLangChange("hi")}
              aria-label="Switch to Hindi"
              id="lang-hi-btn"
            >हि</button>
            <button
              className="theme-toggle"
              onClick={() => onThemeChange(theme === "light" ? "dark" : "light")}
              aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
              id="theme-toggle-btn"
            >
              <Icon name={theme === "light" ? "moon" : "sun"} size={15} />
            </button>
          </div>

          <div className="profile">
            <span className="avatar">AM</span>
            <span>
              <strong>Aarav Mehta</strong>
              <small>Traveller profile</small>
            </span>
            <button aria-label="Settings" id="settings-btn" aria-expanded={settingsOpen} onClick={() => setSettingsOpen((open) => !open)}><Icon name="settings" size={16} /></button>
          </div>
          {settingsOpen && <div className="settings-popover"><strong>Connection status</strong><span>{apiStatus?.llm_enabled ? `${apiStatus.llm_provider ?? "AI model"} primary configured (request not tested)` : "Primary AI API key not configured"}</span><span>{apiStatus?.groq_fallback_enabled ? `${apiStatus.groq_fallback_model ?? "Groq"} fallback configured` : "Groq fallback key not configured"}</span><span>{apiStatus?.gemini_fallback_enabled ? `${apiStatus.gemini_fallback_model ?? "Gemini"} fallback configured` : "Gemini fallback key not configured"}</span><span>{apiStatus?.railradar_enabled ? "RailRadar train API configured" : "RailRadar train API not configured"}</span><span>{apiStatus?.places_enabled ? "Google Places live discovery connected" : "Google Places key not configured"}</span><small>Set these in the project root .env, then restart the API. Provider failures are logged without request details.</small></div>}
        </div>
      </aside>
    </>
  )
}

// ─── Timeline Day Card ───────────────────────────────────────────────────────

function TimelineCard({ day, lang, onAsk }: { day: ItineraryDay; lang: Lang; onAsk?: (text: string) => void }) {
  const [expanded, setExpanded] = useState(true)

  const slots = [
    { icon: "sunrise" as const, label: tr("morning", lang), text: day.morning },
    { icon: "sun" as const, label: tr("afternoon", lang), text: day.afternoon },
    { icon: "moon" as const, label: tr("evening", lang), text: day.evening },
  ]

  return (
    <article className="day-card" id={`day-card-${day.day}`}>
      <div className="day-step">
        <span>{String(day.day).padStart(2, "0")}</span>
      </div>
      <div className="day-content">
        <button
          className="day-heading-btn"
          onClick={() => setExpanded((e) => !e)}
          aria-expanded={expanded}
        >
          <div className="day-heading-inner">
            <p>{tr("dayLabel", lang)} {day.day}</p>
            <h3>{day.city}</h3>
          </div>
          <Icon name="chevron" size={16} className={`chevron-icon ${expanded ? "open" : ""}`} />
        </button>

        {expanded && (
          <div className="day-body">
            <div className="day-grid">
              {slots.map(({ icon, label, text }) => (
                <div className="day-item" key={label}>
                  <span className="time-icon"><Icon name={icon} size={15} /></span>
                  <div>
                    <small>{label}</small>
                    <p>{text}</p>
                  </div>
                </div>
              ))}
            </div>

            {day.meal && (
              <div className="notes-row">
                <span><Icon name="food" size={13} /> Food</span>
                {day.meal}
              </div>
            )}
            {day.transport && (
              <div className="notes-row practical">
                <span><Icon name="route" size={13} /> Transport</span>
                {day.transport}
              </div>
            )}
            {day.note && (
              <div className="safety-note">
                <Icon name="shield" size={15} />
                <span>{day.note}</span>
              </div>
            )}

            <div className="day-actions">
              <button className="action-badge-btn" id={`flight-btn-day-${day.day}`} onClick={() => onAsk?.(`Find flights to ${day.city}`)}>
                <Icon name="plane" size={13} />
                {tr("checkFlights", lang)}
              </button>
              <button className="action-badge-btn" id={`cab-btn-day-${day.day}`}>
                <Icon name="cab" size={13} />
                {tr("checkCabRates", lang)}
              </button>
              <button className="action-badge-btn hotel" id={`hotel-btn-day-${day.day}`}>
                <Icon name="hotel" size={13} />
                {tr("viewHotel", lang)}
              </button>
            </div>
          </div>
        )}
      </div>
    </article>
  )
}

// ─── Budget Estimator ────────────────────────────────────────────────────────

function BudgetEstimator({ budget, lang, travelers = 2, style }: { budget: Budget; lang: Lang; travelers?: number; style?: string }) {
  const [activeTier, setActiveTier] = useState<0 | 1 | 2>(1)

  const tiers = [
    {
      label: tr("budgetTier", lang),
      multiplier: 0.7,
      color: "tier-budget",
      icon: "wallet" as const,
      desc: "Hostels, local dhabas, shared cabs",
    },
    {
      label: tr("midTier", lang),
      multiplier: 1.0,
      color: "tier-mid",
      icon: "star" as const,
      desc: "3-star stays, casual restaurants, private transport",
    },
    {
      label: tr("luxuryTier", lang),
      multiplier: 1.85,
      color: "tier-luxury",
      icon: "spark" as const,
      desc: "Heritage hotels, fine dining, curated experiences",
    },
  ]

  const base = budget.per_person
  const activeMult = tiers[activeTier].multiplier
  const totalPP = Math.round(base * activeMult)
  // group_total is the whole party at the base tier, so the tier-adjusted party
  // total and remaining buffer must both scale by the same multiplier. The
  // previous code hardcoded 2 travellers and ignored the selected tier.
  const groupPP = Math.round(base * travelers * activeMult)
  const remaining = Math.max(0, groupPP - budget.group_total)

  const breakdownEntries = Object.entries(budget.breakdown_per_person)
  const colors = ["clr-stay", "clr-food", "clr-transport", "clr-activity", "clr-contingency"]

  return (
    <section className="budget-estimator" id="budget-estimator">
      <div className="budget-top">
        <div>
          <span className="eyebrow">{tr("budgetEst", lang)}</span>
          <h2>{tr("budgetTitle", lang)}</h2>
        </div>
        <StatusBadge type="estimate">{tr("planningEst", lang)}</StatusBadge>
      </div>

      <div className="tier-switcher" role="group" aria-label="Select budget tier">
        {tiers.map((tier, i) => (
          <button
            key={tier.label}
            className={`tier-btn ${i === activeTier ? "active" : ""} ${tier.color}`}
            onClick={() => setActiveTier(i as 0 | 1 | 2)}
            id={`tier-btn-${i}`}
          >
            <Icon name={tier.icon} size={14} />
            {tier.label}
          </button>
        ))}
      </div>

      <div className="tier-desc">{tiers[activeTier].desc}</div>

      <div className="stacked-bar" aria-label="Budget allocation">
        {breakdownEntries.map(([name, val], idx) => {
          const pct = Math.round((val / base) * 100)
          return (
            <span
              key={name}
              className={colors[idx % colors.length]}
              style={{ width: `${pct}%` }}
              title={name}
            />
          )
        })}
      </div>

      <div className="budget-legend">
        {breakdownEntries.map(([name, val], idx) => (
          <div key={name}>
            <span className={`legend-dot ${colors[idx % colors.length]}`} />
            <span>{name}</span>
            <strong>₹{Math.round(val * activeMult).toLocaleString()}</strong>
          </div>
        ))}
      </div>

      <div className="budget-summary">
        <div>
          <small>{tr("estTotal", lang)}</small>
          <strong>₹{totalPP.toLocaleString()} <em>{tr("perPerson", lang)}</em></strong>
        </div>
        <div>
          <small>{tr("groupTotal", lang)}</small>
          <strong>₹{groupPP.toLocaleString()}</strong>
        </div>
        <div className="remaining">
          <small>{tr("remaining", lang)}</small>
          <strong>₹{remaining.toLocaleString()}</strong>
        </div>
      </div>
      <p className="budget-meta">
        {tr("travellersLabel", lang)}: {travelers}
        {style ? ` · ${style}` : ""}
      </p>

      {budget.over_budget && (
        <div className="over-budget-note">
          <Icon name="shield" size={14} />
          {budget.note}
        </div>
      )}

      <p className="estimate-note">
        <Icon name="shield" size={14} />
        Planning estimate, not a booking quote.
      </p>
    </section>
  )
}

// ─── Trip Snapshot ───────────────────────────────────────────────────────────

function TripSnapshot({ plan, lang, travelers = 2, style }: { plan: NonNullable<AgentResult["plan"]>; lang: Lang; travelers?: number; style?: string }) {
  // These read real planner state; they were previously hardcoded placeholders.
  const styleLabel = style ? style.charAt(0).toUpperCase() + style.slice(1) : "Balanced"
  return (
    <section className="snapshot">
      <div className="section-kicker">
        <Icon name="compass" size={15} /> {tr("tripSnapshot", lang)}
      </div>
      <div className="snapshot-grid">
        <div>
          <Icon name="calendar" size={18} />
          <span>
            <small>{tr("duration", lang)}</small>
            <strong>{plan.itinerary.length} {tr("days", lang)}</strong>
          </span>
        </div>
        <div>
          <Icon name="users" size={18} />
          <span>
            <small>{tr("travellersLabel", lang)}</small>
            <strong>{travelers}</strong>
          </span>
        </div>
        <div className="wide">
          <Icon name="location" size={18} />
          <span>
            <small>Route</small>
            <strong>{plan.cities.join(" → ")}</strong>
          </span>
        </div>
        <div>
          <Icon name="spark" size={18} />
          <span>
            <small>Style</small>
            <strong>{styleLabel}</strong>
          </span>
        </div>
        <div className="total">
          <span>
            <small>Est. group total</small>
            <strong>₹{plan.budget.group_total.toLocaleString()}</strong>
          </span>
          <StatusBadge type={plan.budget.over_budget ? "warning" : "success"}>
            {plan.budget.over_budget ? "Over budget" : tr("planningEst", lang)}
          </StatusBadge>
        </div>
      </div>
    </section>
  )
}

// ─── Evidence Collapse ───────────────────────────────────────────────────────

function EvidencePanel({ evidence, trace, prompting }: { evidence?: AgentResult["evidence"]; trace?: AgentResult["trace"]; prompting?: AgentResult["prompting"] }) {
  const [open, setOpen] = useState(() => Boolean(prompting && !["generated", "complete"].includes(prompting.status)))
  const promptStatus = prompting?.status ?? "fallback"
  const promptStatusLabel = promptStatus === "generated" ? "Grounded model response"
    : promptStatus === "tls_error" ? "AI connection failed · TLS certificate"
      : promptStatus === "network_error" || promptStatus === "connection_error" ? "AI service unreachable"
        : promptStatus === "provider_http_429_insufficient_quota" || promptStatus === "provider_http_429_credit_balance_exhausted" ? "OpenAI API quota or billing limit"
          : promptStatus.startsWith("provider_http_429") ? "OpenAI API rate limit reached"
            : promptStatus === "provider_http_401" || promptStatus === "provider_http_403" ? "OpenAI API access rejected"
        : promptStatus.startsWith("provider_http_") ? `AI provider rejected request · HTTP ${promptStatus.slice("provider_http_".length)}`
          : promptStatus === "invalid_output" ? "AI response format invalid"
            : "Deterministic fallback"
  return (
    <section className={`evidence ${open ? "expanded" : ""}`}>
      <button className="evidence-trigger" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="evidence-icon"><Icon name="spark" /></span>
        <span>
          <strong>Sources & details to verify</strong>
          <small>Optional planning notes</small>
        </span>
        <Icon name="chevron" className={`chevron-icon ${open ? "open" : ""}`} />
      </button>
      {open && (
        <div className="evidence-body">
          {prompting && (
            <div className="evidence-column">
              <h3>Answer generation</h3>
              <p><Icon name={prompting.status === "generated" ? "check" : "clock"} size={13} />{promptStatusLabel} · {prompting.strategy}</p>
            </div>
          )}
          {trace && trace.length > 0 && (
            <div className="evidence-column">
              <h3>Agent activity</h3>
              {trace.map((t, i) => (
                <p key={i}>
                  <Icon name={t.status === "complete" ? "check" : "clock"} size={13} />
                  {t.tool}
                </p>
              ))}
            </div>
          )}
          {evidence && evidence.length > 0 && (
            <div className="evidence-column">
              <h3>Local knowledge used</h3>
              {evidence.slice(0, 4).map((e, i) => (
                <span className="source-chip" key={i}>
                  <Icon name="bookmark" size={12} />
                  {e.title}
                </span>
              ))}
            </div>
          )}
          <div className="evidence-column confirm-list">
            <h3>What needs confirmation</h3>
            {["Safari permit / zone / reporting time", "Hotel availability and price", "Restaurant dietary preparation", "Transport schedules"].map((x) => (
              <p key={x}><span />{x}</p>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

// ─── Transport Results (flights / trains / compare) ──────────────────────────

function TransportResults({ options, lang }: { options?: TransportOptions; lang: Lang }) {
  if (!options) return null
  const flights = options.flights ?? []
  const trains = options.trains ?? options.live_trains ?? []
  const estimates = options.options ?? []
  if (!flights.length && !trains.length && !estimates.length) return null

  const heading = options.mode === "flights" ? tr("flightOptions", lang)
    : options.mode === "trains" ? tr("trainOptions", lang)
    : options.mode === "buses" ? tr("busOptions", lang)
    : options.mode === "cabs" ? tr("cabOptions", lang)
    : tr("routeOptions", lang)

  return (
    <section className="transport-card" id={`transport-${options.mode ?? "any"}`}>
      <div className="section-heading compact">
        <span className="eyebrow">{options.from} → {options.to}</span>
        <h2>{heading}</h2>
      </div>

      {flights.length > 0 && (
        <div className="transport-list">
          {flights.map((f, i) => (
            <article className="transport-row" key={`flight-${i}`}>
              <div className="transport-main">
                <strong>{[f.airline, f.flight_number].filter(Boolean).join(" ") || "Option"}</strong>
                <span className="transport-times">
                  {f.departure_time ?? "—"} <em>→</em> {f.arrival_time ?? "—"}
                </span>
              </div>
              <div className="transport-meta">
                {f.duration_minutes ? <span>{f.duration_minutes} min</span> : null}
                {f.stops ? <span>{f.stops}</span> : null}
                {f.aircraft ? <span>{f.aircraft}</span> : null}
                {f.cabin ? <span>{f.cabin}</span> : null}
                {typeof f.indicative_fare_inr === "number"
                  ? <span className="transport-fare">₹{f.indicative_fare_inr.toLocaleString()}</span>
                  : null}
              </div>
            </article>
          ))}
        </div>
      )}

      {trains.length > 0 && (
        <div className="transport-list">
          {trains.map((row, i) => (
            <article className="transport-row" key={`train-${i}`}>
              <div className="transport-main">
                <strong>{[row.name, row.number].filter(Boolean).join(" · ") || "Train"}</strong>
                <span className="transport-times">
                  {row.departure ?? "—"} <em>→</em> {row.arrival ?? "—"}
                </span>
              </div>
              <div className="transport-meta">
                {row.duration_minutes ? <span>{row.duration_minutes} min</span> : null}
                {row.type ? <span>{row.type}</span> : null}
                {row.status ? <span className="transport-status">{row.status}</span> : null}
                {typeof row.delay_minutes === "number" ? <span>delay {row.delay_minutes} min</span> : null}
              </div>
            </article>
          ))}
        </div>
      )}

      {flights.length === 0 && trains.length === 0 && estimates.length > 0 && (
        <div className="transport-list">
          {estimates.map((o, i) => (
            <article className="transport-row" key={`opt-${i}`}>
              <div className="transport-main">
                <strong>{o.mode}</strong>
                <span className="transport-times">{o.duration}</span>
              </div>
              <div className="transport-meta">
                <span className="transport-fare">{o.cost}</span>
                {o.book_at ? <span className="transport-book">{o.book_at}</span> : null}
              </div>
            </article>
          ))}
        </div>
      )}

      <div className="transport-foot">
        <StatusBadge type={options.live ? "live" : "estimate"}>
          {options.live ? tr("liveData", lang) : tr("estimateData", lang)}
        </StatusBadge>
        {options.source ? <small className="transport-source">{options.source}</small> : null}
      </div>
      {options.notice ? <p className="transport-notice">{options.notice}</p> : null}
    </section>
  )
}

// ─── Assistant Response Card ─────────────────────────────────────────────────

function AssistantMessage({ message, lang, onSavePlace, onAsk }: { message: ChatMessage; lang: Lang; onSavePlace: (place: string) => void; onAsk?: (text: string) => void }) {
  const { text, result } = message
  const plan = result?.plan
  // Prefer the mode-specific payload so a flight request never renders train rows.
  const transport = result?.flight_options ?? result?.train_options ?? result?.transport_options

  return (
    <div className="assistant-row" aria-live="polite">
      <BrandMark />
      <div className="assistant-content">
        <MessageBody text={text} />
        <TransportResults options={transport} lang={lang} />

        {plan && (
          <>
            <TripSnapshot plan={plan} lang={lang}
              travelers={Number(result?.context?.travelers) || 2}
              style={typeof result?.context?.style === "string" ? result.context.style : undefined} />

            <section className="itinerary-section">
              <div className="section-heading">
                <span className="eyebrow">Your route</span>
                <h2>{plan.cities.join(" · ")} · {plan.itinerary.length} days</h2>
                <p>Built with breathing room and local insights.</p>
              </div>
              <div className="timeline">
                {plan.itinerary.map((day) => (
                  <TimelineCard key={day.day} day={day} lang={lang} onAsk={onAsk} />
                ))}
              </div>
            </section>

            <BudgetEstimator budget={plan.budget} lang={lang}
              travelers={Number(result?.context?.travelers) || 2}
              style={typeof result?.context?.style === "string" ? result.context.style : undefined} />
          </>
        )}

        {result?.stays && (
          <section className="stay-card">
            <div className="stay-icon"><Icon name="arch" /></div>
            <div>
              <span className="eyebrow">Stay guidance</span>
              <h2>Where to stay</h2>
              <p>{result.stays.guidance ?? "Choose a well-reviewed stay near your confirmed safari reporting route."}</p>
              <div className="chip-row">
                <span>Near safari route</span>
                <span>Breakfast timing matters</span>
                <span>Verify pickup</span>
              </div>
            </div>
            <StatusBadge type="confirm">Needs confirmation</StatusBadge>
          </section>
        )}

        {!!(result?.hotels_by_city && Object.keys(result.hotels_by_city).length) && (
          <section className="food-section lodging-section">
            <div className="section-heading compact"><span className="eyebrow">Hotels · separate from hostels</span><h2>Places to stay</h2></div>
            {Object.entries(result.hotels_by_city).map(([city, stays]) => <div key={city} className="lodging-city"><h3>{city}</h3><div className="food-grid">{stays.map((stay, i) => <article className="food-card" key={`${city}-${i}`}><span className="food-icon"><Icon name="hotel" /></span><h3>{stay.name}</h3><p>{stay.address || stay.text}</p><StatusBadge type={result.places_live ? "live" : "estimate"}>{result.places_live ? "Live place listing · no availability/pricing" : "Illustrative reference · verify directly"}</StatusBadge><button className="action-badge-btn" onClick={() => onSavePlace(stay.name || city)}><Icon name="bookmark" size={13} />Save</button></article>)}</div></div>)}
          </section>
        )}

        {!!(result?.hostels_by_city && Object.keys(result.hostels_by_city).length) && (
          <section className="food-section lodging-section">
            <div className="section-heading compact"><span className="eyebrow">Hostels</span><h2>Hostel options</h2><p>Listed separately from hotels; check dorm and private-room availability directly.</p></div>
            {Object.entries(result.hostels_by_city).map(([city, stays]) => <div key={city} className="food-grid">{stays.map((stay, i) => <article className="food-card" key={`${city}-hostel-${i}`}><span className="food-icon"><Icon name="hotel" /></span><h3>{stay.name}</h3><p>{stay.address || stay.text}</p><StatusBadge type={result.places_live ? "live" : "estimate"}>{result.places_live ? "Live place listing · not booking availability" : "Illustrative reference · verify directly"}</StatusBadge><button className="action-badge-btn" onClick={() => onSavePlace(stay.name || city)}><Icon name="bookmark" size={13} />Save</button></article>)}</div>)}
          </section>
        )}

        {result?.restaurants && result.restaurants.length > 0 && (
          <section className="food-section">
            <div className="section-heading compact">
              <span className="eyebrow">Food ideas</span>
              <h2>Local dining picks</h2>
            </div>
            <div className="food-grid">
              {result.restaurants.slice(0, 3).map((r, i) => (
                <article className="food-card" key={i}>
                  <span className="food-icon"><Icon name="food" /></span>
                  <h3>{r.name ?? r.title ?? "Local eatery"}</h3>
                  <p>{r.text ?? "A well-reviewed local option."}</p>
                  <StatusBadge type={result.places_live ? "live" : "estimate"}>{result.places_live ? "Live place listing · verify details" : "Illustrative local reference"}</StatusBadge>
                </article>
              ))}
            </div>
          </section>
        )}

        {Object.entries(result?.restaurants_by_city ?? {}).filter(([city]) => city !== (plan?.cities ?? [])[0]).map(([city, places]) => <section className="food-section" key={`food-${city}`}><div className="section-heading compact"><span className="eyebrow">Food · {city}</span><h2>Local dining picks</h2></div><div className="food-grid">{places.map((place, i) => <article className="food-card" key={i}><span className="food-icon"><Icon name="food" /></span><h3>{place.name ?? place.title}</h3><p>{place.address || place.text}</p><StatusBadge type={result?.places_live ? "live" : "estimate"}>{result?.places_live ? "Live place listing · verify details" : "Illustrative local reference"}</StatusBadge></article>)}</div></section>)}

        {!!result?.attractions?.length && <section className="food-section"><div className="section-heading compact"><span className="eyebrow">Things to do</span><h2>Activities & places · {plan?.cities[0]}</h2></div><div className="food-grid">{result.attractions.map((place, i) => <article className="food-card" key={i}><span className="food-icon"><Icon name="compass" /></span><h3>{place.name}</h3><p>{place.address || place.status}</p><StatusBadge type={result.places_live ? "live" : "estimate"}>{result.places_live ? "Live place listing · verify access" : "Curated reference · verify access"}</StatusBadge></article>)}</div></section>}
        {Object.entries(result?.attractions_by_city ?? {}).filter(([city]) => city !== (plan?.cities ?? [])[0]).map(([city, places]) => <section className="food-section" key={`activities-${city}`}><div className="section-heading compact"><span className="eyebrow">Things to do · {city}</span><h2>Activities & places</h2></div><div className="food-grid">{places.map((place, i) => <article className="food-card" key={i}><span className="food-icon"><Icon name="compass" /></span><h3>{place.name}</h3><p>{place.address || place.status}</p><StatusBadge type={result?.places_live ? "live" : "estimate"}>{result?.places_live ? "Live place listing · verify access" : "Curated reference · verify access"}</StatusBadge></article>)}</div></section>)}

        {result && <EvidencePanel evidence={result.evidence} trace={result.trace} prompting={result.prompting} />}
      </div>
    </div>
  )
}

// ─── User Message Bubble ─────────────────────────────────────────────────────

function UserMessage({ text }: { text: string }) {
  return (
    <div className="user-row">
      <div className="user-bubble">{text}</div>
      <span className="user-avatar">AM</span>
    </div>
  )
}

// ─── Quick-start Chips (empty state) ────────────────────────────────────────

const quickStartChips: Record<Lang, string[]> = {
  en: [
    "Plan a 3-day Jaisalmer desert trip for two",
    "Best time to visit Ranthambore for tigers?",
    "Budget Jaipur trip under ₹15,000",
    "Udaipur romantic getaway itinerary",
    "Family trip to Pushkar — 5 days",
    "Compare Jodhpur vs Jaisalmer for 4 days",
  ],
  hi: [
    "जैसलमेर में 3 दिन की रेगिस्तान यात्रा",
    "रणथंभोर में बाघ देखने का सबसे अच्छा समय?",
    "₹15,000 में जयपुर बजट यात्रा",
    "उदयपुर रोमांटिक यात्रा",
    "पुष्कर पारिवारिक यात्रा — 5 दिन",
    "जोधपुर बनाम जैसलमेर — 4 दिन",
  ],
}

function EmptyState({ lang, onChip }: { lang: Lang; onChip: (text: string) => void }) {
  return (
    <div className="empty-state">
      <div className="empty-hero">
        <BrandMark />
        <h1>Rajwada Travel Agent</h1>
        <p>Your intelligent guide to the Land of Kings. Ask anything — from desert safaris to palace stays.</p>
      </div>
      <div className="quick-chips">
        {quickStartChips[lang].map((chip) => (
          <button key={chip} className="quick-chip" onClick={() => onChip(chip)} id={`chip-${chip.slice(0, 20).replace(/\s/g, "-")}`}>
            <Icon name="compass" size={13} />
            {chip}
          </button>
        ))}
      </div>
    </div>
  )
}

// ─── Composer (Input Bar) ────────────────────────────────────────────────────

const followUpSuggestions: Record<Lang, string[]> = {
  en: ["Make it cheaper", "Add a second safari", "Give vegetarian options", "Make it family-friendly", "What to book first?"],
  hi: ["सस्ता बनाएं", "दूसरी सफारी जोड़ें", "शाकाहारी विकल्प दें", "परिवार-अनुकूल बनाएं", "पहले क्या बुक करें?"],
}

function Composer({
  onSubmit, loading, lang, hasMessages
}: {
  onSubmit: (msg: string) => void
  loading: boolean
  lang: Lang
  hasMessages: boolean
}) {
  const [message, setMessage] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = useCallback(() => {
    const val = message.trim()
    if (!val || loading) return
    onSubmit(val)
    setMessage("")
    if (textareaRef.current) textareaRef.current.style.height = "auto"
  }, [message, loading, onSubmit])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit() }
  }

  const autoResize = () => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = "auto"
    ta.style.height = Math.min(ta.scrollHeight, 140) + "px"
  }

  return (
    <div className="composer-zone">
      {hasMessages && (
        <div className="suggestions">
          {followUpSuggestions[lang].map((s) => (
            <button key={s} className="suggestion-chip" onClick={() => setMessage(s)}>
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="composer">
        {/* Jharokha arch accent */}
        <div className="jharokha-accent" aria-hidden="true">
          <svg viewBox="0 0 60 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 16 Q10 0 20 8 Q30 16 40 8 Q50 0 60 16Z" fill="currentColor" opacity=".12" />
          </svg>
        </div>

        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => { setMessage(e.target.value); autoResize() }}
          placeholder={tr("placeholder", lang)}
          aria-label="Message input"
          rows={1}
          onKeyDown={handleKeyDown}
          id="message-input"
          disabled={loading}
        />

        <div className="composer-tools">
          <div className="composer-left">
            <button aria-label="Attach a file" id="attach-btn"><Icon name="attach" size={17} /></button>
            <button aria-label="Use microphone" id="mic-btn"><Icon name="mic" size={17} /></button>
          </div>
          <button
            className={`send-button ${loading ? "loading" : message.trim() ? "ready" : ""}`}
            aria-label="Send message"
            onClick={submit}
            disabled={loading || !message.trim()}
            id="send-btn"
          >
            {loading ? <Icon name="loading" size={16} className="spin" /> : <Icon name="send" size={16} />}
          </button>
        </div>
      </div>
      <p className="disclaimer">{tr("disclaimer", lang)}</p>
    </div>
  )
}

// ─── Chat Canvas ─────────────────────────────────────────────────────────────

function ChatCanvas({ messages, loading, lang, onChipClick, onSavePlace, onAsk }: {
  messages: ChatMessage[]
  loading: boolean
  lang: Lang
  onChipClick: (text: string) => void
  onSavePlace: (place: string) => void
  onAsk?: (text: string) => void
}) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  if (messages.length === 0 && !loading) {
    return <EmptyState lang={lang} onChip={onChipClick} />
  }

  return (
    <main className="conversation" aria-label="Chat messages">
      <div className="conversation-inner">
        {messages.map((msg) =>
          msg.role === "user"
            ? <UserMessage key={msg.id} text={msg.text} />
            : <AssistantMessage key={msg.id} message={msg} lang={lang} onSavePlace={onSavePlace} onAsk={onAsk} />
        )}
        {loading && (
          <div className="assistant-row" aria-live="polite" aria-label="Loading">
            <BrandMark />
            <div className="assistant-content">
              <div className="thinking-card">
                <LoadingDots />
                <span>{tr("searching", lang)}</span>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </main>
  )
}

// ─── Route Map ───────────────────────────────────────────────────────────────

function RouteMap({ cities, origin }: { cities?: string[]; origin?: string }) {
  const coordinates: Record<string, string> = { Jaipur: "106,78", "Sawai Madhopur": "142,123", Ranthambore: "142,123", Jodhpur: "65,130", Udaipur: "94,166", Pushkar: "150,65", Jaisalmer: "42,86" }
  const routePoints = [origin, ...(cities ?? [])].map((city) => city && coordinates[city]).filter(Boolean).join(" ")
  const active = (name: string) => cities?.some((city) => city.toLowerCase() === name.toLowerCase())
  return (
    <div className="route-map">
      <svg viewBox="0 0 240 250" role="img" aria-label="Stylised Rajasthan route map">
        <path className="map-outline" d="M54 28 105 14l24 25 42 1 29 34-13 37 27 42-27 22-6 46-48 15-35-25-44-5-18-39 14-36-19-33 28-29-5-41Z" />
        <path className="map-route" d="M106 78 142 123 94 166M106 78 65 130M106 78 150 65" />
        {routePoints && <polyline className="map-route" points={routePoints} fill="none" />}
        <g className={`map-point ${active("Jaipur") ? "active" : ""}`}><circle cx="106" cy="78" r="4" /><text x="114" y="76">Jaipur</text></g>
        <g className={`map-point ${active("Ranthambore") || active("Sawai Madhopur") ? "active" : ""}`}>
          <circle cx="142" cy="123" r="7" /><circle cx="142" cy="123" r="12" />
          <text x="151" y="126">Ranthambore</text>
        </g>
        <g className={`map-point ${active("Jodhpur") ? "active" : ""}`}><circle cx="65" cy="130" r="4" /><text x="28" y="145">Jodhpur</text></g>
        <g className={`map-point ${active("Udaipur") ? "active" : ""}`}><circle cx="94" cy="166" r="4" /><text x="101" y="174">Udaipur</text></g>
        <g className={`map-point ${active("Pushkar") ? "active" : ""}`}><circle cx="150" cy="65" r="4" /><text x="158" y="62">Pushkar</text></g>
        <g className={`map-point ${active("Jaisalmer") ? "active" : ""}`}><circle cx="42" cy="86" r="4" /><text x="14" y="78">Jaisalmer</text></g>
      </svg>
      <span className="map-caption"><span />{[origin, ...(cities ?? [])].filter(Boolean).join(" → ") || "Choose a route to map it"}</span>
    </div>
  )
}

// ─── Context Panel ───────────────────────────────────────────────────────────

function ContextPanel({ cities, mobile = false }: { cities?: string[]; mobile?: boolean }) {
  const [open, setOpen] = useState(false)

  const content = (
    <>
      <RouteMap cities={cities} />
      <div className="glance">
        <span className="eyebrow">Trip at a glance</span>
        <h2>{cities?.join(" · ") ?? "Rajasthan"}</h2>
        <div className="season-note">
          <Icon name="sun" />
          <span>
            <small>Seasonal note</small>
            Check current weather before departure
          </span>
        </div>
        <h3>Packing reminders</h3>
        {["Refillable water bottle", "Hat and sun protection", "ID for safari permit", "Comfortable walking shoes"].map((x) => (
          <p key={x}><Icon name="check" size={13} />{x}</p>
        ))}
      </div>
      <div className="panel-actions">
        <button className="save-button" id="save-plan-btn"><Icon name="bookmark" /> Save plan</button>
        <button className="export-button" id="export-btn"><Icon name="download" /> Export itinerary</button>
      </div>
    </>
  )

  if (mobile) {
    return (
      <section className={`mobile-details ${open ? "open" : ""}`}>
        <button onClick={() => setOpen(!open)}>
          <span><Icon name="route" /> Trip details</span>
          <Icon name="chevron" className={open ? "open" : ""} />
        </button>
        {open && <div className="mobile-details-body">{content}</div>}
      </section>
    )
  }

  return <aside className="context-panel">{content}</aside>
}

// ─── Top Bar ─────────────────────────────────────────────────────────────────

function TopBar({
  title, cities, onMenuOpen, lang
}: {
  title: string
  cities?: string[]
  onMenuOpen: () => void
  lang: Lang
}) {
  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onMenuOpen} aria-label="Open menu" id="open-menu-btn">
        <Icon name="menu" />
      </button>
      <div className="topbar-title">
        <span>{title}</span>
      </div>
      <div className="topbar-chips">
        <button className="topbar-chip share" id="share-itinerary-btn">
          <Icon name="share" size={13} />
          {tr("shareItin", lang)}
        </button>
        <button className="topbar-chip export" id="export-itinerary-btn">
          <Icon name="export" size={13} />
          {tr("exportPDF", lang)}
        </button>
        <button className="topbar-chip cab" id="book-cab-btn">
          <Icon name="cab" size={13} />
          {tr("bookCab", lang)}
        </button>
      </div>
      <div className="topbar-actions">
        <button aria-label="Share plan" id="share-btn"><Icon name="share" /></button>
        <button aria-label="More options" id="more-btn"><Icon name="more" /></button>
      </div>
    </header>
  )
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [lang, setLang] = useState<Lang>(() => localStorage.getItem("rajasthan-language") === "hi" ? "hi" : "en")
  const [theme, setTheme] = useState<Theme>("light")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [journeys, setJourneys] = useState<Journey[]>(() => {
    const fresh = createJourney(lang)
    try {
      const stored = JSON.parse(localStorage.getItem("rajasthan-journeys") ?? "[]") as Journey[]
      return [fresh, ...stored.filter((j) => Array.isArray(j.messages) && j.messages.some((message) => message.role === "user")).map(restoreJourney).sort((a, b) => b.lastActive.getTime() - a.lastActive.getTime())]
    } catch { return [fresh] }
  })
  const [activeJourneyId, setActiveJourneyId] = useState(() => journeys[0]?.id ?? "")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [savedPlaces, setSavedPlaces] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem("rajasthan-saved-places") ?? "[]") as string[] } catch { return [] }
  })
  const [apiStatus, setApiStatus] = useState<{ llm_enabled?: boolean; llm_provider?: string; places_enabled?: boolean; gemini_fallback_enabled?: boolean; gemini_fallback_model?: string; groq_fallback_enabled?: boolean; groq_fallback_model?: string; railradar_enabled?: boolean } | null>(null)
  const activeJourney = journeys.find((j) => j.id === activeJourneyId) ?? journeys[0]
  const messages = activeJourney?.messages ?? []
  const context: Record<string, unknown> = { ...(activeJourney?.context ?? {}), ...(activeJourney?.routeCities?.length ? { cities: activeJourney.routeCities } : activeJourney?.destination ? { cities: [activeJourney.destination] } : {}), ...(activeJourney?.origin ? { origin_city: activeJourney.origin } : {}), response_language: lang === "hi" ? "hindi_devanagari" : "english", conversation_history: messages.slice(-6).map(({ role, text }) => ({ role, content: text.slice(0, 600) })) }

  useEffect(() => { localStorage.setItem("rajasthan-journeys", JSON.stringify(journeys.filter((journey) => journey.messages.some((message) => message.role === "user")))) }, [journeys])
  useEffect(() => { localStorage.setItem("rajasthan-saved-places", JSON.stringify(savedPlaces)) }, [savedPlaces])
  useEffect(() => { localStorage.setItem("rajasthan-language", lang) }, [lang])
  useEffect(() => { fetch("/api/health").then((r) => readJsonResponse<{ llm_enabled?: boolean; llm_provider?: string; places_enabled?: boolean; gemini_fallback_enabled?: boolean; gemini_fallback_model?: string }>(r)).then(setApiStatus).catch(() => setApiStatus(null)) }, [])

  // Apply dark mode
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
  }, [theme])

  const lastPlan = messages.filter((m) => m.role === "assistant" && m.result?.plan).at(-1)?.result?.plan
  const cities = lastPlan?.cities ?? context.cities as string[] | undefined
  const chatHistory = journeys.filter((journey) => journey.messages.some((message) => message.role === "user")).sort((a, b) => b.lastActive.getTime() - a.lastActive.getTime())

  const appendMessage = (message: ChatMessage) => setJourneys((prev) => prev.map((j) => j.id === activeJourneyId ? { ...j, messages: [...j.messages, message], lastActive: new Date() } : j))

  const submit = useCallback(async (text: string) => {
    if (!text.trim() || loading) return
    const trimmed = text.trim()
    const userMsg: ChatMessage = { id: "u-" + Date.now(), role: "user", text: trimmed, timestamp: new Date() }
    setLoading(true)
    setError("")
    appendMessage(userMsg)

    try {
      const requestedDays = inferTripDays(text) ?? (context.days as number | undefined) ?? inferTripDays(activeJourney?.initialPrompt ?? "")
      const requestedTravelers = inferTravelers(text) ?? (context.travelers as number | undefined)
      const requestedBudget = inferBudget(text) ?? (context.budget as number | undefined)
      const requestedDates = inferTravelDates(text) ?? inferTravelDates(activeJourney?.initialPrompt ?? "")
      const requestedTravelDate = inferTravelDate(text) ?? inferTravelDate(activeJourney?.initialPrompt ?? "")
      const requestContext: Record<string, unknown> = {
        ...context,
        ...(requestedDays ? { days: requestedDays } : {}),
        ...(requestedTravelers ? { travelers: requestedTravelers } : {}),
        ...(requestedBudget ? { budget: requestedBudget } : {}),
        ...(requestedDates ?? {}),
        ...(requestedTravelDate ? { travel_date: requestedTravelDate } : {}),
      }
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Question phrasing must not turn an explicit trip-planning request
        // into a prose-only answer (e.g. “Can you plan 5 days in Jaipur?”).
        body: JSON.stringify({ message: trimmed, context: { ...requestContext, question_only: isQuestionLike(trimmed) && !hasTravelTask(trimmed) } }),
      })
      const data = await readJsonResponse<AgentResult & { error?: string }>(res)
      if (!res.ok) throw new Error(data.error ?? "The planner is unavailable.")

      const assistantMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        text: data.answer,
        result: data,
        timestamp: new Date(),
      }
      const savedContext = { ...(data.context ?? requestContext) }
      delete savedContext.question_only
      delete savedContext.conversation_history
      const origin = extractOrigin(trimmed)
      const routeCities = extractRouteCities(trimmed)
      setJourneys((prev) => prev.map((j) => {
        if (j.id !== activeJourneyId) return j
        const nextTitle = [origin ?? j.origin, ...(routeCities.length ? routeCities : (j.routeCities ?? []))]
          .filter(Boolean)
          .join(" → ")
        return {
          ...j, stage: "planning", context: savedContext,
          ...(origin ? { origin } : {}),
          ...(routeCities.length ? { routeCities, destination: routeCities.join(" → ") } : {}),
          // Only name an untitled trip once real route information exists.
          ...(nextTitle && !/^New Rajasthan trip$/.test(j.title) ? { title: nextTitle } : {}),
          lastActive: new Date(), messages: [...j.messages, assistantMsg],
        }
      }))
    } catch (err) {
      const errText = err instanceof Error ? err.message : "The planner is unavailable."
      setError(errText)
      const errMsg: ChatMessage = {
        id: `a-err-${Date.now()}`,
        role: "assistant",
        text: `Unable to reach the planner: ${errText}`,
        timestamp: new Date(),
      }
      appendMessage(errMsg)
    } finally {
      setLoading(false)
    }
  }, [loading, activeJourney, activeJourneyId, lang, context])

  const handleNewJourney = () => {
    const journey = createJourney(lang)
    setJourneys((prev) => [journey, ...prev])
    setActiveJourneyId(journey.id)
    setError("")
  }

  const openSavedPlace = (place: string) => submit(`Tell me about ${place} and include it in my trip if it fits.`)
  const savePlace = (place: string) => setSavedPlaces((prev) => prev.includes(place) ? prev : [...prev, place])

  const threadTitle = activeJourney?.title ?? (messages.length > 0
    ? (messages[0].text.slice(0, 48) + (messages[0].text.length > 48 ? "…" : ""))
    : "Rajasthan trip planning")

  return (
    <div className="app-shell">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        lang={lang}
        onLangChange={setLang}
        theme={theme}
        onThemeChange={setTheme}
        activeJourneyId={activeJourneyId}
        journeys={chatHistory}
        savedPlaces={savedPlaces}
        cities={cities}
        origin={activeJourney?.origin}
        apiStatus={apiStatus}
        onSavedPlace={openSavedPlace}
        onJourneySelect={setActiveJourneyId}
        onNewJourney={handleNewJourney}
      />

      <section className="workspace">
        <TopBar
          title={threadTitle}
          cities={cities}
          onMenuOpen={() => setSidebarOpen(true)}
          lang={lang}
        />

        <div className="main-grid">
          <ChatCanvas
            messages={messages}
            loading={loading}
            lang={lang}
            onChipClick={submit}
            onSavePlace={savePlace}
            onAsk={submit}
          />
          <ContextPanel cities={cities} />
        </div>

        <Composer
          onSubmit={submit}
          loading={loading}
          lang={lang}
          hasMessages={messages.length > 0}
        />
      </section>
    </div>
  )
}
