"""
Equasis vessel lookup — authenticates with credentials from .env and provides:
  - search(term): search by vessel name, returns rows with ship type
  - search_by_flag(flag_code, page): search all vessels for a flag state
  - search_by_type_category(cat_code, page): search by ship type category globally
  - find_all_bunkering_tankers_global(): page through Other Tankers to collect all Bunkering Tankers
  - lookup_imo(imo): fetch full ship type/flag/GT for a specific IMO
"""
from __future__ import annotations

import os
import re
import threading
import time

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

EQUASIS_EMAIL    = os.getenv("EQUASIS_EMAIL", "")
EQUASIS_PASSWORD = os.getenv("EQUASIS_PASSWORD", "")

_BASE    = "https://www.equasis.org/EquasisWeb"
_LOGIN   = f"{_BASE}/authen/HomePage?fs=HomePage"
_SEARCH  = f"{_BASE}/restricted/Search?fs=Search"
_SHIP    = f"{_BASE}/restricted/ShipInfo?fs=Search"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Origin": "https://www.equasis.org",
}

_session: requests.Session | None = None
_session_lock = threading.Lock()
_last_req = 0.0
_MIN_GAP  = 8.0   # conservative rate-limit — Equasis has strict daily page-view quotas


def _rate_limit() -> None:
    global _last_req
    gap = time.monotonic() - _last_req
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)
    _last_req = time.monotonic()


def _new_session() -> requests.Session | None:
    if not EQUASIS_EMAIL or not EQUASIS_PASSWORD:
        logger.warning("EQUASIS_EMAIL/PASSWORD not set — Equasis disabled")
        return None
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        _rate_limit()
        r = s.post(
            _LOGIN,
            data={"j_email": EQUASIS_EMAIL, "j_password": EQUASIS_PASSWORD, "submit": "Login"},
            headers={**_HEADERS, "Referer": f"{_BASE}/public/HomePage",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        # Login success: JSESSIONID cookie is set and we see the user's name
        if "JSESSIONID" in s.cookies and "logout" in r.text.lower():
            logger.info("Equasis session established")
            return s
        logger.error(f"Equasis login failed (status={r.status_code})")
        return None
    except Exception as e:
        logger.error(f"Equasis login error: {e}")
        return None


def _get_session() -> requests.Session | None:
    global _session
    with _session_lock:
        if _session is None:
            _session = _new_session()
        return _session


def _reset_session() -> None:
    global _session
    with _session_lock:
        _session = None


def _clean(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', s)).strip()


def _parse_search_rows(html: str) -> list[dict]:
    """
    Parse vessel result rows from an Equasis search response.
    Row structure: <th>IMO</th> <td>Name</td> <td>GT</td> <td>Type</td> <td>Year</td> <td>Flag</td>
    Deduplicates by IMO (table appears twice for desktop/mobile).
    """
    seen: set[str] = set()
    results: list[dict] = []

    for tbody_m in re.finditer(r'<tbody>(.*?)</tbody>', html, re.DOTALL):
        for row in re.finditer(r'<tr[^>]*>(.*?)</tr>', tbody_m.group(1), re.DOTALL):
            row_html = row.group(1)
            imo_m = re.search(r"P_IMO\.value='(\d+)'", row_html)
            if not imo_m:
                continue
            imo = imo_m.group(1)
            if imo in seen:
                continue
            seen.add(imo)

            # Extract all <td> cells (not <th>)
            tds = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
            cells = [_clean(td) for td in tds]

            name      = cells[0] if len(cells) > 0 else ""
            gt_str    = cells[1] if len(cells) > 1 else ""
            ship_type = cells[2] if len(cells) > 2 else ""
            year      = cells[3] if len(cells) > 3 else ""
            flag_raw  = cells[4] if len(cells) > 4 else ""

            # Flag: "Spain (ESP)" → "Spain"
            flag = re.sub(r'\s*\([A-Z]+\)\s*$', '', flag_raw).strip()
            flag = flag.replace("Not Known", "").strip()

            try:
                gt: int | None = int(gt_str.replace(",", "")) if gt_str.isdigit() else None
            except (ValueError, AttributeError):
                gt = None

            results.append({
                "imo": imo, "name": name,
                "ship_type": ship_type, "flag": flag,
                "gt": gt, "year": year,
            })
    return results


# Equasis numeric flag codes — extracted from the P_FLAG select dropdown.
# Real shipping registries at the top for quick reference.
FLAG_CODES: dict[str, str] = {
    "Panama": "0485", "Liberia": "0375", "Marshall Islands": "0404",
    "Bahamas": "0045", "Singapore": "0570", "Hong Kong, China": "0275",
    "Malta": "0400", "Cayman Islands": "0115", "Cyprus": "0150",
    "Greece": "0245", "China Peoples's Republic": "0125", "Japan": "0335",
    "Norway": "0470", "Norway (NIS)": "0471", "United Kingdom": "0670",
    "United States of America": "0675", "Russia": "0532", "Turkey": "0655",
    "India": "0290", "South Korea": "0360", "Korea (Republic of)": "0360",
    "Indonesia": "0295", "Philippines": "0505", "Tuvalu": "0663",
    "Palau (Republic of)": "0482", "Vanuatu": "0685", "Cook islands": "0137",
    "Afghanistan": "0003", "Albania": "0005", "Algeria": "0010",
    "Andorra": "0012", "Angola": "0015", "Anguilla": "0020",
    "Antigua and Barbuda": "0028", "Antilles, Netherlands": "CM01",
    "Argentina": "0030", "Armenia": "0031", "Aruba": "0033",
    "Australia": "0035", "Austria": "0040", "Azerbaijan": "0043",
    "Bahrain": "0050", "Bangladesh": "0055", "Barbados": "0060",
    "Belarus": "0064", "Belgium": "0065", "Belize": "0067",
    "Benin": "0070", "Bermuda": "0075", "Bhutan": "0078",
    "Bolivia": "0080", "Bosnia Herzegovina": "0082", "Brazil": "0085",
    "Brunei Darussalam": "0088", "Bulgaria": "0090", "Cambodia": "0098",
    "Cameroon": "0100", "Canada": "0105", "Cape Verde": "0110",
    "Chile": "0120", "Colombia": "0130", "Comoros": "0135",
    "Croatia": "0144", "Cuba": "0145", "Curacao": "0025",
    "Denmark": "0165", "Djibouti": "0166", "Dominica": "0168",
    "Dominican Republic": "0170", "Ecuador": "0175", "Egypt": "0180",
    "El Salvador": "0183", "Equatorial Guinea": "0190", "Eritrea": "0186",
    "Estonia": "0184", "Ethiopia": "0185", "Falkland Islands": "0198",
    "Faroe Islands": "0195", "Fiji": "0200", "Finland": "0205",
    "France": "0210", "France (RIF)": "0211", "Gabon": "0215",
    "Gambia": "0220", "Georgia": "0224", "Germany": "0230",
    "Ghana": "0235", "Gibraltar": "0240", "Greenland": "0765",
    "Grenada": "0250", "Guatemala": "0255", "Guernsey": "0258",
    "Guinea": "0267", "Guyana": "0260", "Haiti": "0271",
    "Honduras": "0270", "Hungary": "0280", "Iceland": "0285",
    "Iran": "0300", "Iraq": "0305", "Ireland": "0310",
    "Isle of Man": "0403", "Israel": "0315", "Italy": "0320",
    "Ivory Coast": "0325", "Jamaica": "0330", "Jersey": "0337",
    "Jordan": "0340", "Kazakhstan": "0344", "Kenya": "0345",
    "Kiribati": "0350", "Korea Democratic Republic": "0355",
    "Kuwait": "0365", "Latvia": "0367", "Lebanon": "0370",
    "Libya": "0380", "Lithuania": "0382", "Luxembourg": "0383",
    "Malaysia": "0390", "Maldives": "0395", "Mauritania": "0405",
    "Mauritius": "0410", "Mexico": "0415", "Moldova": "0419",
    "Monaco": "0420", "Mongolia": "0422", "Montenegro": "0424",
    "Morocco": "0430", "Mozambique": "0435", "Myanmar union of": "0095",
    "Namibia": "0439", "Netherlands": "0445", "New Zealand": "0450",
    "Nicaragua": "0455", "Nigeria": "0465", "Niue": "0467",
    "Oman": "0475", "Pakistan": "0480", "Papua New Guinea": "0490",
    "Paraguay": "0495", "Peru": "0500", "Poland": "0510",
    "Portugal": "0520", "Qatar": "0525", "Romania": "0530",
    "Saudi Arabia": "0555", "Senegal": "0560", "Serbia": "0562",
    "Seychelles": "0565", "Sierra leone": "0567", "Slovakia": "0572",
    "Slovenia": "0573", "Solomon Islands": "0575", "Somalia": "0580",
    "South Africa": "0585", "Spain": "0590", "Sri lanka": "0595",
    "St Vincent and Grenadines": "0540", "Sudan": "0600",
    "Suriname": "0605", "Sweden": "0610", "Switzerland": "0615",
    "Syrian Arab Republic": "0620", "Taiwan": "0625", "Tanzania": "0630",
    "Thailand": "0635", "Togo": "0640", "Tonga": "0642",
    "Trinidad and Tobago": "0645", "Tunisia": "0650",
    "Turkmenistan": "0656", "Turks Caicos Islands": "0660",
    "Uganda": "0735", "Ukraine": "0664", "United Arab Emirates": "0666",
    "Uruguay": "0680", "Uzbekistan": "0682", "Venezuela": "0690",
    "Vietnam": "0695", "Yemen": "0715", "Zambia": "0810", "Zimbabwe": "0820",
}

# Ship type category codes from P_CatTypeShip select
CAT_TYPE_CODES: dict[str, str] = {
    "Other Tankers": "8",          # includes Bunkering Tanker, Water Tanker, Asphalt, etc.
    "Oil and Chemical Tankers": "6",
    "Service Ships": "11",
    "Gas Tankers": "7",
    "Bulk Carriers": "5",
    "Container Ships": "3",
    "General Cargo Ships": "1",
    "Offshore Vessels": "10",
    "Other ships": "13",
    "Passenger Ships": "9",
    "Ro-Ro Cargo Ships": "4",
    "Specialized Cargo Ships": "2",
    "Tugs": "12",
    "Fishing vessels": "14",
}


def _post_search(data: dict, referer: str | None = None) -> str | None:
    """Rate-limited POST to Equasis search. Returns HTML or None on failure."""
    s = _get_session()
    if s is None:
        return None
    try:
        _rate_limit()
        r = s.post(
            _SEARCH,
            data=data,
            headers={**_HEADERS,
                     "Referer": referer or f"{_BASE}/authen/HomePage?fs=HomePage",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        if r.status_code == 401 or "CtrlGeneralError" in r.url:
            _reset_session()
            return None
        if "consultation rate limit" in r.text.lower() or "session has now been disconnected" in r.text.lower():
            logger.warning("Equasis rate limit hit — backing off")
            _reset_session()
            return None
        return r.text
    except Exception as e:
        logger.debug(f"Equasis POST error: {e}")
        return None


def _post_advanced_search(page: int = 1,
                           flag_code: str = "",
                           cat_type: str = "",
                           status: str = "S") -> str | None:
    """
    Submit the Equasis advanced search form with criteria filters.
    flag_code: Equasis numeric code from FLAG_CODES (e.g. '0485' for Panama).
    cat_type: P_CatTypeShip code (e.g. '8' = Other Tankers).
    status: 'S' = In Service/Commission (active vessels only).
    """
    flag_rb = "CM" if flag_code else "HC"
    cat_rb  = "CM" if cat_type  else "HC"
    return _post_search({
        "P_ENTREE": "", "P_ENTREE_HOME_HIDDEN": "",
        "P_PAGE": page, "P_PAGE_COMP": page, "P_PAGE_SHIP": page,
        "ongletActifSC": "ship",
        "Submit": "Advanced Search",
        "buttonAdvancedSearch": "advancedOk",  # JS sets this on form submit
        "P_IMO": "", "P_NAME": "", "P_CALLSIGN": "", "P_MMSI": "",
        "P_CLASS_rb": "HC", "P_CLASS": "",
        "P_CLASS_ST_rb": "HC", "P_CLASS_ST": "",
        "P_FLAG_rb": flag_rb, "P_FLAG": flag_code,
        "P_CatTypeShip_rb": cat_rb, "P_CatTypeShip": cat_type,
        "P_STATUS": status,
    }, referer=_SEARCH)


def _has_more_pages(rows: list) -> bool:
    """Equasis returns exactly 100 rows per page until the last page."""
    return len(rows) >= 100


def search(term: str, referer: str | None = None) -> list[dict]:
    """Search Equasis by vessel name/term. Returns list of {imo, name, ship_type, flag, gt, year}."""
    html = _post_search({
        "P_ENTREE": term,
        "P_ENTREE_HOME_HIDDEN": term,
        "P_PAGE": 1, "P_PAGE_COMP": 1, "P_PAGE_SHIP": 1,
        "ongletActifSC": "ship",
        "Submit": "Advanced Search",
        "buttonAdvancedSearch": "",
        "P_IMO": "", "P_NAME": "", "P_CALLSIGN": "", "P_MMSI": "",
    }, referer=referer)
    return _parse_search_rows(html) if html else []


def search_by_flag(flag_name_or_code: str, page: int = 1) -> tuple[list[dict], bool]:
    """
    Search Equasis for vessels registered under a flag state.
    flag_name_or_code: country name (e.g. 'Panama') or Equasis numeric code (e.g. '0485').
    Returns (results, has_more) — has_more=True if there are likely more pages.
    """
    numeric = FLAG_CODES.get(flag_name_or_code, flag_name_or_code)
    html = _post_advanced_search(page=page, flag_code=numeric)
    if not html:
        return [], False
    results = _parse_search_rows(html)
    return results, _has_more_pages(results)


def search_by_type_category(cat_code: str, page: int = 1, status: str = "S") -> tuple[list[dict], bool]:
    """
    Search Equasis globally by ship type category.
    cat_code: '8' = Other Tankers (contains Bunkering Tanker), '6' = Oil and Chemical Tankers.
    Returns (results, has_more).
    """
    html = _post_advanced_search(page=page, cat_type=cat_code, status=status)
    if not html:
        return [], False
    results = _parse_search_rows(html)
    return results, _has_more_pages(results)


def find_bunkering_tankers_by_flag(flag_name_or_code: str, max_pages: int = 100) -> list[dict]:
    """
    Scan all pages for a flag state and return vessels with 'bunker' in ship_type.
    Reads search listings only — no individual vessel page visits required.
    """
    found: list[dict] = []

    def _is_bunkering(row: dict) -> bool:
        return "bunker" in row.get("ship_type", "").lower()

    for page in range(1, max_pages + 1):
        results, has_more = search_by_flag(flag_name_or_code, page=page)
        if not results:
            break
        new = [r for r in results if _is_bunkering(r)]
        found.extend(new)
        logger.debug(f"Flag {flag_name_or_code!r} p{page}: {len(results)} vessels, {len(new)} bunkering")
        if not has_more:
            break

    logger.info(f"Flag {flag_name_or_code!r}: {len(found)} Bunkering Tankers found")
    return found


def find_all_bunkering_tankers_global(
    cat_codes: list | None = None,
    max_pages: int = 200,
    progress_cb=None,
) -> list[dict]:
    """
    Global scan: page through ship type categories to collect all Bunkering Tankers worldwide.
    cat_codes: list of P_CatTypeShip values. Defaults to ['8'] (Other Tankers).
    progress_cb: optional callable(page, results_this_page, total_found) for live updates.
    Returns list of {imo, name, ship_type, flag, gt, year}.
    """
    if cat_codes is None:
        cat_codes = ["8"]

    def _is_bunkering(row: dict) -> bool:
        return "bunker" in row.get("ship_type", "").lower()

    found: list[dict] = []
    seen_imos: set[str] = set()

    for cat_code in cat_codes:
        logger.info(f"Global scan: starting category {cat_code!r}")
        for page in range(1, max_pages + 1):
            results, has_more = search_by_type_category(cat_code, page=page)
            if not results:
                logger.info(f"Category {cat_code!r}: finished at page {page}")
                break

            for row in results:
                if _is_bunkering(row) and row.get("imo") and row["imo"] not in seen_imos:
                    found.append(row)
                    seen_imos.add(row["imo"])

            logger.info(
                f"Category {cat_code!r} p{page}: {len(results)} vessels, "
                f"{sum(1 for r in results if _is_bunkering(r))} bunkering, "
                f"total found: {len(found)}"
            )

            if progress_cb:
                try:
                    progress_cb(page, results, len(found))
                except Exception:
                    pass

            if not has_more:
                logger.info(f"Category {cat_code!r}: completed at page {page}")
                break

    logger.info(f"Global scan done: {len(found)} unique Bunkering Tankers")
    return found


def lookup_imo(imo: str) -> dict | None:
    """
    Look up a vessel by IMO on Equasis and return {ship_type, flag, gt, name, imo}.
    Requires a prior search to establish the correct session state.
    """
    s = _get_session()
    if s is None:
        return None
    # Ensure we're in Search state
    try:
        _rate_limit()
        r = s.post(
            _SHIP,
            data={"P_IMO": imo},
            headers={**_HEADERS,
                     "Referer": _SEARCH,
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if r.status_code != 200 or "CtrlGeneralError" in r.text[:5000]:
            return None
        if "consultation rate limit" in r.text.lower() or "session has now been disconnected" in r.text.lower():
            logger.warning("Equasis consultation rate limit hit — backing off")
            _reset_session()
            return None
        html = r.text

        # Extract ship name from <b>VESSEL NAME</b>
        name_m = re.search(r'<b>([A-Z][A-Z0-9 \-\.]+)</b>', html)
        name = name_m.group(1).strip() if name_m else ""

        # Ship type: <b>Type of ship  </b> ... next div content
        type_m = re.search(
            r'Type of ship\s*</b>\s*</div>\s*<div[^>]*>\s*([^\n<]{3,60})',
            html, re.IGNORECASE
        )
        ship_type = type_m.group(1).strip() if type_m else ""

        # Flag
        flag_m = re.search(
            r'Flag\s*</b>\s*</div>\s*<div[^>]*>\s*([^\n<]{3,40})',
            html, re.IGNORECASE
        )
        flag = flag_m.group(1).strip() if flag_m else ""

        # Gross tonnage
        gt_m = re.search(
            r'Gross tonnage\s*</b>\s*</div>\s*<div[^>]*>\s*([\d\s,]+)',
            html, re.IGNORECASE
        )
        gt = None
        if gt_m:
            try:
                gt = int(gt_m.group(1).replace(",", "").replace(" ", "").strip())
            except ValueError:
                pass

        return {"imo": imo, "name": name, "ship_type": ship_type, "flag": flag, "gt": gt}

    except Exception as e:
        logger.debug(f"Equasis lookup_imo({imo}) error: {e}")
        return None


def classify_ship_type(ship_type: str) -> dict:
    """Map an Equasis ship type to a bunker confidence level."""
    st = ship_type.lower()
    if "bunker" in st:
        return {"level": "confirmed", "label": "Bunkering Tanker"}
    if st in ("tanker", "oil tanker", "oil products tanker", "oil/chemical tanker"):
        return {"level": "likely", "label": ship_type}
    if "tanker" in st or "oil" in st:
        return {"level": "possible", "label": ship_type}
    if not ship_type or ship_type in ("", "Unknown"):
        return {"level": "unknown", "label": "—"}
    return {"level": "not_tanker", "label": ship_type}


if __name__ == "__main__":
    import sys
    term = sys.argv[1] if len(sys.argv) > 1 else "bunker"
    print(f"Searching Equasis for: {term!r}")
    results = search(term)
    print(f"{len(results)} results:")
    for r in results:
        print(f"  IMO={r['imo']} | {r['name']:<30} | {r['ship_type']:<30} | {r['flag']} | {r['gt']} GT")
