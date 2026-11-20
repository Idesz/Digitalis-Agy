"""Fetch and normalize public AI leaderboard data for the dashboard."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple


MAX_SOURCE_BYTES = 8 * 1024 * 1024
SOURCE_TIMEOUT_SECONDS = 12
MAX_ROWS = 50

RANKING_TYPES = [
    {
        "id": "benchlm",
        "name": "BenchLM",
        "metric": "BenchAlign v5 (0-100)",
        "source_url": "https://benchlm.ai/data",
        "fetch_url": "https://benchlm.ai/data/leaderboard.json",
        "source_label": "BenchLM.ai",
        "attribution": "Data from BenchLM.ai. CC BY-NC 4.0; non-commercial use only.",
        "adapter": "benchlm_json",
    },
    {
        "id": "epoch_ai",
        "name": "Epoch AI",
        "metric": "Epoch Capabilities Index (ECI)",
        "source_url": "https://epoch.ai/eci",
        "fetch_url": "https://epoch.ai/data/eci_scores.csv",
        "source_label": "Epoch AI ECI",
        "attribution": "Data from Epoch AI. CC BY 4.0.",
        "adapter": "epoch_csv",
    },
    {
        "id": "lmsys_arena_elo",
        "name": "LMSYS Chatbot Arena - Text",
        "metric": "Arena rating / human preference (Elo)",
        "source_url": "https://arena.ai/leaderboard/chat/text",
        "fetch_url": "https://arena.ai/leaderboard/chat/text",
        "source_label": "Arena Text leaderboard",
        "attribution": "Official Arena Text leaderboard; fetched from the public page.",
        "adapter": "arena_html",
    },
]


class LeaderboardFetchError(RuntimeError):
    """A public leaderboard could not be fetched or parsed."""


def _download(url: str, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "Digitalis-Agy dashboard/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=SOURCE_TIMEOUT_SECONDS) as response:
            payload = response.read(MAX_SOURCE_BYTES + 1)
    except (urllib.error.URLError, OSError) as exc:
        status = getattr(exc, "code", None)
        detail = f"HTTP {status}" if status else str(exc)
        raise LeaderboardFetchError(
            f"A rangsorforrás nem érhető el ({detail})."
        ) from exc

    if len(payload) > MAX_SOURCE_BYTES:
        raise LeaderboardFetchError("A rangsorforrás válasza túllépi a méretkorlátot.")
    return payload


_ARENA_ROW = re.compile(
    r"^(?P<model>.+): (?P<score>[\d,]+(?:\.\d+)?), "
    r"95% CI (?P<lower>[\d,]+(?:\.\d+)?) to (?P<upper>[\d,]+(?:\.\d+)?)$"
)
_ARENA_DATE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2},\s+\d{4}\b"
)


class _ArenaPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[Dict[str, Any]] = []
        self.text: List[str] = []
        self._models = set()

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "li":
            return
        label = dict(attrs).get("aria-label") or ""
        match = _ARENA_ROW.fullmatch(label.strip())
        if not match:
            return
        model = match.group("model").strip()
        if not model or model in self._models:
            return
        self._models.add(model)
        self.rows.append({
            "rank": len(self.rows) + 1,
            "model": model,
            "score": (
                f'{match.group("score")} '
                f'(95% CI {match.group("lower")} to {match.group("upper")})'
            ),
        })

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _parse_arena_html(payload: bytes) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        html = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LeaderboardFetchError("Az Arena oldal kódolása nem UTF-8.") from exc

    parser = _ArenaPageParser()
    parser.feed(html)
    if not parser.rows:
        raise LeaderboardFetchError(
            "Az Arena oldal betöltődött, de nem találtam rangsorsorokat; "
            "a nyilvános oldal szerkezete megváltozhatott."
        )

    updated = _ARENA_DATE.search(" ".join(parser.text))
    updated_at = updated.group(0) if updated else None
    for row in parser.rows:
        row["updated_at"] = updated_at
    return parser.rows[:MAX_ROWS], updated_at


def _parse_benchlm_json(payload: bytes) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeaderboardFetchError("A BenchLM JSON-válasza nem olvasható.") from exc

    if not isinstance(data, dict):
        raise LeaderboardFetchError("A BenchLM rangsor válasza nem JSON-objektum.")
    if data.get("license") != "CC BY-NC 4.0":
        raise LeaderboardFetchError(
            "A BenchLM adatlicence megváltozott; frissítés előtt ellenőrzés szükséges."
        )
    items = data.get("items")
    if not isinstance(items, list):
        raise LeaderboardFetchError("A BenchLM rangsor formátuma megváltozott.")

    rows = []
    for item in items:
        if not isinstance(item, dict) or item.get("rankingEligible") is False:
            continue
        model = item.get("model")
        rank = item.get("rank")
        score = item.get("displayScore", item.get("score"))
        if not model or rank is None or score is None:
            continue
        try:
            numeric_score = float(score)
            numeric_rank = int(rank)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_score):
            continue
        score_text = f"{numeric_score:g}"
        interval = item.get("scoreInterval90")
        if (
            isinstance(interval, dict)
            and interval.get("lower") is not None
            and interval.get("upper") is not None
        ):
            score_text += f" (90% CI {interval['lower']}-{interval['upper']})"
        evidence = item.get("evidenceStatus")
        if evidence:
            score_text += f" [{evidence}]"
        rows.append({
            "rank": numeric_rank,
            "model": str(model),
            "score": score_text,
            "updated_at": data.get("sourceLastUpdated"),
        })

    if not rows:
        raise LeaderboardFetchError("A BenchLM válasza nem tartalmaz rangsorolható modelleket.")
    rows.sort(key=lambda row: row["rank"])
    return rows[:MAX_ROWS], data.get("sourceLastUpdated")


def _parse_epoch_csv(payload: bytes) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or not {"Model", "eci", "date"}.issubset(reader.fieldnames):
            raise LeaderboardFetchError("Az Epoch AI ECI CSV-oszlopai megváltoztak.")

        candidates = []
        for item in reader:
            model = (item.get("Display name") or item.get("Model") or "").strip()
            try:
                score = float(item.get("eci", ""))
            except (TypeError, ValueError):
                continue
            if not model or not math.isfinite(score):
                continue
            low = item.get("eci_ci_low", "").strip()
            high = item.get("eci_ci_high", "").strip()
            score_text = f"{score:.2f}"
            if low and high:
                score_text += f" (CI {low}-{high})"
            candidates.append({
                "model": model,
                "numeric_score": score,
                "score": score_text,
                "updated_at": item.get("date") or None,
            })
    except (UnicodeDecodeError, csv.Error) as exc:
        raise LeaderboardFetchError("Az Epoch AI ECI CSV-válasza nem olvasható.") from exc

    if not candidates:
        raise LeaderboardFetchError("Az Epoch AI ECI CSV nem tartalmaz rangsorolható modelleket.")
    candidates.sort(key=lambda row: row["numeric_score"], reverse=True)
    rows = [
        {
            "rank": index,
            "model": item["model"],
            "score": item["score"],
            "updated_at": item["updated_at"],
        }
        for index, item in enumerate(candidates[:MAX_ROWS], start=1)
    ]
    latest = max(
        (row["updated_at"] for row in rows if row["updated_at"]),
        default=None,
    )
    return rows, latest


def fetch_leaderboard(ranking_id: str) -> Dict[str, Any]:
    ranking = next((item for item in RANKING_TYPES if item["id"] == ranking_id), None)
    if ranking is None:
        raise ValueError(f"Unknown ranking type: {ranking_id}")

    adapter = ranking["adapter"]
    if adapter == "arena_html":
        payload = _download(ranking["fetch_url"], "text/html")
        rows, source_updated = _parse_arena_html(payload)
    elif adapter == "benchlm_json":
        payload = _download(ranking["fetch_url"], "application/json")
        rows, source_updated = _parse_benchlm_json(payload)
    elif adapter == "epoch_csv":
        payload = _download(ranking["fetch_url"], "text/csv")
        rows, source_updated = _parse_epoch_csv(payload)
    else:
        raise LeaderboardFetchError(f"Nincs adapter ehhez a rangsorhoz: {ranking_id}")

    return {
        "type": ranking_id,
        "name": ranking["name"],
        "metric": ranking["metric"],
        "status": "ready",
        "rows": rows,
        "message": "Még nincs azonos mérőszámon mért, összehasonlítható Digitalis-Agy eredmény.",
        "source_url": ranking["source_url"],
        "source_label": ranking["source_label"],
        "attribution": ranking["attribution"],
        "source_last_updated": source_updated,
    }
