from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import quote

from ....exceptions import LLM4FreeE
from .base import DuckDuckGoBase


class WeatherData(TypedDict):
    location: str
    current: Dict[str, Any]
    daily_forecast: List[Dict[str, Any]]
    hourly_forecast: List[Dict[str, Any]]


class DuckDuckGoWeather(DuckDuckGoBase):
    name = "duckduckgo"
    category = "weather"

    def run(self, *args, **kwargs) -> WeatherData:
        location = args[0] if args else kwargs.get("location")
        language = args[1] if len(args) > 1 else kwargs.get("language", "en")

        assert location, "location is mandatory"
        lang = language.split("-")[0]
        url = f"https://duckduckgo.com/js/spice/forecast/{quote(location)}/{lang}"

        resp = self._get_url("GET", url).content
        resp_text = resp.decode("utf-8")

        if "ddg_spice_forecast(" not in resp_text:
            raise LLM4FreeE(f"No weather data found for {location}")

        json_text = resp_text[resp_text.find("(") + 1 : resp_text.rfind(")")]
        try:
            result = json.loads(json_text)
        except Exception as e:
            raise LLM4FreeE(f"Error parsing weather JSON: {e}")

        if not result or "currentWeather" not in result or "forecastDaily" not in result:
            raise LLM4FreeE(f"Invalid weather data format for {location}")

        current = result.get("currentWeather") or {}
        metadata = current.get("metadata") or {}
        days = (result.get("forecastDaily") or {}).get("days", [])

        formatted_data: WeatherData = {
            "location": metadata.get("ddg-location", "Unknown"),
            "current": {
                "condition": current.get("conditionCode"),
                "temperature_c": current.get("temperature"),
                "feels_like_c": current.get("temperatureApparent"),
                "humidity": current.get("humidity"),
                "wind_speed_ms": current.get("windSpeed"),
                "wind_direction": current.get("windDirection"),
                "visibility_m": current.get("visibility"),
            },
            "daily_forecast": [],
            "hourly_forecast": [],
        }

        def _fmt(iso: str | None) -> str:
            if not iso:
                return ""
            try:
                return datetime.fromisoformat(
                    iso.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d" if "T" in iso else "%H:%M")
            except (ValueError, TypeError):
                return ""

        for day in days:
            if not isinstance(day, dict):
                continue
            formatted_data["daily_forecast"].append(
                {
                    "date": _fmt(day.get("forecastStart")),
                    "condition": (day.get("daytimeForecast") or {}).get("conditionCode"),
                    "max_temp_c": day.get("temperatureMax"),
                    "min_temp_c": day.get("temperatureMin"),
                    "sunrise": _fmt(day.get("sunrise")),
                    "sunset": _fmt(day.get("sunset")),
                }
            )

        hourly = (result.get("forecastHourly") or {}).get("hours", [])
        for hour in hourly:
            if not isinstance(hour, dict):
                continue
            formatted_data["hourly_forecast"].append(
                {
                    "time": _fmt(hour.get("forecastStart")),
                    "condition": hour.get("conditionCode"),
                    "temperature_c": hour.get("temperature"),
                    "feels_like_c": hour.get("temperatureApparent"),
                    "humidity": hour.get("humidity"),
                    "wind_speed_ms": hour.get("windSpeed"),
                    "wind_direction": hour.get("windDirection"),
                    "visibility_m": hour.get("visibility"),
                }
            )

        return formatted_data
