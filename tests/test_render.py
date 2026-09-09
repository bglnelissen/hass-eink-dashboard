from __future__ import annotations

import dataclasses
import datetime as dt
import io
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from custom_components.eink_dashboard.const import (
    COLOR_BLACK,
    COLOR_DARK_GRAY,
    COLOR_GRAY,
    PADDING,
)
from custom_components.eink_dashboard.render import (
    WidgetMetrics,
    _compute_metrics,
    _day_heading,
    _draw_card_container,
    _draw_card_row,
    _draw_chip,
    _draw_chip_flow,
    _fit_text,
    _format_relative_date,
    _is_emoji,
    _load_font,
    _parse_days_until,
    _parse_event_start,
    _split_runs,
    render_dashboard,
)
from tests.helpers import (
    assert_all_white,
    assert_has_dark_pixels,
    assert_has_gray_pixels,
    assert_scales_proportionally,
    assert_vertically_centered,
    content_bbox,
    make_config,
    pixel,
    png_to_image,
)

MOCK_WEATHER_STATE = {
    "weather.home": {
        "state": "sunny",
        "attributes": {
            "temperature": 22,
            "humidity": 58,
            "wind_speed": 12,
            "temperature_unit": "°C",
            "wind_speed_unit": "km/h",
            "pressure": 1013,
            "pressure_unit": "hPa",
            "cloud_coverage": 45,
            "precipitation_unit": "mm",
            "forecast": [
                {
                    "datetime": "2026-05-02T12:00:00",
                    "temperature": 24,
                    "templow": 16,
                    "condition": "sunny",
                    "precipitation": 0,
                },
                {
                    "datetime": "2026-05-03T12:00:00",
                    "temperature": 19,
                    "templow": 14,
                    "condition": "cloudy",
                    "precipitation": 5,
                },
                {
                    "datetime": "2026-05-04T12:00:00",
                    "temperature": 21,
                    "templow": 15,
                    "condition": "partlycloudy",
                    "precipitation": 0,
                },
            ],
        },
    },
}


class TestRenderDashboard:
    def test_empty_widget_list_returns_white_image(self) -> None:
        config = {"width": 100, "height": 100}
        result = render_dashboard([], config)

        img = png_to_image(result)
        assert img.mode == "L"
        assert img.size == (100, 100)
        assert pixel(img, 50, 50) == 255

    def test_returns_valid_png(self) -> None:
        config = {"width": 200, "height": 300}
        result = render_dashboard([], config)

        img = png_to_image(result)
        assert img.format == "PNG"
        assert img.size == (200, 300)

    def test_rotation_90(self) -> None:
        config = {"width": 200, "height": 100, "rotation": 90}
        result = render_dashboard([], config)

        img = png_to_image(result)
        assert img.size == (100, 200)

    def test_rotation_270(self) -> None:
        config = {"width": 200, "height": 100, "rotation": 270}
        result = render_dashboard([], config)

        img = png_to_image(result)
        assert img.size == (100, 200)

    def test_unknown_widget_type_is_skipped(self) -> None:
        config = {"width": 100, "height": 100}
        widgets = [{"type": "nonexistent", "x": 10, "y": 10}]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert img.size == (100, 100)


class TestRenderText:
    def test_text_draws_pixels(self) -> None:
        config = {"width": 200, "height": 100}
        widgets = [
            {
                "type": "text",
                "x": 10,
                "y": 10,
                "text": "Hello",
                "font_size": 20,
            }
        ]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert_has_dark_pixels(img, 10, 10, 100, 40)

    def test_text_right_align(self) -> None:
        config = {"width": 200, "height": 100}
        widgets = [
            {
                "type": "text",
                "x": 0,
                "y": 10,
                "text": "Hi",
                "font_size": 20,
                "align": "right",
            }
        ]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert_has_dark_pixels(img, 140, 10, 200, 40)

    def test_text_center_align(self) -> None:
        config = {"width": 200, "height": 100}
        widgets = [
            {
                "type": "text",
                "x": 0,
                "y": 10,
                "text": "Hi",
                "font_size": 20,
                "align": "center",
            }
        ]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert_has_dark_pixels(img, 80, 10, 120, 40)

    def test_text_custom_color(self) -> None:
        config = {"width": 200, "height": 100}
        widgets = [
            {
                "type": "text",
                "x": 10,
                "y": 10,
                "text": "Gray",
                "font_size": 20,
                "color": 160,
            }
        ]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert_has_gray_pixels(img, 10, 10, 80, 40, low=100, high=200)


class TestRenderLine:
    def test_horizontal_line(self) -> None:
        config = {"width": 100, "height": 100}
        widgets = [
            {
                "type": "line",
                "x": 10,
                "y": 50,
                "x2": 90,
                "y2": 50,
                "color": 0,
            }
        ]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert pixel(img, 50, 50) == 0
        assert pixel(img, 50, 10) == 255

    def test_line_custom_color(self) -> None:
        config = {"width": 100, "height": 100}
        widgets = [
            {
                "type": "line",
                "x": 0,
                "y": 50,
                "x2": 99,
                "y2": 50,
                "color": 160,
            }
        ]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert pixel(img, 50, 50) == 160


class TestRenderSeparator:
    def test_separator_spans_width(self) -> None:
        config = {"width": 200, "height": 100}
        widgets = [{"type": "separator", "y": 50, "color": 0}]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert pixel(img, PADDING, 50) == 0
        assert pixel(img, 175, 50) == 0
        # Outside separator range should be white
        assert pixel(img, 10, 50) == 255

    def test_separator_default_color(self) -> None:
        config = {"width": 200, "height": 100}
        widgets = [{"type": "separator", "y": 50}]
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        # Default is COLOR_LIGHT_GRAY = 180
        assert pixel(img, 100, 50) == 180


class TestRenderWeather:
    _DEFAULTS: dict[str, object] = {
        "width": 600,
        "height": 400,
        "states": MOCK_WEATHER_STATE,
    }

    def _config(self, **overrides: object) -> dict[str, object]:
        return make_config(self._DEFAULTS, **overrides)

    def test_weather_draws_temperature(self) -> None:
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING + 80, 10, 300, 70)

    def test_weather_draws_forecast(self) -> None:
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
                "forecast_days": 3,
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, 50, 110, 550, 200)

    def test_weather_missing_entity_is_noop(self) -> None:
        widgets = [
            {
                "type": "weather",
                "entity": "weather.nonexistent",
                "x": PADDING,
                "y": 10,
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_all_white(img, 0, 0, 600, 400)

    def test_weather_no_forecast(self) -> None:
        states = {
            "weather.home": {
                "state": "cloudy",
                "attributes": {
                    "temperature": 15,
                    "humidity": 70,
                    "wind_speed": 20,
                    "forecast": [],
                },
            }
        }
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
            }
        ]
        result = render_dashboard(
            widgets,
            {"width": 600, "height": 300, "states": states},
        )
        img = png_to_image(result)
        assert img.size == (600, 300)
        assert_has_dark_pixels(
            img, PADDING, 10, PADDING + 90, 100, threshold=200
        )
        assert_has_dark_pixels(img, PADDING + 80, 10, 300, 70)

    def test_weather_icon_sunny(self) -> None:
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(
            img, PADDING, 10, PADDING + 90, 100, threshold=200
        )

    def test_weather_landscape_layout(self) -> None:
        """Weather widget on a wide, short canvas (e.g. TRMNL OG 800x480)."""
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
                "forecast_days": 3,
            }
        ]
        config = self._config(width=800, height=480)
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        # Temperature drawn in the left area
        assert_has_dark_pixels(img, PADDING + 80, 10, 350, 70)
        # Detail chips row (humidity, pressure, wind, cloud)
        assert_has_dark_pixels(img, PADDING, 80, 400, 100)
        # Forecast section visible
        assert_has_dark_pixels(img, 50, 110, 750, 220)

    def test_weather_narrow_layout(self) -> None:
        """Weather widget on a narrow display still renders temp and
        details."""
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
            }
        ]
        config = self._config(width=350, height=250)
        result = render_dashboard(widgets, config)

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING + 80, 10, 300, 70)
        assert_has_dark_pixels(img, PADDING, 80, 326, 100)

    def test_weather_draws_detail_chips(self) -> None:
        """Detail row shows humidity, pressure, wind, cloud coverage."""
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 80, 500, 100)

    def test_weather_forecast_precipitation(self) -> None:
        """Precipitation amounts appear under forecast days when > 0."""
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
                "forecast_days": 3,
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        # Forecast area with hi/lo temps and precipitation
        assert_has_dark_pixels(img, 50, 150, 550, 210)

    def test_weather_forecast_separate_hilo(self) -> None:
        """High and low temps are on separate lines in forecast."""
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
                "forecast_days": 3,
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        # High temp row (y ~ 162 at s=1: forecast_y+52)
        assert_has_dark_pixels(img, 50, 155, 550, 175)
        # Low temp row (y ~ 180 at s=1: forecast_y+70)
        assert_has_dark_pixels(img, 50, 175, 550, 195, threshold=200)

    def test_weather_rainy_condition(self) -> None:
        states = {
            "weather.home": {
                "state": "rainy",
                "attributes": {
                    "temperature": 10,
                    "humidity": 90,
                    "wind_speed": 25,
                    "forecast": [],
                },
            }
        }
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
            }
        ]
        result = render_dashboard(
            widgets,
            {"width": 600, "height": 200, "states": states},
        )
        img = png_to_image(result)
        assert_has_dark_pixels(
            img, PADDING, 10, PADDING + 90, 100, threshold=200
        )


MOCK_SENSOR_STATES = {
    "sensor.living_room_temperature": {
        "state": "22.1",
        "attributes": {
            "unit_of_measurement": "°C",
            "friendly_name": "Living Room",
        },
    },
    "sensor.bedroom_temperature": {
        "state": "19.8",
        "attributes": {
            "unit_of_measurement": "°C",
            "friendly_name": "Bedroom",
        },
    },
    "sensor.humidity": {
        "state": "45",
        "attributes": {
            "unit_of_measurement": "%",
            "friendly_name": "Humidity",
        },
    },
}


class TestRenderSensorRows:
    _DEFAULTS: dict[str, object] = {
        "width": 400,
        "height": 300,
        "states": MOCK_SENSOR_STATES,
    }

    def _config(self, **overrides: object) -> dict[str, object]:
        return make_config(self._DEFAULTS, **overrides)

    def test_sensor_rows_draws_labels(self) -> None:
        widgets = [
            {
                "type": "sensor_rows",
                "x": PADDING,
                "y": 10,
                "entities": [
                    "sensor.living_room_temperature",
                    "sensor.bedroom_temperature",
                ],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 200, 40)
        assert_has_dark_pixels(img, PADDING, 40, 200, 70)

    def test_sensor_rows_with_title(self) -> None:
        widgets = [
            {
                "type": "sensor_rows",
                "x": PADDING,
                "y": 10,
                "title": "Temperature",
                "entities": [
                    "sensor.living_room_temperature",
                ],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 200, 35)

    def test_sensor_rows_values_right_aligned(
        self,
    ) -> None:
        widgets = [
            {
                "type": "sensor_rows",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.living_room_temperature"],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, 300, 10, 400, 40)

    def test_sensor_rows_missing_entity_skipped(
        self,
    ) -> None:
        widgets = [
            {
                "type": "sensor_rows",
                "x": PADDING,
                "y": 10,
                "entities": [
                    "sensor.nonexistent",
                    "sensor.humidity",
                ],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 200, 40)

    def test_sensor_rows_empty_entities(self) -> None:
        widgets = [
            {
                "type": "sensor_rows",
                "x": PADDING,
                "y": 10,
                "entities": [],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_all_white(img, 0, 0, 400, 300)


class TestRenderDeviceBattery:
    _DEFAULTS: dict[str, object] = {
        "width": 400,
        "height": 100,
        "device_battery_level": 75,
    }

    def _config(self, **overrides: object) -> dict[str, object]:
        return make_config(self._DEFAULTS, **overrides)

    def test_device_battery_draws_fill(self) -> None:
        widgets = [{"type": "device_battery", "x": PADDING, "y": 20}]
        result = render_dashboard(widgets, self._config())
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING + 1, 33, PADDING + 16, 42)

    def test_device_battery_none_is_noop(self) -> None:
        widgets = [{"type": "device_battery", "x": PADDING, "y": 20}]
        result = render_dashboard(
            widgets, self._config(device_battery_level=None)
        )
        img = png_to_image(result)
        assert_all_white(img, 0, 0, 400, 100)

    def test_device_battery_missing_key_is_noop(self) -> None:
        widgets = [{"type": "device_battery", "x": PADDING, "y": 20}]
        result = render_dashboard(widgets, {"width": 400, "height": 100})
        img = png_to_image(result)
        assert_all_white(img, 0, 0, 400, 100)

    def test_device_battery_draws_percentage_text(self) -> None:
        widgets = [{"type": "device_battery", "x": PADDING, "y": 20}]
        result = render_dashboard(widgets, self._config())
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING + 28, 29, PADDING + 70, 47)

    def test_device_battery_draws_nub(self) -> None:
        widgets = [{"type": "device_battery", "x": PADDING, "y": 20}]
        result = render_dashboard(widgets, self._config())
        img = png_to_image(result)
        assert_has_dark_pixels(
            img, PADDING + 23, 35, PADDING + 25, 40, threshold=200
        )

    def test_device_battery_zero_percent(self) -> None:
        widgets = [{"type": "device_battery", "x": PADDING, "y": 20}]
        result = render_dashboard(
            widgets, self._config(device_battery_level=0)
        )
        img = png_to_image(result)
        assert_has_dark_pixels(
            img, PADDING, 32, PADDING + 25, 43, threshold=200
        )

    def test_device_battery_100_percent(self) -> None:
        widgets = [{"type": "device_battery", "x": PADDING, "y": 20}]
        result = render_dashboard(
            widgets, self._config(device_battery_level=100)
        )
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING + 1, 33, PADDING + 21, 42)

    def test_device_battery_icon_scales_with_font_size(self) -> None:
        # font_size=48 → s=2 → icon is 44×20 instead of default 22×10
        widgets = [
            {
                "type": "device_battery",
                "x": PADDING,
                "y": 40,
                "font_size": 48,
            }
        ]
        result = render_dashboard(
            widgets, self._config(device_battery_level=75)
        )
        img = png_to_image(result)
        # Scaled icon body: PADDING to PADDING+44, icon_y=64 to 84
        assert_has_dark_pixels(
            img, PADDING, 64, PADDING + 44, 85, threshold=200
        )

        # At default font_size, the gap between nub-end (PADDING+24) and
        # label-start (PADDING+30) is clear — proving the scaled icon actually
        # extends its body into a region the default size never reaches.
        result_default = render_dashboard(
            [{"type": "device_battery", "x": PADDING, "y": 40}],
            self._config(device_battery_level=75),
        )
        img_default = png_to_image(result_default)
        assert all(
            pixel(img_default, x, y) >= 200
            for x in range(PADDING + 25, PADDING + 30)
            for y in range(40, 50)
        )


MOCK_STATUS_ICON_STATES = {
    "binary_sensor.front_door": {
        "state": "on",
        "attributes": {
            "friendly_name": "Front Door",
            "device_class": "door",
        },
    },
    "binary_sensor.kitchen_window": {
        "state": "off",
        "attributes": {
            "friendly_name": "Kitchen Window",
            "device_class": "window",
        },
    },
    "binary_sensor.water_alarm": {
        "state": "on",
        "attributes": {
            "friendly_name": "Water Alarm",
            "device_class": "moisture",
        },
    },
    "binary_sensor.motion": {
        "state": "off",
        "attributes": {
            "friendly_name": "Motion",
            "device_class": "motion",
        },
    },
}


class TestRenderStatusIcons:
    _DEFAULTS: dict[str, object] = {
        "width": 500,
        "height": 200,
        "states": MOCK_STATUS_ICON_STATES,
    }

    def _config(self, **overrides: object) -> dict[str, object]:
        return make_config(self._DEFAULTS, **overrides)

    def test_status_icons_draws_entities(self) -> None:
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "entities": [
                    "binary_sensor.front_door",
                    "binary_sensor.kitchen_window",
                ],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 300, 36, threshold=200)

    def test_status_icons_problem_fills_square(self) -> None:
        # front_door: state=on, device_class=door → problem → filled square
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "entities": ["binary_sensor.front_door"],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(
            img, PADDING, 14, PADDING + 13, 27, threshold=64
        )

    def test_status_icons_ok_draws_outline(self) -> None:
        # kitchen_window: state=off → not a problem → outline only
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "entities": ["binary_sensor.kitchen_window"],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        # Interior of the outline square should NOT be solid black
        interior_black = all(
            pixel(img, x, y) < 64
            for x in range(PADDING + 2, PADDING + 11)
            for y in range(16, 25)
        )
        assert not interior_black
        # But the outline itself should have some dark pixels
        assert_has_dark_pixels(
            img, PADDING, 14, PADDING + 13, 27, threshold=200
        )

    def test_status_icons_with_title(self) -> None:
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "title": "Doors",
                "entities": ["binary_sensor.front_door"],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 200, 40)
        assert_has_dark_pixels(img, PADDING, 40, 200, 70)

    def test_status_icons_missing_entity_skipped(self) -> None:
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "entities": [
                    "binary_sensor.nonexistent",
                    "binary_sensor.kitchen_window",
                ],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 300, 36, threshold=200)

    def test_status_icons_empty_entities_is_noop(self) -> None:
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "entities": [],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_all_white(img, 0, 0, 500, 200)

    def test_status_icons_wraps_to_next_row(self) -> None:
        # Narrow canvas forces wrapping after first entity
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "entities": [
                    "binary_sensor.front_door",
                    "binary_sensor.kitchen_window",
                    "binary_sensor.water_alarm",
                    "binary_sensor.motion",
                ],
            }
        ]
        result = render_dashboard(
            widgets,
            {"width": 200, "height": 200, "states": MOCK_STATUS_ICON_STATES},
        )

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 36, 180, 80, threshold=200)


MOCK_WASTE_SCHEDULE_STATES = {
    "sensor.restmull": {
        "state": "2026-05-03",
        "attributes": {"friendly_name": "Restmull"},
    },
    "sensor.gelbe_tonne": {
        "state": "2026-05-04",
        "attributes": {"friendly_name": "Gelbe Tonne"},
    },
    "sensor.papier": {
        "state": "2026-05-05",
        "attributes": {"friendly_name": "Papier"},
    },
}

_TODAY = dt.date(2026, 5, 2)
_PATCH_NOW = "custom_components.eink_dashboard.render.date"


class TestWasteDateHelpers:
    def test_parse_iso_date(self) -> None:
        today = dt.date(2026, 5, 2)
        assert _parse_days_until("2026-05-02", today) == 0
        assert _parse_days_until("2026-05-03", today) == 1
        assert _parse_days_until("2026-05-09", today) == 7

    def test_parse_integer_string(self) -> None:
        today = dt.date(2026, 5, 2)
        assert _parse_days_until("0", today) == 0
        assert _parse_days_until("3", today) == 3

    def test_parse_invalid_returns_none(self) -> None:
        today = dt.date(2026, 5, 2)
        assert _parse_days_until("unavailable", today) is None
        assert _parse_days_until("unknown", today) is None
        assert _parse_days_until("", today) is None

    def test_format_today(self) -> None:
        assert _format_relative_date(0, "2026-05-02") == "today"

    def test_format_tomorrow(self) -> None:
        assert _format_relative_date(1, "2026-05-03") == "tomorrow"

    def test_format_in_n_days(self) -> None:
        assert _format_relative_date(3, "2026-05-05") == "in 3 days"

    def test_format_none_returns_raw(self) -> None:
        assert _format_relative_date(None, "unavailable") == "unavailable"

    def test_format_negative_returns_raw(self) -> None:
        assert _format_relative_date(-1, "2026-05-01") == "2026-05-01"


class TestRenderWasteSchedule:
    _DEFAULTS: dict[str, object] = {
        "width": 500,
        "height": 300,
        "states": MOCK_WASTE_SCHEDULE_STATES,
    }

    def _config(self, **overrides: object) -> dict[str, object]:
        return make_config(self._DEFAULTS, **overrides)

    def test_waste_schedule_draws_entities(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": [
                    "sensor.restmull",
                    "sensor.gelbe_tonne",
                ],
            }
        ]
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 300, 38, threshold=200)
        assert_has_dark_pixels(img, PADDING, 38, 300, 66, threshold=200)

    def test_waste_schedule_with_title(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "title": "Waste Collection",
                "entities": ["sensor.restmull"],
            }
        ]
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 250, 40)
        assert_has_dark_pixels(img, PADDING, 42, 300, 70, threshold=200)

    def test_waste_schedule_missing_entity_skipped(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": [
                    "sensor.nonexistent",
                    "sensor.restmull",
                ],
            }
        ]
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 300, 38, threshold=200)

    def test_waste_schedule_empty_entities_is_noop(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": [],
            }
        ]
        result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_all_white(img, 0, 0, 500, 300)

    def test_waste_schedule_today_fills_circle(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.restmull"],
            }
        ]
        # restmull state is 2026-05-03; today = 2026-05-03 → days=0
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = dt.date(2026, 5, 3)
            result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(
            img, PADDING, 16, PADDING + 11, 27, threshold=64
        )

    def test_waste_schedule_tomorrow_fills_circle(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.restmull"],
            }
        ]
        # restmull state is 2026-05-03; today = 2026-05-02 → days=1
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(
            img, PADDING, 16, PADDING + 11, 27, threshold=64
        )

    def test_waste_schedule_future_draws_outline(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.gelbe_tonne"],
            }
        ]
        # gelbe_tonne state is 2026-05-04; today = 2026-05-02 → days=2
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        # Interior of circle should NOT be solid black
        interior_black = all(
            pixel(img, x, y) < 64
            for x in range(PADDING + 2, PADDING + 9)
            for y in range(18, 25)
        )
        assert not interior_black
        # But the circle boundary should have some non-white pixels
        assert_has_dark_pixels(
            img, PADDING, 16, PADDING + 11, 27, threshold=240
        )

    def test_waste_schedule_date_right_aligned(self) -> None:
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.restmull"],
            }
        ]
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config())

        img = png_to_image(result)
        assert_has_dark_pixels(img, 380, 10, 476, 38, threshold=200)

    def test_waste_schedule_integer_state(self) -> None:
        states = {
            "sensor.restmull": {
                "state": "3",
                "attributes": {"friendly_name": "Restmull"},
            }
        }
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.restmull"],
            }
        ]
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config(states=states))

        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 300, 38, threshold=200)

    def test_waste_schedule_past_date_skipped(self) -> None:
        states = {
            "sensor.restmull": {
                "state": "2026-05-01",
                "attributes": {"friendly_name": "Restmull"},
            }
        }
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.restmull"],
            }
        ]
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config(states=states))

        img = png_to_image(result)
        assert_all_white(img, 0, 0, 500, 300)

    def test_waste_schedule_beyond_3_days_skipped(self) -> None:
        states = {
            "sensor.restmull": {
                "state": "2026-05-06",
                "attributes": {"friendly_name": "Restmull"},
            }
        }
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "entities": ["sensor.restmull"],
            }
        ]
        # 2026-05-06 is 4 days from today (2026-05-02) → filtered out
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(widgets, self._config(states=states))

        img = png_to_image(result)
        assert_all_white(img, 0, 0, 500, 300)


class TestFontSizeControls:
    """Verify that font_size config is respected per widget type."""

    def test_weather_custom_font_size(self) -> None:
        # font_size=11 → s=11/32≈0.34; icon_size=22, temp at x+26
        # detail_y = y+icon_size+round(8*s) = 10+22+3 = 35
        widgets = [
            {
                "type": "weather",
                "entity": "weather.home",
                "x": PADDING,
                "y": 10,
                "font_size": 11,
            }
        ]
        result = render_dashboard(
            widgets,
            {"width": 400, "height": 200, "states": MOCK_WEATHER_STATE},
        )
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING + 22, 10, PADDING + 100, 40)
        # Detail chips at detail_y=35 (wrong /22 divisor shifts them to y=46)
        assert_has_dark_pixels(img, PADDING, 35, PADDING + 200, 46)

    def test_sensor_rows_custom_font_size(self) -> None:
        # font_size=30 → s≈1.364; row_height=41
        # Row 1 at y=10; row 2 at y=51.
        widgets = [
            {
                "type": "sensor_rows",
                "x": PADDING,
                "y": 10,
                "font_size": 30,
                "entities": [
                    "sensor.living_room_temperature",
                    "sensor.bedroom_temperature",
                ],
            }
        ]
        result = render_dashboard(
            widgets,
            {"width": 400, "height": 200, "states": MOCK_SENSOR_STATES},
        )
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 200, 42)
        # Row 2 exists at the scaled position
        assert_has_dark_pixels(img, PADDING, 51, 200, 85)

    def test_device_battery_custom_font_size(self) -> None:
        # font_size=20 — larger percentage label
        widgets = [
            {"type": "device_battery", "x": PADDING, "y": 20, "font_size": 20}
        ]
        result = render_dashboard(
            widgets,
            {"width": 400, "height": 100, "device_battery_level": 75},
        )
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING + 28, 10, PADDING + 90, 40)

    def test_status_icons_custom_font_size(self) -> None:
        # font_size=14 → s=14/28=0.5; sz=round(12*0.5)=6
        # icon_top = y + round(4*0.5) = 12; icon fills [24,12]-[30,18]
        widgets = [
            {
                "type": "status_icons",
                "x": PADDING,
                "y": 10,
                "font_size": 14,
                "entities": list(MOCK_STATUS_ICON_STATES.keys()),
            }
        ]
        result = render_dashboard(
            widgets,
            {"width": 500, "height": 100, "states": MOCK_STATUS_ICON_STATES},
        )
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 12, PADDING + 6, 18)
        # Column just past the icon (x=PADDING+7) in the icon's y-band
        # must be white — verifies the icon is smaller than default sz=12.
        icon_not_oversized = all(
            pixel(img, PADDING + 7, y) == 255 for y in range(12, 18)
        )
        assert icon_not_oversized

    def test_waste_schedule_custom_font_size(self) -> None:
        # font_size=42 → s=42/28=1.5; row_height=round(28*1.5)=42
        # Row 1 at y=10; row 2 at y=52.
        widgets = [
            {
                "type": "waste_schedule",
                "x": PADDING,
                "y": 10,
                "font_size": 42,
                "entities": ["sensor.restmull", "sensor.gelbe_tonne"],
            }
        ]
        with patch(_PATCH_NOW, wraps=dt.date) as mock_dt:
            mock_dt.today.return_value = _TODAY
            result = render_dashboard(
                widgets,
                {
                    "width": 400,
                    "height": 150,
                    "states": MOCK_WASTE_SCHEDULE_STATES,
                },
            )
        img = png_to_image(result)
        assert_has_dark_pixels(img, PADDING, 10, 350, 50, threshold=200)
        assert_has_dark_pixels(img, PADDING, 52, 350, 100, threshold=200)


class TestLoadFont:
    def test_medium_returns_different_object(self) -> None:
        # medium=True must produce a distinct cached object
        regular = _load_font(18)
        medium = _load_font(18, medium=True)
        assert regular is not medium
        # When the TTF is available, verify the actual font style
        if hasattr(medium, "getname"):
            family, style = medium.getname()
            if style != "Regular":
                assert style == "Medium"

    def test_regular_is_cached(self) -> None:
        assert _load_font(18) is _load_font(18)

    def test_medium_is_cached(self) -> None:
        assert _load_font(18, medium=True) is _load_font(18, medium=True)

    def test_size_clamped_to_minimum(self) -> None:
        assert _load_font(0) is not None
        assert _load_font(-5) is not None
        assert _load_font(0, medium=True) is not None
        assert _load_font(-5, medium=True) is not None


class TestComputeMetrics:
    def test_returns_frozen_dataclass(self) -> None:
        m = _compute_metrics(56)
        assert isinstance(m, WidgetMetrics)
        with pytest.raises(AttributeError):
            m.border = 99  # type: ignore[misc]

    def test_reference_baseline_h56(self) -> None:
        m = _compute_metrics(56)
        assert m.border == 2
        assert m.padding == 12
        assert m.radius == 12
        assert m.icon_dia == 36
        assert m.font_primary == 18
        assert m.font_secondary == 14
        assert m.divider == 4
        assert m.inner_gap == 12
        assert m.left_bar == 4

    def test_minimum_clamps_small_row_h(self) -> None:
        m = _compute_metrics(10)
        assert m.border == 2
        assert m.font_primary == 10
        assert m.font_secondary == 10
        assert m.divider == 2
        assert m.left_bar == 2

    def test_unclamped_fields_small_row_h(self) -> None:
        m = _compute_metrics(10)
        assert m.padding == 2
        assert m.radius == 2
        assert m.icon_dia == 6
        assert m.inner_gap == 2

    def test_scales_at_large_row_h(self) -> None:
        m = _compute_metrics(200)
        assert m.border == 8
        assert m.padding == 42
        assert m.radius == 42
        assert m.icon_dia == 128
        assert m.font_primary == 64
        assert m.font_secondary == 50
        assert m.divider == 14
        assert m.inner_gap == 42
        assert m.left_bar == 14

    def test_clamp_boundary_border(self) -> None:
        assert _compute_metrics(37).border == 2  # clamped: round(1.48) = 1 < 2
        assert _compute_metrics(100).border == 4  # natural: round(4.0) = 4 > 2

    def test_all_fields_present(self) -> None:
        m = _compute_metrics(56)
        for f in dataclasses.fields(WidgetMetrics):
            assert isinstance(getattr(m, f.name), int), f"{f.name} is not int"


class TestDrawCardContainer:
    # Canvas 400x200, card at (10, 10)-(310, 110), metrics from row_h=56.
    _X, _Y, _W, _H = 10, 10, 300, 100

    def _blank(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("L", (400, 200), 255)
        return img, ImageDraw.Draw(img)

    def _m(self) -> WidgetMetrics:
        return _compute_metrics(
            56
        )  # border=2, padding=12, radius=12, left_bar=4

    def test_border_draws_dark_pixels_on_all_edges(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_container(draw, self._X, self._Y, self._W, self._H, m)
        assert_has_dark_pixels(
            img,
            self._X + m.radius,
            self._Y,
            self._X + self._W - m.radius,
            self._Y + m.border,
        )
        assert_has_dark_pixels(
            img,
            self._X + m.radius,
            self._Y + self._H - m.border,
            self._X + self._W - m.radius,
            self._Y + self._H,
        )
        assert_has_dark_pixels(
            img,
            self._X,
            self._Y + m.radius,
            self._X + m.border,
            self._Y + self._H - m.radius,
        )
        assert_has_dark_pixels(
            img,
            self._X + self._W - m.border,
            self._Y + m.radius,
            self._X + self._W,
            self._Y + self._H - m.radius,
        )

    def test_border_interior_is_white(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_container(draw, self._X, self._Y, self._W, self._H, m)
        inset = m.border + 1  # +1 for Pillow anti-aliasing at the border edge
        assert_all_white(
            img,
            self._X + inset,
            self._Y + m.radius,  # skip rounded corners
            self._X + self._W - inset,
            self._Y + self._H - m.radius,
        )

    def test_border_returns_zero_offset(self) -> None:
        _, draw = self._blank()
        m = self._m()
        offset = _draw_card_container(
            draw, self._X, self._Y, self._W, self._H, m
        )
        assert offset == 0

    def test_left_bar_draws_gray_on_left(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_container(
            draw, self._X, self._Y, self._W, self._H, m, card_style="left_bar"
        )
        assert_has_gray_pixels(
            img,
            self._X,
            self._Y + 10,
            self._X + m.left_bar,
            self._Y + self._H - 10,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )

    def test_left_bar_right_edge_is_white(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_container(
            draw, self._X, self._Y, self._W, self._H, m, card_style="left_bar"
        )
        # Area to the right of bar + padding should have no decoration
        right_start = self._X + m.left_bar + m.padding + 1
        assert_all_white(
            img, right_start, self._Y, self._X + self._W, self._Y + 1
        )
        assert_all_white(
            img,
            right_start,
            self._Y + 10,
            self._X + self._W,
            self._Y + self._H - 10,
        )

    def test_left_bar_returns_offset(self) -> None:
        _, draw = self._blank()
        m = self._m()
        offset = _draw_card_container(
            draw, self._X, self._Y, self._W, self._H, m, card_style="left_bar"
        )
        assert offset == m.left_bar + m.padding

    def test_none_leaves_canvas_white(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_container(
            draw, self._X, self._Y, self._W, self._H, m, card_style="none"
        )
        assert_all_white(
            img, self._X, self._Y, self._X + self._W, self._Y + self._H
        )

    def test_none_returns_zero_offset(self) -> None:
        _, draw = self._blank()
        m = self._m()
        offset = _draw_card_container(
            draw, self._X, self._Y, self._W, self._H, m, card_style="none"
        )
        assert offset == 0

    def test_left_bar_widens_for_2_level_display(self) -> None:
        img, draw = self._blank()
        m = self._m()  # left_bar=4
        offset = _draw_card_container(
            draw,
            self._X,
            self._Y,
            self._W,
            self._H,
            m,
            card_style="left_bar",
            grayscale_levels=2,
        )
        # At h=56: widened bar = max(10, 4*3) = 12px.
        # Gray should extend to pixel 12.
        widened = max(10, m.left_bar * 3)
        assert_has_gray_pixels(
            img,
            self._X + m.left_bar,
            self._Y + 10,
            self._X + widened,
            self._Y + self._H - 10,
            low=COLOR_GRAY - 20,
            high=COLOR_GRAY + 20,
        )
        assert offset == widened + m.padding

    def test_left_bar_standard_width_without_2_level(self) -> None:
        img, draw = self._blank()
        m = self._m()  # left_bar=4
        _draw_card_container(
            draw,
            self._X,
            self._Y,
            self._W,
            self._H,
            m,
            card_style="left_bar",
            grayscale_levels=16,
        )
        # Pixel just past the normal bar width (4px) should be white.
        assert_all_white(
            img,
            self._X + m.left_bar + 1,
            self._Y + 1,
            self._X + 10,
            self._Y + self._H - 1,
        )

    def test_left_bar_no_config_uses_standard_width(self) -> None:
        img, draw = self._blank()
        m = self._m()  # left_bar=4
        _draw_card_container(
            draw, self._X, self._Y, self._W, self._H, m, card_style="left_bar"
        )
        # Pixel just past the normal bar width (4px) should be white.
        assert_all_white(
            img,
            self._X + m.left_bar + 1,
            self._Y + 1,
            self._X + 10,
            self._Y + self._H - 1,
        )

    def test_default_card_style_is_border(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_container(draw, self._X, self._Y, self._W, self._H, m)
        # Default should behave identically to card_style="border".
        assert_has_dark_pixels(
            img,
            self._X + m.radius,
            self._Y,
            self._X + self._W - m.radius,
            self._Y + m.border,
        )


class TestDrawCardRow:
    # Canvas 400x200, row at (10, 10), width 300, row_h 56 (reference size).
    _X, _Y, _W, _H = 10, 10, 300, 56

    def _blank(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("L", (400, 200), 255)
        return img, ImageDraw.Draw(img)

    def _m(self) -> WidgetMetrics:
        # padding=12, icon_dia=36, inner_gap=12,
        # font_primary=18, font_secondary=14
        return _compute_metrics(56)

    def test_icon_circle_drawn(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw, img, self._X, self._Y, self._W, self._H, m, primary="Hello"
        )
        # Icon circle: cx=10+12=22, circle_y=10+(56-36)//2=20, dia=36
        assert_has_gray_pixels(img, 22, 20, 58, 56, low=100, high=140)

    def test_letter_fallback(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            m,
            primary="Temperature",
        )
        # Gray circle is drawn
        assert_has_gray_pixels(img, 24, 22, 56, 54, low=100, high=140)
        # White letter pixels exist inside the circle
        # (letter on gray background)
        center_x = 22 + m.icon_dia // 2
        center_y = 20 + m.icon_dia // 2
        has_white = any(
            pixel(img, x, y) == 255
            for x in range(center_x - 6, center_x + 6)
            for y in range(center_y - 6, center_y + 6)
        )
        assert has_white, "no white letter pixels found inside icon circle"

    def test_primary_text_drawn(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw, img, self._X, self._Y, self._W, self._H, m, primary="Hello"
        )
        # Text starts at x=22+36+12=70
        assert_has_dark_pixels(img, 70, self._Y, 250, self._Y + self._H)

    def test_secondary_text_drawn(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            m,
            primary="Hello",
            secondary="world",
        )
        # Primary (black) in the upper half, secondary (gray) below it.
        assert_has_dark_pixels(img, 70, 22, 250, 40)
        assert_has_gray_pixels(img, 70, 40, 250, 56, low=100, high=140)

    def test_primary_only_centered(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw, img, self._X, self._Y, self._W, self._H, m, primary="Hello"
        )
        # Single-line primary should be vertically centered in the row.
        # Tolerance of 5px allows for PIL ascender/descender space that
        # may not contain rendered pixels, shifting the glyph center.
        text_bb = content_bbox(
            img, 70, self._Y, self._X + self._W, self._Y + self._H
        )
        assert text_bb is not None, "no text content found"
        text_cy = (text_bb[1] + text_bb[3]) / 2
        row_cy = self._Y + self._H / 2  # 10 + 28 = 38
        assert abs(text_cy - row_cy) <= 5, (
            f"primary text not centered: "
            f"text_cy={text_cy:.1f}, row_cy={row_cy:.1f}"
        )

    def test_value_right_aligned(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            m,
            primary="Hello",
            value="today",
        )
        # Value text near right edge (x+w-padding = 298)
        assert_has_gray_pixels(
            img,
            240,
            self._Y,
            self._X + self._W,
            self._Y + self._H,
            low=100,
            high=140,
        )

    def test_icon_vertically_centered_with_text(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            m,
            primary="Hello",
            secondary="world",
        )
        # Tolerance of 4px accounts for font metric vs rendered-pixel
        # center offset.
        assert_vertically_centered(
            img,
            icon_region=(22, self._Y, 58, self._Y + self._H),
            text_region=(70, self._Y, 250, self._Y + self._H),
            tolerance=4.0,
        )

    def test_no_value_area_is_white(self) -> None:
        img, draw = self._blank()
        m = self._m()
        _draw_card_row(
            draw, img, self._X, self._Y, self._W, self._H, m, primary="Hi"
        )
        # Area near right padding edge should be white when no value provided
        right_area_start = self._X + self._W - m.padding - 20
        assert_all_white(
            img,
            right_area_start,
            self._Y,
            self._X + self._W,
            self._Y + self._H,
        )

    def test_icon_scales_with_row_h(self) -> None:
        # Render at row_h=56 (small) and row_h=112 (large, 2×).
        row_h_s = 56
        img_s, draw_s = self._blank()
        m_s = _compute_metrics(row_h_s)
        _draw_card_row(
            draw_s,
            img_s,
            self._X,
            self._Y,
            self._W,
            row_h_s,
            m_s,
            primary="Hello",
        )
        icon_x_s = self._X + m_s.padding
        cy_s = self._Y + (row_h_s - m_s.icon_dia) // 2
        region_s = (
            icon_x_s,
            cy_s,
            icon_x_s + m_s.icon_dia,
            cy_s + m_s.icon_dia,
        )

        row_h_l = 112
        img_l = Image.new("L", (800, 400), 255)
        draw_l = ImageDraw.Draw(img_l)
        m_l = _compute_metrics(row_h_l)
        _draw_card_row(
            draw_l,
            img_l,
            self._X,
            self._Y,
            self._W * 2,
            row_h_l,
            m_l,
            primary="Hello",
        )
        icon_x_l = self._X + m_l.padding
        cy_l = self._Y + (row_h_l - m_l.icon_dia) // 2
        region_l = (
            icon_x_l,
            cy_l,
            icon_x_l + m_l.icon_dia,
            cy_l + m_l.icon_dia,
        )

        assert_scales_proportionally(
            img_s,
            img_l,
            region_s,
            region_l,
            expected_ratio=2.0,
        )


class TestDrawChip:
    # Canvas 400x200; chip at (20, 20) with h=40.
    _X, _Y, _H = 20, 20, 40

    def _blank(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("L", (400, 200), 255)
        return img, ImageDraw.Draw(img)

    def _font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return _load_font(14)

    def _border(self) -> int:
        # border = max(2, round(40 * 0.04)) = 2
        return max(2, round(self._H * 0.04))

    def test_pill_shape_corners_white(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip(
            draw, img, self._X, self._Y, self._H, "OK", font, self._border()
        )
        # The extreme corner pixels of the chip bounding box should be white
        # because the pill shape clips them away (radius = h // 2 = 20).
        assert pixel(img, self._X, self._Y) == 255, (
            "top-left corner should be white (pill clips it)"
        )
        assert pixel(img, self._X, self._Y + self._H - 1) == 255, (
            "bottom-left corner should be white (pill clips it)"
        )

    def test_pill_border_drawn(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip(
            draw, img, self._X, self._Y, self._H, "OK", font, self._border()
        )
        # The top-center of the pill must have border pixels (past the radius
        # zone on each side); radius=20 so center is well past the arc zone.
        radius = self._H // 2
        mid_x = self._X + radius + 5
        assert_has_dark_pixels(img, mid_x, self._Y, mid_x + 10, self._Y + 3)

    def test_text_drawn_inside_chip(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip(
            draw, img, self._X, self._Y, self._H, "OK", font, self._border()
        )
        # Text "OK" should produce dark pixels somewhere inside the chip
        # interior (well inward from the pill border).
        pad_h = round(self._H * 0.18)
        text_x = self._X + pad_h
        assert_has_dark_pixels(
            img,
            text_x,
            self._Y + 4,
            text_x + 30,
            self._Y + self._H - 4,
        )

    def test_normal_mode_white_interior(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip(
            draw, img, self._X, self._Y, self._H, "X", font, self._border()
        )
        # A horizontal strip at the vertical center (between border pixels)
        # that avoids the text column should be white in normal mode.
        # Text "X" is narrow; check the right half of the chip interior.
        mid_y = self._Y + self._H // 2
        # Right of text: measure text width to find a safe white zone.
        bbox = draw.textbbox((0, 0), "X", font=font)
        text_w = bbox[2] - bbox[0]
        pad_h = round(self._H * 0.18)
        text_right = self._X + pad_h + text_w
        chip_right = text_right + pad_h
        # There must be a white region between text end and chip right edge.
        assert_all_white(
            img,
            text_right + 2,
            mid_y - 2,
            chip_right - 2,
            mid_y + 2,
        )

    def test_inverted_mode_dark_fill(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip(
            draw,
            img,
            self._X,
            self._Y,
            self._H,
            "OK",
            font,
            self._border(),
            inverted=True,
        )
        # Check a pixel in the fill area that is left of the text column
        # so it is not overdrawn by white text.
        # Text starts at cx = x + pad_h = 20 + round(40*0.18) = 20+7 = 27.
        # A pixel at x=24 (inside pill, before text) must be black fill.
        mid_y = self._Y + self._H // 2
        assert pixel(img, self._X + 4, mid_y) == COLOR_BLACK, (
            "inverted chip interior (left of text) should be black"
        )

    def test_returns_correct_x(self) -> None:
        img, draw = self._blank()
        font = self._font()
        result = _draw_chip(
            draw, img, self._X, self._Y, self._H, "OK", font, self._border()
        )
        bbox = draw.textbbox((0, 0), "OK", font=font)
        text_w = bbox[2] - bbox[0]
        pad_h = round(self._H * 0.18)
        expected_chip_w = pad_h * 2 + text_w
        assert result == self._X + expected_chip_w, (
            f"expected {self._X + expected_chip_w}, got {result}"
        )


class TestDrawChipFlow:
    # Canvas 500x400; flow at (20, 20), width 350, chip h=40.
    _X, _Y, _W, _H = 20, 20, 350, 40

    def _blank(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("L", (500, 400), 255)
        return img, ImageDraw.Draw(img)

    def _font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return _load_font(14)

    def _border(self) -> int:
        return max(2, round(self._H * 0.04))

    def test_single_chip(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip_flow(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            [{"text": "OK"}],
            font,
            self._border(),
        )
        # A chip must be drawn at the starting position.
        assert_has_dark_pixels(
            img, self._X, self._Y, self._X + 80, self._Y + self._H
        )
        # No chips should appear on the second row.
        gap = round(self._H * 0.29)
        row2_y = self._Y + self._H + gap
        assert_all_white(
            img, self._X, row2_y, self._X + self._W, row2_y + self._H
        )

    def test_multiple_chips_with_gaps(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip_flow(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            [{"text": "A"}, {"text": "B"}],
            font,
            self._border(),
        )
        # Both chips must be drawn on the first row.
        assert_has_dark_pixels(
            img, self._X, self._Y, self._X + 40, self._Y + self._H
        )
        # Measure where first chip ends to find the gap region.
        bbox_a = draw.textbbox((0, 0), "A", font=font)
        text_w_a = bbox_a[2] - bbox_a[0]
        pad_h = round(self._H * 0.18)
        chip_w_a = pad_h * 2 + text_w_a
        gap = round(self._H * 0.29)
        gap_start = self._X + chip_w_a
        gap_end = gap_start + gap
        # Vertical center strip of the gap region should be white.
        mid_y = self._Y + self._H // 2
        if gap > 2:
            assert_all_white(
                img,
                gap_start + 1,
                mid_y - 2,
                gap_end - 1,
                mid_y + 2,
            )

    def test_wrapping(self) -> None:
        img, draw = self._blank()
        font = self._font()
        # Use long labels so chips fill the row quickly and wrap.
        chips = [
            {"text": "Temperature"},
            {"text": "Humidity"},
            {"text": "Pressure"},
            {"text": "Luminance"},
        ]
        _draw_chip_flow(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            chips,
            font,
            self._border(),
        )
        # At least one chip must appear on the second row.
        gap = round(self._H * 0.29)
        row2_y = self._Y + self._H + gap
        assert_has_dark_pixels(
            img,
            self._X,
            row2_y,
            self._X + self._W,
            row2_y + self._H,
        )

    def test_returns_final_y(self) -> None:
        img, draw = self._blank()
        font = self._font()
        # Single row of chips: result should be y + h.
        result = _draw_chip_flow(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            [{"text": "OK"}],
            font,
            self._border(),
        )
        assert result == self._Y + self._H, (
            f"single-row flow should return {self._Y + self._H}, got {result}"
        )

    def test_empty_chips_returns_y(self) -> None:
        img, draw = self._blank()
        font = self._font()
        result = _draw_chip_flow(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            [],
            font,
            self._border(),
        )
        assert result == self._Y, (
            f"empty chips should return y={self._Y}, got {result}"
        )

    def test_inverted_chip_in_flow(self) -> None:
        img, draw = self._blank()
        font = self._font()
        _draw_chip_flow(
            draw,
            img,
            self._X,
            self._Y,
            self._W,
            self._H,
            [{"text": "ALERT", "inverted": True}],
            font,
            self._border(),
        )
        mid_y = self._Y + self._H // 2
        assert pixel(img, self._X + 4, mid_y) == COLOR_BLACK, (
            "inverted chip in flow should have black fill"
        )


class TestChartAxisOverrides:
    """y_min / y_max let a widget pin the left axis instead of
    auto-scaling to the data range.
    """

    @staticmethod
    def _config(**extra):
        """Display config with one flat series at value 50."""
        base = 1_700_000_000.0
        points = [{"t": base + i * 3600, "v": 50.0} for i in range(6)]
        return {
            "width": 400,
            "height": 240,
            "histories": {"sensor.demo": points},
            **extra,
        }

    @staticmethod
    def _widget(**extra):
        return {
            "type": "chart",
            "x": 0,
            "y": 0,
            "w": 400,
            "h": 240,
            "config": {"series": [{"entity": "sensor.demo"}]},
            **extra,
        }

    def test_override_changes_the_rendering(self) -> None:
        zonder = render_dashboard([self._widget()], self._config())
        met = render_dashboard(
            [self._widget(y_min=0, y_max=200)], self._config()
        )
        assert zonder != met

    @staticmethod
    def _lijn_rij(png: bytes) -> int:
        """Row index of the flat data line: the row with the most dark
        pixels, ignoring the bottom strip that holds the axis labels.
        """
        img = Image.open(io.BytesIO(png)).convert("L")
        beste, beste_aantal = 0, -1
        for y in range(img.height - 60):
            aantal = sum(
                1 for x in range(img.width) if img.getpixel((x, y)) < 128
            )
            if aantal > beste_aantal:
                beste, beste_aantal = y, aantal
        assert beste_aantal > 0, "de grafiek tekende niets"
        return beste

    def test_larger_y_max_pushes_the_line_down(self) -> None:
        """A flat line at 50 sits lower as the axis maximum grows,
        because the same value maps further down a taller range.
        """
        krap = render_dashboard(
            [self._widget(y_min=0, y_max=100)], self._config()
        )
        ruim = render_dashboard(
            [self._widget(y_min=0, y_max=1000)], self._config()
        )
        assert self._lijn_rij(ruim) > self._lijn_rij(krap)

    def test_only_y_max_is_honoured(self) -> None:
        alleen_max = render_dashboard(
            [self._widget(y_max=500)], self._config()
        )
        zonder = render_dashboard([self._widget()], self._config())
        assert alleen_max != zonder


class TestCalendarHelpers:
    """Pure helpers behind the calendar widget."""

    def test_timed_event_splits_into_date_and_time(self) -> None:
        assert _parse_event_start("2026-09-11T13:30:00+02:00") == (
            dt.date(2026, 9, 11),
            "13:30",
        )

    def test_all_day_event_has_no_time(self) -> None:
        assert _parse_event_start("2026-09-11") == (
            dt.date(2026, 9, 11),
            "",
        )

    @pytest.mark.parametrize("raw", ["", "not a date", "2026-13-45"])
    def test_unparsable_start_returns_none(self, raw: str) -> None:
        assert _parse_event_start(raw) is None

    def test_heading_says_today_and_tomorrow(self) -> None:
        today = dt.date(2026, 9, 9)
        args = (
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            "Today",
            "Tomorrow",
        )
        assert _day_heading(today, today, *args) == "Today"
        assert _day_heading(dt.date(2026, 9, 10), today, *args) == "Tomorrow"
        assert _day_heading(dt.date(2026, 9, 11), today, *args) == "Fri 11 Sep"

    def test_heading_uses_the_supplied_names(self) -> None:
        """Day and month names are overridable, so another language works."""
        heading = _day_heading(
            dt.date(2026, 9, 11),
            dt.date(2026, 9, 9),
            ["ma", "di", "wo", "do", "vr", "za", "zo"],
            [
                "jan",
                "feb",
                "mrt",
                "apr",
                "mei",
                "jun",
                "jul",
                "aug",
                "sep",
                "okt",
                "nov",
                "dec",
            ],
            "vandaag",
            "morgen",
        )
        assert heading == "vr 11 sep"

    def test_fit_text_shortens_with_an_ellipsis(self) -> None:
        font = _load_font(20)
        img = Image.new("L", (200, 50), 255)
        draw = ImageDraw.Draw(img)
        kort = _fit_text(draw, "a very long appointment title", font, 60)
        assert kort.endswith("\u2026")
        assert draw.textlength(kort, font=font) <= 60

    def test_fit_text_leaves_short_text_alone(self) -> None:
        font = _load_font(20)
        draw = ImageDraw.Draw(Image.new("L", (200, 50), 255))
        assert _fit_text(draw, "ok", font, 200) == "ok"


class TestRenderCalendar:
    """The calendar widget drawing events from config['calendar_events']."""

    @staticmethod
    def _events(aantal: int = 3) -> list[dict]:
        vandaag = dt.date.today()
        return [
            {
                "entity_id": f"calendar.c{i % 2}",
                "calendar": f"C{i % 2}",
                "start": f"{vandaag + dt.timedelta(days=i)}T09:0{i}:00",
                "summary": f"Event {i}",
                "all_day": False,
            }
            for i in range(aantal)
        ]

    @staticmethod
    def _render(widget_extra=None, events=None) -> Image.Image:
        widget = {"type": "calendar", "x": 10, "y": 10, "w": 380}
        widget.update(widget_extra or {})
        png = render_dashboard(
            [widget],
            {
                "width": 400,
                "height": 480,
                "calendar_events": events or [],
            },
        )
        return png_to_image(png)

    @staticmethod
    def _inkt(img: Image.Image) -> int:
        """Number of dark pixels, a rough proxy for how much was drawn."""
        return sum(img.histogram()[:128])

    def test_no_events_draws_nothing(self) -> None:
        assert_all_white(self._render(events=[]), 0, 0, 400, 480)

    def test_events_are_drawn(self) -> None:
        assert_has_dark_pixels(
            self._render(events=self._events()), 0, 0, 400, 480
        )

    def test_max_events_limits_the_list(self) -> None:
        veel = self._render({"max_events": 6}, events=self._events(6))
        weinig = self._render({"max_events": 2}, events=self._events(6))
        assert self._inkt(veel) > self._inkt(weinig)

    def test_entities_filter_hides_other_calendars(self) -> None:
        alles = self._render(events=self._events(4))
        een = self._render(
            {"entities": ["calendar.c0"]}, events=self._events(4)
        )
        assert self._inkt(een) < self._inkt(alles)

    def test_show_calendar_adds_the_name(self) -> None:
        """With show_calendar on, the calendar name is drawn as well."""
        events = [
            {
                "entity_id": "calendar.bas",
                "calendar": "Bas",
                "start": f"{dt.date.today()}T14:00:00",
                "summary": "Avond",
            }
        ]
        zonder = self._render(events=events)
        met = self._render({"show_calendar": True}, events=events)
        assert self._inkt(met) > self._inkt(zonder)

    def test_name_is_shown_even_when_the_summary_repeats_it(self) -> None:
        """Bas wants the name on every row, also on "Guust hockeytraining"."""
        events = [
            {
                "entity_id": "calendar.guust",
                "calendar": "Guust",
                "start": f"{dt.date.today()}T17:00:00",
                "summary": "Guust hockeytraining",
            }
        ]
        zonder = self._render(events=events)
        met = self._render({"show_calendar": True}, events=events)
        assert self._inkt(met) > self._inkt(zonder)

    @pytest.mark.parametrize("positie", ["inline", "column", "below_time"])
    def test_every_position_draws_the_name(self, positie: str) -> None:
        events = [
            {
                "entity_id": "calendar.bas",
                "calendar": "Bas",
                "start": f"{dt.date.today()}T14:00:00",
                "summary": "Avond",
            }
        ]
        zonder = self._render(events=events)
        met = self._render(
            {"show_calendar": True, "calendar_position": positie},
            events=events,
        )
        assert self._inkt(met) > self._inkt(zonder)

    def test_max_chars_shortens_the_name(self) -> None:
        """Long names get cut so the column keeps one width."""
        events = [
            {
                "entity_id": "calendar.lyndenlaan",
                "calendar": "Lyndenlaan",
                "start": f"{dt.date.today()}T09:00:00",
                "summary": "Papier",
            }
        ]
        heel = self._render(
            {"show_calendar": True, "calendar_position": "inline"},
            events=events,
        )
        kort = self._render(
            {
                "show_calendar": True,
                "calendar_position": "inline",
                "calendar_max_chars": 3,
            },
            events=events,
        )
        assert self._inkt(kort) < self._inkt(heel)

    def test_below_time_leaves_more_room_for_the_summary(self) -> None:
        """The point of below_time: the summary keeps the full width."""
        events = [
            {
                "entity_id": "calendar.familie",
                "calendar": "Familie",
                "start": f"{dt.date.today()}T10:15:00",
                "summary": (
                    "Verhuizing Stephan en Marlies uitgesteld naar okt"
                ),
            }
        ]
        inline = self._render(
            {"show_calendar": True, "calendar_position": "inline"},
            events=events,
        )
        onder = self._render(
            {"show_calendar": True, "calendar_position": "below_time"},
            events=events,
        )
        # Meer inkt op de regel betekent dat er meer van de titel past.
        assert self._inkt(onder) > self._inkt(inline)

    @staticmethod
    def _tijdvak(img: Image.Image) -> set[int]:
        """The tones used in the time of the first event.

        With the default font size the time column is 64 wide and the day
        heading sits above it, so the digits land between y 36 and y 58.
        """
        return {
            v
            for v, n in enumerate(img.crop((10, 36, 74, 58)).histogram())
            if n
        }

    def test_time_is_drawn_darker_than_the_calendar_name(self) -> None:
        """The time carries the row, so it gets its own darker tone."""
        img = self._render({"show_calendar": True}, events=self._events(1))
        assert COLOR_DARK_GRAY in self._tijdvak(img)
        assert COLOR_GRAY not in self._tijdvak(img)

    def test_time_color_overrides_the_default(self) -> None:
        """How dark that lands depends on the panel, so it is settable."""
        img = self._render({"time_color": COLOR_BLACK}, events=self._events(1))
        tonen = self._tijdvak(img)
        assert COLOR_BLACK in tonen
        assert COLOR_DARK_GRAY not in tonen

    def test_unparsable_events_are_skipped_not_fatal(self) -> None:
        kapot = [{"entity_id": "calendar.c0", "start": "", "summary": "x"}]
        assert_all_white(self._render(events=kapot), 0, 0, 400, 480)

    def test_no_day_heading_without_room_for_an_event(self) -> None:
        """A heading at the very bottom with nothing under it looks broken."""
        vandaag = dt.date.today()
        events = [
            {
                "entity_id": "calendar.c0",
                "calendar": "C0",
                "start": f"{vandaag + dt.timedelta(days=i)}T09:00:00",
                "summary": f"Event {i}",
            }
            for i in range(30)
        ]
        img = self._render({"max_events": 30, "row_height": 30}, events=events)
        # De onderste 30 pixels zijn leeg: daar paste geen hele dag meer.
        onderkant = img.crop((0, img.height - 30, img.width, img.height))
        assert sum(onderkant.histogram()[:128]) == 0

    def test_events_stop_at_the_bottom_edge(self) -> None:
        """A long list must not draw past the canvas."""
        img = self._render({"max_events": 50}, events=self._events(50))
        onderste_rij = [img.getpixel((x, 479)) for x in range(400)]
        assert all(p == 255 for p in onderste_rij)


class TestEmojiFallback:
    """Emoji are drawn with Noto Emoji instead of coming out as tofu."""

    @pytest.mark.parametrize(
        "char", ["\U0001f3d1", "\U0001f487", "\u26f3", "\U0001f5d1"]
    )
    def test_pictographs_are_emoji(self, char: str) -> None:
        assert _is_emoji(char)

    @pytest.mark.parametrize("char", ["a", "1", " ", "\u00e9", "-", "\u20ac"])
    def test_plain_text_is_not_emoji(self, char: str) -> None:
        assert not _is_emoji(char)

    def test_runs_split_on_the_boundary(self) -> None:
        assert _split_runs("hockey \U0001f3d1 nu") == [
            ("hockey ", False),
            ("\U0001f3d1", True),
            (" nu", False),
        ]

    def test_variation_selector_stays_with_its_emoji(self) -> None:
        """A trailing U+FE0F must not start a text run of its own."""
        runs = _split_runs("\U0001f3d5\ufe0f kamp")
        assert runs[0] == ("\U0001f3d5\ufe0f", True)
        assert runs[1] == (" kamp", False)

    def test_text_without_emoji_is_one_run(self) -> None:
        assert _split_runs("gewone tekst") == [("gewone tekst", False)]

    def test_empty_text_has_no_runs(self) -> None:
        assert _split_runs("") == []

    def test_emoji_actually_draws_ink(self) -> None:
        """Without the fallback font this would be an empty box or nothing."""
        png = render_dashboard(
            [
                {
                    "type": "text",
                    "x": 5,
                    "y": 5,
                    "text": "\U0001f3d1",
                    "font_size": 30,
                }
            ],
            {"width": 80, "height": 50},
        )
        assert_has_dark_pixels(png_to_image(png), 0, 0, 80, 50)

    def test_emoji_can_be_switched_off_per_widget(self) -> None:
        met = render_dashboard(
            [
                {
                    "type": "text",
                    "x": 5,
                    "y": 5,
                    "text": "\U0001f3d1",
                    "font_size": 30,
                }
            ],
            {"width": 80, "height": 50},
        )
        zonder = render_dashboard(
            [
                {
                    "type": "text",
                    "x": 5,
                    "y": 5,
                    "text": "\U0001f3d1",
                    "font_size": 30,
                    "emoji": False,
                }
            ],
            {"width": 80, "height": 50},
        )
        assert met != zonder


class TestRenderClock:
    """The analog clock: a circle and two hands, nothing else."""

    @staticmethod
    def _render(extra=None) -> Image.Image:
        widget = {"type": "clock", "x": 10, "y": 10, "radius": 20}
        widget.update(extra or {})
        return png_to_image(
            render_dashboard([widget], {"width": 80, "height": 80})
        )

    @staticmethod
    def _inkt(img: Image.Image) -> int:
        return sum(img.histogram()[:200])

    def test_it_draws_something(self) -> None:
        assert_has_dark_pixels(self._render({"time": "10:10"}), 0, 0, 80, 80)

    def test_it_stays_inside_its_box(self) -> None:
        """Nothing outside x..x+2r, so it cannot overlap its neighbours."""
        img = self._render({"time": "10:10"})
        rand = img.crop((0, 0, 80, 10))
        assert sum(rand.histogram()[:128]) == 0

    def test_a_fixed_time_renders_the_same_every_call(self) -> None:
        eerst = self._render({"time": "03:25"})
        weer = self._render({"time": "03:25"})
        assert eerst.tobytes() == weer.tobytes()

    def test_different_times_look_different(self) -> None:
        assert (
            self._render({"time": "03:00"}).tobytes()
            != self._render({"time": "09:00"}).tobytes()
        )

    def test_center_dot_adds_ink(self) -> None:
        zonder = self._render({"time": "10:10"})
        met = self._render({"time": "10:10", "center_dot": True})
        assert self._inkt(met) > self._inkt(zonder)

    def test_offset_winds_the_clock_forward(self) -> None:
        """The screen lags, so the hands may run ahead to compensate."""
        from unittest.mock import patch

        import custom_components.eink_dashboard.render as R

        vast = dt.datetime(2026, 9, 9, 10, 0)

        class NepDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return vast

        with patch.object(R, "datetime", NepDatetime):
            zonder = self._render()
            met = self._render({"offset_minutes": 30})
        assert zonder.tobytes() != met.tobytes()

    def test_offset_is_ignored_for_a_fixed_time(self) -> None:
        """A time given by hand must be drawn exactly as given."""
        kaal = self._render({"time": "04:20"})
        met = self._render({"time": "04:20", "offset_minutes": 30})
        assert kaal.tobytes() == met.tobytes()

    def test_midnight_does_not_crash(self) -> None:
        """Hour 0 must map to 12 o'clock, not to an angle of nothing."""
        assert_has_dark_pixels(self._render({"time": "00:00"}), 0, 0, 80, 80)
