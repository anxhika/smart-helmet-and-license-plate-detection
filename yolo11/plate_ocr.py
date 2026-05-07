from typing import Optional

import cv2
import requests

from yolo11.config import YOLO11Config, get_plate_recognizer_token


def read_plate_number(lp_crop, config: YOLO11Config) -> Optional[str]:
    token = get_plate_recognizer_token(config)
    if token is None:
        return None

    ok, encoded = cv2.imencode(".jpg", lp_crop)
    if not ok:
        return None

    try:
        response = requests.post(
            config.plate_recognizer_url,
            data={"regions": [config.plate_recognizer_region]},
            files={"upload": ("lp.jpg", encoded.tobytes(), "image/jpeg")},
            headers={"Authorization": f"Token {token}"},
            timeout=config.plate_recognizer_timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None

    candidate = results[0].get("plate")
    if not isinstance(candidate, str):
        return None

    cleaned = candidate.strip().lower()
    return cleaned if cleaned else None
