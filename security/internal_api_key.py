"""Xác thực internal API key — luôn bắt buộc, không bypass."""
from fastapi import Header, HTTPException

from config.settings import get_settings


def verify_internal_api_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> None:
    configured_key = get_settings().internal_api_key
    if not configured_key or not configured_key.strip():
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY chưa được cấu hình")
    if not x_internal_api_key or x_internal_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")
