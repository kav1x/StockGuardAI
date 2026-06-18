import base64
import hashlib
import io
import html
import difflib
import os
import re
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from database import (
    authenticate_user,
    approve_manual_payment,
    create_payment,
    create_user,
    current_month_scan_count,
    get_financial_summary,
    get_active_subscription_plans,
    get_all_subscription_plans,
    get_admin_overview_stats,
    get_or_create_project,
    get_scan,
    get_scan_logs_for_admin,
    get_subscription_plan,
    get_user,
    get_user_plan_details,
    get_users_for_admin,
    get_monthly_revenue_summary,
    get_payment_by_ref,
    get_payments,
    get_plan_revenue_breakdown,
    init_db,
    list_all_users_with_usage,
    list_projects,
    list_batch_names,
    list_scans,
    save_scan,
    scan_report_rows,
    total_scan_count,
    set_user_disabled,
    set_user_enabled,
    reset_user_monthly_usage,
    reject_manual_payment,
    update_subscription_plan,
    update_user_plan,
    update_user_profile,
    reset_default_subscription_plans,
)

if load_dotenv:
    load_dotenv()
        
def get_current_user() -> Dict:
    """Fetch the current session user from SQLite and reject disabled accounts."""

    user_id = st.session_state.get("user_id")
    if not user_id:
        return {}
    user = get_user(user_id)
    if not user:
        st.session_state.clear()
        return {}
    if int(user.get("is_disabled") or 0):
        st.session_state.clear()
        st.error("Your account has been disabled. Contact support.")
        st.stop()
    return user


def is_admin_user(user: Dict) -> bool:
    """Return True when the current user has admin privileges."""

    if not user:
        return False

    role = str(user.get("role") or user.get("user_role") or "").strip().lower()
    if role == "admin":
        return True

    email = str(user.get("email") or "").strip().lower()
    admin_email = str(os.getenv("ADMIN_EMAIL") or "").strip().lower()
    return bool(admin_email and email == admin_email)


def require_login() -> Dict:
    """Guard protected pages so anonymous users never render private content."""

    user = get_current_user()
    if not user:
        render_auth_page()
        st.stop()
    return user


def login_is_temporarily_locked() -> bool:
    """Simple session-based login throttling for the beta MVP."""

    locked_until = float(st.session_state.get("login_locked_until", 0))
    return time.time() < locked_until


def record_failed_login_attempt() -> None:
    attempts = int(st.session_state.get("login_failed_attempts", 0)) + 1
    st.session_state["login_failed_attempts"] = attempts
    if attempts >= LOGIN_MAX_ATTEMPTS:
        st.session_state["login_locked_until"] = time.time() + LOGIN_LOCK_SECONDS


def clear_login_attempts() -> None:
    st.session_state.pop("login_failed_attempts", None)
    st.session_state.pop("login_locked_until", None)


def get_current_plan(user: Dict) -> Dict:
    """Load the user's current plan from SQLite, with a safe Free fallback."""

    plan = get_user_plan_details(user_id=user["id"]) or get_subscription_plan("free")
    if not plan:
        raise RuntimeError("Subscription plans are not initialized.")
    return plan


def plan_price_label(plan: Dict) -> str:
    price = float(plan["price_usd"])
    if price <= 0:
        return "$0"
    if price.is_integer():
        return f"${int(price)}/{plan['billing_label']}"
    return f"${price:.2f}/{plan['billing_label']}"


def plan_marketing_label(plan: Dict) -> str:
    """Return a display label from database plan metadata, not hardcoded prices."""

    name = plan.get("plan_name", "")
    if name.lower() == "pro":
        return "Most Popular"
    if bool(plan.get("client_folders")):
        return "Agency Scale"
    if bool(plan.get("zip_export")):
        return "Creator Workflow"
    return "Simple Start"


def is_user_on_plan(user: Dict, plan: Dict) -> bool:
    current = get_current_plan(user)
    return current["plan_key"] == plan["plan_key"]


class DatabaseBackedPlans:
    """Compatibility adapter so older UI code reads plan data from SQLite."""

    def _plans_by_name(self) -> Dict[str, Dict]:
        plans = get_active_subscription_plans()
        return {plan["plan_name"]: plan for plan in plans}

    def keys(self):
        return list(self._plans_by_name().keys())

    def __getitem__(self, plan_value: str) -> Dict:
        plan = get_subscription_plan(plan_value)
        if plan:
            return plan
        plans = self._plans_by_name()
        if plan_value in plans:
            return plans[plan_value]
        return get_subscription_plan("free")


PLANS = DatabaseBackedPlans()

# Application constants
PRODUCT_NAME = "StockGuard AI"
PRODUCT_TAGLINE = "Premium AI review workspace for safer stock uploads."
PRODUCT_BYLINE = "A product by BEE CLUSTER."
LANDING_FOOTER_TEXT = "© 2026 StockGuard AI. A product by BEE CLUSTER."
FEEDBACK_EMAIL = "support@stockguard.ai"
ADMIN_EMAIL = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
REPORT_FOOTER_TEXT = "Generated by StockGuard AI for stock contributor review and export."
MAX_IMAGE_SIZE_MB = 25
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCK_SECONDS = 15 * 60
NEAR_DUPLICATE_PERCENT = 95.0
DEFAULT_SCAN_MODE = "Balanced Recommended"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
PROFILE_PHOTO_DIR = Path(__file__).resolve().parent / "profile_uploads"
ALLOWED_PROFILE_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_PROFILE_PHOTO_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_PROFILE_PHOTO_SIZE_BYTES = 2 * 1024 * 1024
MAX_AVATAR_SIZE = 512

# Logo data URI — loaded once, used across auth, sidebar, and favicon
_SG_LOGO_DATA_URI: str | None = None
def _sg_logo_data_uri() -> str:
    global _SG_LOGO_DATA_URI
    if _SG_LOGO_DATA_URI is None:
        try:
            with open("logo.png", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            _SG_LOGO_DATA_URI = f"data:image/png;base64,{b64}"
        except Exception:
            _SG_LOGO_DATA_URI = ""
    return _SG_LOGO_DATA_URI
SCAN_MODE_PRESETS = {
    "Broad Review": 75,
    "Balanced Recommended": 85,
    "Strict": 90,
    "Near Duplicate Only": 95,
}


def escape_html(value: object) -> str:
    """Safely render text for Streamlit HTML snippets."""

    return html.escape(str(value), quote=True)


def clean_icon(icon: object) -> str:
    """Normalize icon labels used across the UI cards."""

    text = str(icon or "").strip()
    return text or "✦"


def safe_mapping(value: object) -> Dict[str, Dict]:
    """Return a dictionary for UI code paths that may receive None or invalid data."""

    return value if isinstance(value, dict) else {}


def safe_sequence(value: object) -> List[Dict]:
    """Return a list for report helpers that may receive None or invalid data."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    return [value] if value else []


def cleanup_temporary_scan_files() -> None:
    """Clear scan data from session memory.

    Uploaded images are held in Streamlit memory/session only for the current scan
    so CSV/ZIP export can work. The app does not write uploaded image files to
    disk; scan history stores report rows only. No temp files exist on disk.
    """

    reset_scan_session_state()


def reset_scan_session_state(session_state: Dict | None = None) -> None:
    """Clear stale scan state so the user can start a fresh upload in the same tab."""

    target = st.session_state if session_state is None else session_state

    for key in ("last_scan_result", "active_scan_context", "scan_file_uploader"):
        target.pop(key, None)

    for key in list(target.keys()):
        if key.startswith("uploaded_") or key.startswith("scan_file_uploader"):
            target.pop(key, None)


def _avatar_initials(user: Dict) -> str:
    """Generate initials string from user display name, name, or email."""
    display = user.get("display_name") or user.get("name") or user.get("email", "SG")
    parts = display.strip().split()
    init = "".join(p[:1] for p in parts[:2]).upper()
    return init or "SG"


def _profile_photo_data_uri(user: Dict) -> str | None:
    """Return base64 data URI for the user's profile photo, or None."""
    path_str = user.get("profile_photo_path")
    if not path_str:
        return None
    photo_path = Path(path_str)
    if not photo_path.is_file():
        return None
    try:
        with open(photo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = photo_path.suffix.lower()
        mime = "image/jpeg"
        if ext in {".png"}:
            mime = "image/png"
        elif ext in {".webp"}:
            mime = "image/webp"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _render_avatar_html(user: Dict, size: int = 36, font_size: str | None = None) -> str:
    """Render an HTML img or initials span for the user's avatar."""
    photo_uri = _profile_photo_data_uri(user)
    fs = font_size or f"max(0.65rem, {size * 0.32}px)"
    if photo_uri:
        return f'<img class="sg-avatar" src="{photo_uri}" alt="Profile photo" style="width:{size}px; height:{size}px; object-fit:cover; border-radius:50%;">'
    initials = _avatar_initials(user)
    return f'<span class="sg-avatar" style="width:{size}px; height:{size}px; font-size:{fs};">{escape_html(initials)}</span>'


def validate_profile_photo(uploaded_file) -> str | None:
    """Validate an uploaded profile photo. Returns an error message or None."""
    if uploaded_file is None:
        return "No file provided."
    if uploaded_file.size > MAX_PROFILE_PHOTO_SIZE_BYTES:
        return f"File is too large ({uploaded_file.size / 1024 / 1024:.1f} MB). Maximum is 2 MB."
    name = (uploaded_file.name or "").lower()
    ext = Path(name).suffix
    if ext not in ALLOWED_PROFILE_PHOTO_EXTENSIONS:
        return f"File type '{ext}' is not allowed. Allowed: JPG, JPEG, PNG, WEBP."
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        image.verify()
    except Exception:
        return "File is corrupted or is not a valid image."
    return None


def crop_to_square(image: Image.Image, crop_mode: str) -> Image.Image:
    """Crop an image to a square based on the given mode, then resize to MAX_AVATAR_SIZE."""
    w, h = image.size
    side = min(w, h)
    if crop_mode == "center":
        left = (w - side) // 2
        top = (h - side) // 2
    elif crop_mode == "top":
        left = (w - side) // 2
        top = 0
    elif crop_mode == "bottom":
        left = (w - side) // 2
        top = h - side
    elif crop_mode == "left":
        left = 0
        top = (h - side) // 2
    elif crop_mode == "right":
        left = w - side
        top = (h - side) // 2
    else:
        left = (w - side) // 2
        top = (h - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((MAX_AVATAR_SIZE, MAX_AVATAR_SIZE), Image.LANCZOS)


def process_profile_photo(uploaded_file, crop_mode: str) -> bytes:
    """Validate, crop, resize, and return processed avatar bytes as optimized JPEG."""
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    if image.mode != "RGB":
        image = image.convert("RGB")
    avatar = crop_to_square(image, crop_mode)
    buf = io.BytesIO()
    avatar.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()


def save_profile_photo(user_id: int, image_bytes: bytes) -> str:
    """Save processed avatar bytes to profile_uploads/ and return the file path."""
    PROFILE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"user_{user_id}_{uuid.uuid4().hex}.jpg"
    file_path = PROFILE_PHOTO_DIR / filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return str(file_path.resolve())


def delete_profile_photo_file(user_id: int, user: Dict) -> None:
    """Delete the user's old profile photo if it exists inside profile_uploads."""
    old_path_str = user.get("profile_photo_path")
    if not old_path_str:
        return
    old_path = Path(old_path_str).resolve()
    profile_dir = PROFILE_PHOTO_DIR.resolve()
    if old_path.parent == profile_dir and old_path.is_file():
        try:
            old_path.unlink()
        except Exception:
            pass


@dataclass
class UploadedImage:
    """Small container for one valid uploaded image.

    The original file bytes are preserved for ZIP export. The thumbnail is used
    for Streamlit previews, and the 224px model image is used for ResNet50 so
    we do not keep feeding full-resolution files into the AI model.
    """

    name: str
    original_name: str
    image: Image.Image
    thumbnail: Image.Image
    model_image: Image.Image
    bytes_data: bytes
    original_width: int
    original_height: int
    processing_status: str = "OK"


class UnionFind:
    """Simple union-find structure for grouping connected similar images."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def apply_custom_styles() -> None:
    """Apply the safe global styling foundation for readable Streamlit UI."""

    st.markdown(
        """
        <style>
        html,
        body,
        #root,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        main,
        .block-container {
            background: #F7F9FC !important;
            color: #0F172A !important;
        }

        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1280px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            margin: 0 auto !important;
        }

        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            gap: 18px !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
        }

        .stMarkdown p,
        .stMarkdown li,
        .stTextInput label,
        .stSelectbox label,
        .stSlider label,
        .stRadio label,
        .stTextArea label {
            color: #475569 !important;
        }

        [data-testid="stMain"] p,
        [data-testid="stMain"] span,
        [data-testid="stMain"] div,
        [data-testid="stMain"] label {
            color: #0F172A !important;
        }

        [data-testid="stMain"] [data-testid="stDataFrame"] *,
        [data-testid="stMain"] [data-testid="stTable"] *,
        [data-testid="stMain"] [data-testid="stExpander"] *,
        [data-testid="stMain"] [data-testid="stFileUploader"] * {
            color: #0F172A !important;
            background-color: transparent !important;
        }

        [data-testid="stMain"] [data-testid="stDataFrame"],
        [data-testid="stMain"] [data-testid="stTable"],
        [data-testid="stMain"] [data-testid="stDataFrame"] > div,
        [data-testid="stMain"] [data-testid="stTable"] > div,
        [data-testid="stMain"] [data-testid="stExpander"] [data-testid="stDataFrame"] {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04) !important;
        }

        [data-testid="stMain"] [data-testid="stDataFrame"] table,
        [data-testid="stMain"] [data-testid="stTable"] table,
        [data-testid="stMain"] [data-testid="stDataFrame"] th,
        [data-testid="stMain"] [data-testid="stDataFrame"] td,
        [data-testid="stMain"] [data-testid="stTable"] th,
        [data-testid="stMain"] [data-testid="stTable"] td,
        [data-testid="stMain"] table,
        [data-testid="stMain"] thead,
        [data-testid="stMain"] tbody,
        [data-testid="stMain"] tr,
        [data-testid="stMain"] th,
        [data-testid="stMain"] td {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #E2E8F0 !important;
        }

        [data-testid="stMain"] th,
        [data-testid="stMain"] [role="columnheader"] {
            background: #F8FAFC !important;
            color: #334155 !important;
            font-weight: 700 !important;
        }

        [data-testid="stMain"] tr:hover td,
        [data-testid="stMain"] [data-testid="stDataFrame"] tr:hover td,
        [data-testid="stMain"] [data-testid="stTable"] tr:hover td {
            background: #F8FAFC !important;
        }

        [data-testid="stMain"] td,
        [data-testid="stMain"] [role="gridcell"] {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }

        [data-testid="stMain"] [data-testid="stExpander"] > div > button,
        [data-testid="stMain"] [data-testid="stExpander"] summary,
        [data-testid="stMain"] [data-testid="stExpander"] button {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 18px rgba(15, 23, 42, 0.04) !important;
            font-weight: 700 !important;
        }

        [data-testid="stMain"] [data-testid="stFileUploader"] section,
        [data-testid="stMain"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            background: #FFFFFF !important;
            border: 1px dashed #93C5FD !important;
            border-radius: 16px !important;
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.08) !important;
            color: #0F172A !important;
        }

        [data-testid="stMain"] [data-testid="stFileUploader"] label,
        [data-testid="stMain"] [data-testid="stFileUploader"] span,
        [data-testid="stMain"] [data-testid="stFileUploader"] p,
        [data-testid="stMain"] [data-testid="stFileUploader"] div,
        [data-testid="stMain"] [data-testid="stFileUploader"] li {
            color: #0F172A !important;
            background-color: transparent !important;
        }

        [data-testid="stMain"] [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid transparent !important;
        }

        [data-testid="stMain"] button[aria-label="Search"],
        [data-testid="stMain"] button[aria-label="Show/hide columns"],
        [data-testid="stMain"] button[aria-label="Download as CSV"],
        [data-testid="stMain"] button[kind="secondary"],
        [data-testid="stMain"] .stDownloadButton > button,
        [data-testid="stMain"] .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 18px rgba(15, 23, 42, 0.05) !important;
        }

        [data-testid="stMain"] div[data-testid="stTextInput"] div[data-baseweb="input"],
        [data-testid="stMain"] div[data-testid="stTextInput"] input,
        [data-testid="stMain"] div[data-testid="stTextArea"] textarea,
        [data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="input"],
        [data-testid="stMain"] div[data-testid="stSelectbox"] input,
        [data-testid="stMain"] div[data-baseweb="select"] > div,
        [data-testid="stMain"] div[data-baseweb="textarea"] textarea,
        [data-testid="stMain"] div[data-testid="stNumberInput"] div[data-baseweb="input"],
        [data-testid="stMain"] div[data-testid="stMultiSelect"] div[data-baseweb="input"] {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04) !important;
        }

        [data-testid="stFileUploader"] section {
            background: #FFFFFF !important;
            border: 1px dashed #93C5FD !important;
            border-radius: 16px !important;
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.08) !important;
        }

        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid transparent !important;
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stSelectbox"] div[data-baseweb="input"]:focus-within {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12) !important;
        }

        .sg-page,
        .sg-main,
        .sg-card,
        .sg-section-card,
        .sg-page-header,
        .sg-empty-state,
        .sg-table-card,
        .sg-metric-card,
        .sg-action-card,
        .locked-card {
            color: #0F172A !important;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06) !important;
        }

        .locked-card {
            text-align: center !important;
            padding: 2rem 1.5rem !important;
        }

        .locked-card .sg-badge {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 5px 13px !important;
            border-radius: 999px !important;
            border: 1px solid #BFDBFE !important;
            background: #EFF6FF !important;
            color: #2563EB !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            margin-bottom: 9px !important;
        }

        .locked-card .status-badge {
            display: inline-block !important;
            padding: 4px 11px !important;
            border-radius: 999px !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
        }

        .locked-card .status-badge.purple {
            color: #5B21B6 !important;
            background: #F3E8FF !important;
            border: 1px solid #D8B4FE !important;
        }

        .locked-card h2 {
            font-size: 1.1rem !important;
            margin: 0.5rem 0 0.35rem 0 !important;
        }

        .locked-card p {
            margin-bottom: 0.75rem !important;
        }

        .sg-page-subtitle,
        .sg-muted,
        .sg-card-subtitle {
            color: #64748B !important;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 12px !important;
            font-weight: 700 !important;
            border: 1px solid transparent !important;
            transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease, border-color 140ms ease !important;
        }

        .stButton > button[kind="secondary"],
        .stButton > button[kind="tertiary"],
        .stButton > button[data-kind="secondary"] {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 10px 18px rgba(15, 23, 42, 0.05) !important;
        }

        .stButton > button[kind="secondary"]:hover,
        .stButton > button[kind="tertiary"]:hover,
        .stButton > button[data-kind="secondary"]:hover {
            background: #F8FAFC !important;
            border-color: #94A3B8 !important;
            color: #111827 !important;
            transform: translateY(-1px) !important;
        }

        .stButton > button:not([kind="secondary"]):not([kind="tertiary"]),
        [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18) !important;
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            filter: brightness(1.03) !important;
            transform: translateY(-1px) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0F172A 100%) !important;
            padding: 0.75rem 0.75rem 1rem !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            scrollbar-width: thin !important;
        }

        [data-testid="stSidebar"] > div {
            padding: 0 !important;
        }

        [data-testid="stSidebar"] nav,
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 10px !important;
        }

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] .stButton > button {
            color: #E5E7EB !important;
        }

        [data-testid="stSidebar"] .sg-muted,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown span {
            color: #94A3B8 !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: #1F2937 !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 12px !important;
            min-height: 42px !important;
            height: 42px !important;
            padding: 0.45rem 0.65rem !important;
            margin-bottom: 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
            color: #FFFFFF !important;
            border-color: transparent !important;
            box-shadow: 0 10px 18px rgba(37, 99, 235, 0.18) !important;
        }

        [data-testid="stSidebar"] .stButton > button:focus {
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14) !important;
        }

        [data-testid="stSidebar"] .sg-nav-active {
            display: block !important;
            padding: 0.55rem 0.65rem !important;
            border-radius: 12px !important;
            margin: 0 !important;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(124, 58, 237, 0.18)) !important;
            border: 1px solid rgba(129, 140, 248, 0.45) !important;
            color: #EFF6FF !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 18px rgba(99, 102, 241, 0.12) !important;
        }

        [data-testid="stMain"] [data-testid="stDataFrame"],
        [data-testid="stMain"] [data-testid="stDataFrame"] > div,
        [data-testid="stMain"] [data-testid="stTable"],
        [data-testid="stMain"] [data-testid="stTable"] > div,
        [data-testid="stMain"] [data-testid="stDataFrame"] table,
        [data-testid="stMain"] [data-testid="stTable"] table,
        [data-testid="stMain"] .stDataFrame {
            border-radius: 14px !important;
            border: 1px solid #E2E8F0 !important;
            background: #FFFFFF !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04) !important;
            color: #0F172A !important;
        }

        [data-testid="stMain"] [data-testid="stDataFrame"] table,
        [data-testid="stMain"] [data-testid="stTable"] table {
            background: #FFFFFF !important;
            color: #0F172A !important;
        }

        [data-testid="stMain"] [data-testid="stDataFrame"] th,
        [data-testid="stMain"] [data-testid="stDataFrame"] td,
        [data-testid="stMain"] [data-testid="stTable"] th,
        [data-testid="stMain"] [data-testid="stTable"] td {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #E2E8F0 !important;
        }

        [data-testid="stDataFrame"] tr:hover td,
        [data-testid="stTable"] tr:hover td {
            background: #F8FAFC !important;
        }

        [data-testid="stExpander"] > div > button {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 18px rgba(15, 23, 42, 0.04) !important;
            font-weight: 700 !important;
        }

        [data-testid="stExpander"] > div > button:hover {
            background: #F8FAFC !important;
            color: #111827 !important;
        }

        [data-testid="stExpander"] [data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px !important;
            background: #FFFFFF !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04) !important;
        }

        .sg-light-table-shell {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto !important;
            background: #FFFFFF !important;
        }

        .sg-table-wrap {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto !important;
        }

        .sg-light-table-shell table,
        .sg-table-wrap table,
        .sg-light-table {
            width: 100% !important;
            max-width: 100% !important;
            table-layout: fixed !important;
        }

        .sg-cell-truncate,
        .sg-light-table-shell td.table-ellipsis,
        .sg-light-table-shell td.filename-cell,
        .sg-light-table-shell td.reason-cell,
        .sg-light-table td,
        .sg-light-table th {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            max-width: 220px !important;
        }

        .sg-light-table-shell td,
        .sg-table-wrap td {
            word-break: break-word !important;
        }

        .sg-filter-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 18px !important;
            padding: 0.9rem !important;
            box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06) !important;
        }

        .sg-upload-zone {
            border: 1px dashed #93C5FD !important;
            background: linear-gradient(135deg, #F8FBFF 0%, #EEF4FF 100%) !important;
            border-radius: 18px !important;
            box-shadow: 0 18px 30px rgba(37, 99, 235, 0.08) !important;
        }

        .sg-topbar {
            display: flex !important;
            align-items: flex-start !important;
            justify-content: space-between !important;
            flex-wrap: wrap !important;
            gap: 0.75rem 1rem !important;
            width: 100% !important;
            padding: 0.15rem 0 0.35rem 0 !important;
        }

        .sg-topbar-title {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            color: #0F172A !important;
        }

        .sg-topbar-subtitle {
            color: #64748B !important;
            font-size: 0.92rem !important;
            line-height: 1.35 !important;
            max-width: 42rem !important;
        }

        .sg-topbar-right {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
            flex-wrap: wrap !important;
            gap: 0.35rem 0.5rem !important;
            row-gap: 0.35rem !important;
        }

        .sg-pill {
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.35rem !important;
            border-radius: 999px !important;
            padding: 0.28rem 0.55rem !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em !important;
            white-space: nowrap !important;
        }

        .sg-pill-success {
            color: #166534 !important;
            background: #ECFDF5 !important;
            border: 1px solid #A7F3D0 !important;
        }

        .sg-pill-muted {
            color: #334155 !important;
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
        }

        .sg-footer {
            border-top: 1px solid #E2E8F0 !important;
            margin-top: 32px !important;
            padding: 0.9rem 0.15rem 24px 0.15rem !important;
            color: #64748B !important;
            font-size: 0.92rem !important;
            line-height: 1.5 !important;
            text-align: left !important;
            display: block !important;
        }

        .sg-footer > div {
            text-align: left !important;
        }

        .status-card,
        .sg-metric-card,
        .sg-action-card,
        .sg-saas-card,
        .sg-card,
        .sg-section-card,
        .sg-empty-state,
        .locked-card,
        .admin-metric-card,
        .admin-user-card,
        .admin-plan-card,
        .admin-hero {
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            background: #FFFFFF !important;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06) !important;
            display: flex !important;
            flex-direction: column !important;
        }

        .status-card .label,
        .sg-metric-card .sg-metric-label,
        .sg-card-title {
            color: #0F172A !important;
            font-weight: 700 !important;
        }

        .status-card .value,
        .sg-metric-card .sg-metric-value {
            color: #111827 !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
        }

        .status-card .hint,
        .sg-metric-card .sg-muted,
        .sg-muted {
            color: #64748B !important;
            line-height: 1.4 !important;
        }

        .sg-metric-card {
            padding: 1rem !important;
            min-height: 132px !important;
            border-radius: 16px !important;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06) !important;
        }

        .admin-metric-card {
            padding: 1rem !important;
            min-height: 138px !important;
            border-radius: 16px !important;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06) !important;
        }

        .admin-metric-icon {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 2rem !important;
            height: 2rem !important;
            border-radius: 999px !important;
            background: linear-gradient(135deg, #EFF6FF, #F5F3FF) !important;
            color: #2563EB !important;
            font-size: 0.95rem !important;
            font-weight: 800 !important;
            border: 1px solid #E2E8F0 !important;
            margin-bottom: 0.45rem !important;
        }

        .admin-metric-label,
        .admin-metric-value,
        .admin-metric-desc {
            color: #0F172A !important;
        }

        .admin-metric-label {
            font-size: 0.86rem !important;
            text-transform: none !important;
            letter-spacing: 0.01em !important;
            color: #475569 !important;
        }

        .admin-metric-value {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            margin-top: 0.15rem !important;
            line-height: 1.2 !important;
        }

        .admin-metric-desc {
            color: #64748B !important;
            font-size: 0.92rem !important;
            line-height: 1.35 !important;
            margin-top: 0.2rem !important;
        }

        .admin-section-title {
            color: #111827 !important;
            font-size: 0.98rem !important;
            font-weight: 800 !important;
            margin: 1rem 0 0.35rem !important;
        }

        .admin-hero {
            padding: 1rem !important;
            border-radius: 18px !important;
            margin-bottom: 0.75rem !important;
        }

        .sg-metric-icon,
        .sg-action-badge {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-width: 2rem !important;
            height: 2rem !important;
            padding: 0 0.35rem !important;
            border-radius: 999px !important;
            background: linear-gradient(135deg, #EFF6FF, #F5F3FF) !important;
            color: #2563EB !important;
            font-weight: 800 !important;
            border: 1px solid #E2E8F0 !important;
            margin-bottom: 0.35rem !important;
        }

        .sg-action-card {
            padding: 1rem !important;
            border-radius: 16px !important;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06) !important;
            height: 100% !important;
        }

        .sg-section-card {
            padding: 1rem !important;
            border-radius: 18px !important;
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06) !important;
        }

        .sg-table-wrap {
            overflow-x: auto !important;
            max-width: 100% !important;
            border-radius: 16px !important;
            border: 1px solid #E2E8F0 !important;
            background: #FFFFFF !important;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06) !important;
        }

        .sg-light-table {
            width: 100% !important;
            border-collapse: collapse !important;
            min-width: 860px !important;
            background: #FFFFFF !important;
            color: #0F172A !important;
        }

        .sg-light-table th,
        .sg-light-table td {
            padding: 0.75rem 0.85rem !important;
            border-bottom: 1px solid #E2E8F0 !important;
            text-align: left !important;
            vertical-align: middle !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            max-width: 260px !important;
        }

        .sg-light-table th {
            background: #F8FAFC !important;
            color: #334155 !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.04em !important;
            text-transform: uppercase !important;
            font-weight: 700 !important;
        }

        .sg-light-table tr:hover td {
            background: #F8FAFC !important;
        }

        .sg-scan-mode-card {
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            background: #FFFFFF !important;
            padding: 0.9rem !important;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04) !important;
        }

        .sg-scan-mode-card.active {
            border-color: #2563EB !important;
            background: linear-gradient(135deg, rgba(239, 246, 255, 0.96), rgba(245, 243, 255, 0.96)) !important;
            box-shadow: 0 16px 28px rgba(37, 99, 235, 0.14) !important;
        }

        .sg-empty-state .icon,
        .status-card .icon,
        .feature-icon {
            box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.18) !important;
        }

        header[data-testid="stHeader"] {
            display: none !important;
        }

        [data-testid="stToolbar"] {
            display: none !important;
        }

        #MainMenu {
            visibility: hidden !important;
        }

        /* Hide only the Streamlit sidebar collapse/expand control without hiding the sidebar */
        /* Targets known Streamlit control selectors and aria/title variants. */
        button[kind="header"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarNavCollapseButton"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            background: transparent !important;
        }

        /* aria-label / title variants used by some Streamlit versions */
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"],
        button[title="Close sidebar"],
        button[title="Open sidebar"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Narrow the rule to only affect the control inside the sidebar container */
        [data-testid="stSidebar"] button[aria-label="Close sidebar"],
        [data-testid="stSidebar"] button[aria-label="Open sidebar"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Final polish: pill variants */
        .sg-pill { border-radius: 999px !important; padding: 4px 10px !important; font-size: 12px !important; font-weight: 700 !important; }
        .sg-pill-success { color: #065f46 !important; background: #ecfdf5 !important; border: 1px solid #bbf7d0 !important; }
        .sg-pill-warning { color: #92400e !important; background: #fffbeb !important; border: 1px solid #fef3c7 !important; }
        .sg-pill-danger { color: #7f1d1d !important; background: #fff1f2 !important; border: 1px solid #fecaca !important; }
        .sg-pill-pro { color: #4c1d95 !important; background: #f5f3ff !important; border: 1px solid #e9d5ff !important; }
        .sg-pill-free { color: #374151 !important; background: #f8fafc !important; border: 1px solid #e6eef8 !important; }

        /* Generic status badges (outside auth context) */
        .status-badge {
            display: inline-block !important;
            padding: 3px 10px !important;
            border-radius: 999px !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            line-height: 1.4 !important;
        }
        .status-badge.purple { color: #5B21B6 !important; background: #F3E8FF !important; border: 1px solid #D8B4FE !important; }
        .status-badge.blue { color: #1E40AF !important; background: #DBEAFE !important; border: 1px solid #93C5FD !important; }
        .status-badge.green { color: #065F46 !important; background: #D1FAE5 !important; border: 1px solid #6EE7B7 !important; }
        .status-badge.red { color: #991B1B !important; background: #FEE2E2 !important; border: 1px solid #FCA5A5 !important; }
        .status-badge.yellow { color: #92400E !important; background: #FEF3C7 !important; border: 1px solid #FCD34D !important; }

        /* Buttons: main content primary/secondary/danger styles */
        [data-testid="stMain"] .stButton > button,
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button {
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.5rem 0.9rem !important;
            transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease, border-color 140ms ease !important;
        }

        [data-testid="stMain"] .stButton > button:not([kind="secondary"]):not([kind="tertiary"]),
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:not([kind="secondary"]) {
            background: linear-gradient(135deg, #5B21B6 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.18) !important;
            border: 1px solid transparent !important;
        }

        [data-testid="stMain"] .stButton > button[kind="secondary"],
        [data-testid="stMain"] .stButton > button[data-kind="secondary"],
        [data-testid="stMain"] .stButton > button[kind="tertiary"] {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05) !important;
        }

        [data-testid="stMain"] .stButton > button[kind="danger"] {
            background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid transparent !important;
            box-shadow: 0 8px 18px rgba(220, 38, 38, 0.18) !important;
        }

        /* Dashboard and admin spacing consistency */
        [data-testid="stMainBlockContainer"] {
            width: 100% !important;
            max-width: 1280px !important;
            box-sizing: border-box !important;
            padding-left: clamp(1rem, 2vw, 1.65rem) !important;
            padding-right: clamp(1rem, 2vw, 1.65rem) !important;
            overflow-x: clip !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            overflow-x: hidden !important;
        }

        [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
            gap: 1rem !important;
            align-items: stretch !important;
        }

        [data-testid="stMain"] [data-testid="column"] > div {
            min-width: 0 !important;
        }

        .sg-page-header {
            padding: 1.3rem 1.4rem !important;
            margin-bottom: 0.2rem !important;
            border-radius: 20px !important;
        }

        .sg-page-title {
            margin-bottom: 0.3rem !important;
            line-height: 1.15 !important;
        }

        .sg-page-subtitle {
            line-height: 1.5 !important;
            max-width: 52rem !important;
        }

        .sg-card,
        .sg-section-card,
        .sg-metric-card,
        .sg-action-card,
        .sg-page,
        .locked-card,
        .admin-metric-card,
        .admin-plan-card,
        .admin-user-card,
        .admin-hero {
            box-sizing: border-box !important;
            border-radius: 18px !important;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06) !important;
        }

        .sg-card,
        .sg-section-card,
        .locked-card {
            gap: 0.35rem !important;
            padding: 1.15rem 1.25rem !important;
        }

        .sg-section-card {
            margin: 0.1rem 0 0.15rem !important;
        }

        .sg-metric-card,
        .admin-metric-card {
            gap: 0.4rem !important;
            min-height: 146px !important;
            padding: 1.1rem 1.15rem !important;
            justify-content: flex-start !important;
        }

        .sg-metric-card .sg-muted,
        .admin-metric-desc {
            margin-top: 0.15rem !important;
        }

        .sg-action-card {
            gap: 0.45rem !important;
            min-height: 168px !important;
            padding: 1.15rem 1.2rem !important;
        }

        .sg-metric-icon,
        .sg-action-badge,
        .admin-metric-icon {
            margin-bottom: 0.25rem !important;
            flex: 0 0 auto !important;
        }

        .admin-hero {
            padding: 1.3rem 1.4rem !important;
            margin-bottom: 1rem !important;
            gap: 0.45rem !important;
        }

        .admin-hero h2 {
            margin: 0 !important;
            line-height: 1.2 !important;
        }

        .admin-section-title {
            margin: 1.35rem 0 0.55rem !important;
        }

        .admin-plan-card {
            min-height: 230px !important;
            padding: 1.15rem !important;
            gap: 0.45rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.admin-user-card) {
            padding: 1.15rem !important;
            margin-bottom: 0.9rem !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 18px !important;
            background: #FFFFFF !important;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.admin-user-card) > div {
            gap: 0.8rem !important;
        }

        .admin-user-card {
            height: auto !important;
            padding: 0 !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            gap: 0.8rem !important;
        }

        .admin-user-top,
        .admin-stat-row {
            display: flex !important;
            align-items: flex-start !important;
            justify-content: space-between !important;
            flex-wrap: wrap !important;
            gap: 0.75rem 1rem !important;
        }

        .admin-stat-row {
            padding-top: 0.8rem !important;
            border-top: 1px solid #E2E8F0 !important;
        }

        .admin-stat {
            min-width: 110px !important;
            flex: 1 1 120px !important;
        }

        [data-testid="stSidebar"] {
            height: 100vh !important;
            padding: 0 !important;
            overflow-x: hidden !important;
            overflow-y: hidden !important;
            scrollbar-width: thin !important;
            scrollbar-color: rgba(129, 140, 248, 0.55) transparent !important;
        }

        [data-testid="stSidebar"] > div {
            height: 100% !important;
            padding: 0 !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            display: none !important;
            visibility: hidden !important;
            flex: 0 0 0 !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: 0 !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            box-sizing: border-box !important;
            height: 100% !important;
            display: block !important;
            padding: 0 !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            overscroll-behavior: contain !important;
            scrollbar-gutter: stable !important;
            scrollbar-width: thin !important;
            scrollbar-color: rgba(129, 140, 248, 0.48) transparent !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            box-sizing: border-box !important;
            min-height: 100% !important;
            padding: 16px 0.75rem 1rem !important;
            margin: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div:first-child,
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div:first-child > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] .element-container:first-child,
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.25rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
            width: 4px !important;
            height: 4px !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
            background: transparent !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
            background: rgba(129, 140, 248, 0.34) !important;
            border: none !important;
            border-radius: 999px !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {
            background: rgba(96, 165, 250, 0.55) !important;
        }

        .sg-sidebar-logo {
            display: flex !important;
            align-items: center !important;
            gap: 0.65rem !important;
            padding: 0 0.3rem !important;
        }

        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]:has(.sg-sidebar-logo) {
            margin: 0 0 12px !important;
            padding: 0 !important;
        }

        .sg-brand-stack {
            min-width: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 0.12rem !important;
        }

        .sg-brand-mark {
            width: 2.55rem !important;
            height: 2.55rem !important;
            min-width: 2.55rem !important;
            border-radius: 13px !important;
            object-fit: contain !important;
            display: block !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.24) !important;
        }

        .sg-brand-name {
            color: #F8FAFC !important;
            font-size: 0.98rem !important;
            font-weight: 850 !important;
            line-height: 1.15 !important;
        }

        .sg-sidebar-product {
            color: #CBD5E1 !important;
            font-size: 0.74rem !important;
            font-weight: 650 !important;
            line-height: 1.25 !important;
        }

        .sg-sidebar-company {
            color: #7DD3FC !important;
            font-size: 0.66rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
            letter-spacing: 0.045em !important;
        }

        .sg-avatar {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 36px !important;
            height: 36px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            line-height: 1 !important;
            flex-shrink: 0 !important;
            overflow: hidden !important;
            object-fit: cover !important;
        }

        .sg-sidebar-user {
            display: flex !important;
            align-items: center !important;
            gap: 0.65rem !important;
            padding: 0.65rem 0.75rem !important;
            margin: 0.4rem 0 0.1rem !important;
            border-top: 1px solid rgba(129, 140, 248, 0.12) !important;
            border-bottom: 1px solid rgba(129, 140, 248, 0.12) !important;
        }

        .sg-sidebar-user-info {
            display: flex !important;
            flex-direction: column !important;
            min-width: 0 !important;
        }

        .sg-sidebar-user-name {
            color: #E2E8F0 !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        .st-key-sidebar_plan_summary {
            box-sizing: border-box !important;
            padding: 0.82rem 0.85rem 0.95rem !important;
            margin: 0 0 0.1rem !important;
            border: 1px solid rgba(129, 140, 248, 0.18) !important;
            border-radius: 15px !important;
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.82), rgba(30, 41, 59, 0.58)) !important;
            box-shadow: 0 12px 26px rgba(2, 6, 23, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.025) !important;
            overflow: visible !important;
        }

        .st-key-sidebar_plan_summary > div {
            gap: 0.45rem !important;
        }

        .sg-usage-card {
            padding: 0 0 9px !important;
            margin: 0 !important;
            border: none !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        .sg-usage-header {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0.6rem !important;
        }

        .sg-usage-plan {
            color: #F8FAFC !important;
            font-size: 0.92rem !important;
            font-weight: 850 !important;
            line-height: 1.2 !important;
        }

        .sg-usage-value {
            margin-top: 0.48rem !important;
            margin-bottom: 0.08rem !important;
            color: #FFFFFF !important;
            font-size: 1rem !important;
            font-weight: 850 !important;
            line-height: 1.2 !important;
        }

        .sg-usage-meta {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0.5rem !important;
            margin-top: 0.3rem !important;
            margin-bottom: 0.28rem !important;
            color: #94A3B8 !important;
            font-size: 0.73rem !important;
            line-height: 1.3 !important;
        }

        .sg-usage-meta span {
            min-width: 0 !important;
            white-space: nowrap !important;
        }

        .sg-usage-meta span:last-child {
            margin-left: auto !important;
            text-align: right !important;
        }

        .sg-active-status {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0.22rem 0.58rem !important;
            border: 1px solid rgba(16, 185, 129, 0.35) !important;
            border-radius: 999px !important;
            background: rgba(16, 185, 129, 0.12) !important;
            color: #A7F3D0 !important;
            font-size: 0.7rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.03em !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
        }

        .sg-nav-section-title {
            display: block !important;
            position: relative !important;
            z-index: 2 !important;
            min-height: 16px !important;
            margin: 0 !important;
            padding: 0 0.35rem !important;
            color: #94A3B8 !important;
            font-size: 0.67rem !important;
            font-weight: 800 !important;
            line-height: 16px !important;
            letter-spacing: 0.11em !important;
            text-transform: uppercase !important;
            overflow: visible !important;
        }

        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]:has(.sg-nav-section-title) {
            position: relative !important;
            z-index: 2 !important;
            min-height: 16px !important;
            margin: 0.78rem 0 0.32rem !important;
            padding: 0 !important;
            overflow: visible !important;
        }

        [data-testid="stSidebar"] .stButton {
            margin: 0 0 0.18rem !important;
            position: relative !important;
            z-index: 0 !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            box-sizing: border-box !important;
            min-height: 35px !important;
            height: 35px !important;
            padding: 0.35rem 0.65rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            border-radius: 10px !important;
            background: rgba(30, 41, 59, 0.68) !important;
            border-color: rgba(148, 163, 184, 0.14) !important;
            box-shadow: none !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(51, 65, 85, 0.9) !important;
            border-color: rgba(96, 165, 250, 0.42) !important;
            box-shadow: 0 7px 16px rgba(2, 6, 23, 0.14) !important;
            transform: none !important;
        }

        [data-testid="stSidebar"] .stButton > button:focus-visible {
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.16) !important;
        }

        [data-testid="stSidebar"] .sg-nav-active {
            box-sizing: border-box !important;
            position: relative !important;
            z-index: 0 !important;
            width: 100% !important;
            min-height: 35px !important;
            height: 35px !important;
            margin: 0 0 0.18rem !important;
            padding: 0.35rem 0.65rem !important;
            display: flex !important;
            align-items: center !important;
            border-radius: 10px !important;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.24), rgba(124, 58, 237, 0.18)) !important;
            border: 1px solid rgba(129, 140, 248, 0.38) !important;
            color: #F8FAFC !important;
            font-size: 0.88rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            overflow: hidden !important;
            isolation: isolate !important;
            box-shadow: inset 3px 0 0 #60A5FA, inset 0 0 0 1px rgba(96, 165, 250, 0.05) !important;
        }

        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]:has(.sg-nav-active) {
            margin: 0 !important;
            overflow: hidden !important;
            border-radius: 10px !important;
        }

        .st-key-sidebar_logout_action {
            margin-top: 0.82rem !important;
            padding-top: 0.72rem !important;
            border-top: 1px solid rgba(148, 163, 184, 0.14) !important;
        }

        .st-key-sidebar_logout_action .stButton {
            margin-bottom: 0 !important;
        }

        .st-key-sidebar_logout_action .stButton > button {
            width: 100% !important;
            min-height: 35px !important;
            height: 35px !important;
            background: rgba(239, 68, 68, 0.10) !important;
            color: #FCA5A5 !important;
            border: 1px solid rgba(248, 113, 113, 0.35) !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            font-weight: 700 !important;
        }

        .st-key-sidebar_logout_action .stButton > button:hover {
            background: rgba(239, 68, 68, 0.18) !important;
            color: #FECACA !important;
            border-color: rgba(248, 113, 113, 0.55) !important;
            box-shadow: 0 8px 18px rgba(127, 29, 29, 0.16) !important;
            transform: none !important;
        }

        .st-key-sidebar_logout_action .stButton > button:focus-visible {
            box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.16) !important;
        }

        .sg-sidebar-footer {
            padding: 0.65rem 0.35rem 0.75rem !important;
            color: rgba(148, 163, 184, 0.76) !important;
            font-size: 0.72rem !important;
            font-weight: 550 !important;
            line-height: 1.4 !important;
            letter-spacing: 0.02em !important;
            text-align: left !important;
        }

        .sg-sidebar-footer .sg-sidebar-footer-brand {
            color: rgba(191, 219, 254, 0.92) !important;
            font-weight: 800 !important;
            letter-spacing: 0.055em !important;
        }

        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]:has(.sg-sidebar-footer) {
            margin: 0 !important;
            padding: 0 !important;
        }

        .sg-usage-progress {
            width: 100% !important;
            height: 7px !important;
            margin-top: 0.38rem !important;
            margin-bottom: 0 !important;
            border-radius: 999px !important;
            overflow: hidden !important;
            background: rgba(148, 163, 184, 0.18) !important;
            box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.12) !important;
            flex: 0 0 auto !important;
        }

        .sg-usage-progress-fill {
            height: 100% !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, #6366F1 0%, #7C3AED 100%) !important;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.35) !important;
            transition: width 180ms ease !important;
        }

        /* Scan result export actions */
        .st-key-result_export_actions {
            margin: 0.85rem 0 1.1rem !important;
            padding: 1.15rem 1.2rem 0.95rem !important;
            border: 1px solid rgba(99, 102, 241, 0.24) !important;
            border-radius: 20px !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #F7F8FF 52%, #F5F3FF 100%) !important;
            box-shadow: 0 18px 38px rgba(79, 70, 229, 0.11) !important;
            overflow: hidden !important;
        }

        .st-key-result_export_actions .sg-export-actions-heading {
            margin-bottom: 0.9rem !important;
        }

        .st-key-result_export_actions .sg-export-actions-heading h3 {
            margin: 0 0 0.22rem !important;
            color: #111827 !important;
            font-size: 1.08rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.015em !important;
        }

        .st-key-result_export_actions .sg-export-actions-heading p {
            margin: 0 !important;
            color: #64748B !important;
            font-size: 0.91rem !important;
            line-height: 1.45 !important;
        }

        .st-key-result_export_actions [data-testid="stHorizontalBlock"] {
            gap: 0.8rem !important;
            align-items: flex-start !important;
        }

        .st-key-result_export_actions .st-key-download_clean_zip_primary button {
            min-height: 46px !important;
            background: linear-gradient(135deg, #2563EB 0%, #6366F1 48%, #7C3AED 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(79, 70, 229, 0.28) !important;
            border-radius: 13px !important;
            box-shadow: 0 12px 24px rgba(79, 70, 229, 0.24) !important;
            font-weight: 800 !important;
        }

        .st-key-result_export_actions .st-key-download_clean_zip_primary button:hover:not(:disabled) {
            background: linear-gradient(135deg, #1D4ED8 0%, #4F46E5 48%, #6D28D9 100%) !important;
            border-color: rgba(79, 70, 229, 0.5) !important;
            box-shadow: 0 15px 28px rgba(79, 70, 229, 0.3) !important;
            transform: translateY(-1px) !important;
        }

        .st-key-result_export_actions .st-key-download_csv_report_primary button,
        .st-key-result_export_actions .st-key-start_new_scan_primary_actions button {
            min-height: 46px !important;
            border-radius: 13px !important;
            font-weight: 750 !important;
        }

        .st-key-result_export_actions .st-key-download_csv_report_primary button {
            background: #FFFFFF !important;
            color: #3730A3 !important;
            border: 1px solid rgba(99, 102, 241, 0.34) !important;
            box-shadow: 0 10px 20px rgba(79, 70, 229, 0.08) !important;
        }

        .st-key-result_export_actions .st-key-start_new_scan_primary_actions button {
            background: rgba(255, 255, 255, 0.62) !important;
            color: #334155 !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: none !important;
        }

        .st-key-result_export_actions [data-testid="stCaptionContainer"] {
            min-height: 2.45rem !important;
            margin-top: 0.1rem !important;
            color: #64748B !important;
            line-height: 1.35 !important;
        }

        /* Final shared premium polish layer */
        [data-testid="stMainBlockContainer"] {
            padding-top: clamp(1rem, 2vh, 1.45rem) !important;
            padding-bottom: 2.6rem !important;
        }

        [data-testid="stMain"] h1 {
            font-size: clamp(1.8rem, 3vw, 2.35rem) !important;
            line-height: 1.12 !important;
            letter-spacing: -0.035em !important;
        }

        [data-testid="stMain"] h2 {
            font-size: clamp(1.3rem, 2vw, 1.65rem) !important;
            line-height: 1.2 !important;
            letter-spacing: -0.02em !important;
            margin-top: 0.4rem !important;
        }

        [data-testid="stMain"] h3,
        .sg-card-title {
            line-height: 1.3 !important;
            letter-spacing: -0.012em !important;
        }

        .sg-page-header,
        .sg-saas-card,
        .sg-card,
        .sg-section-card,
        .sg-empty-state,
        .locked-card,
        .sg-metric-card,
        .sg-action-card,
        .status-card,
        .admin-hero,
        .admin-metric-card,
        .admin-user-card,
        .admin-plan-card {
            border-color: rgba(148, 163, 184, 0.28) !important;
            border-radius: 18px !important;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.055) !important;
        }

        .sg-page-header {
            padding: 1.25rem 1.35rem !important;
            margin-bottom: 0.45rem !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%) !important;
        }

        .sg-saas-card,
        .sg-card,
        .sg-section-card,
        .sg-empty-state,
        .locked-card {
            padding: 1.1rem 1.2rem !important;
        }

        .sg-page-subtitle,
        .sg-card-subtitle,
        .sg-muted,
        [data-testid="stCaptionContainer"] {
            color: #64748B !important;
            line-height: 1.5 !important;
        }

        [data-testid="stMain"] .stButton > button,
        [data-testid="stMain"] .stDownloadButton > button,
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button {
            min-height: 42px !important;
            border-radius: 12px !important;
            padding: 0.52rem 0.9rem !important;
            font-weight: 750 !important;
            letter-spacing: -0.005em !important;
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background 140ms ease !important;
        }

        [data-testid="stMain"] .stDownloadButton > button[kind="primary"],
        [data-testid="stMain"] .stButton > button[kind="primary"],
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #6366F1 50%, #7C3AED 100%) !important;
            color: #FFFFFF !important;
            border-color: transparent !important;
            box-shadow: 0 11px 24px rgba(79, 70, 229, 0.2) !important;
        }

        [data-testid="stMain"] .stButton > button:hover:not(:disabled),
        [data-testid="stMain"] .stDownloadButton > button:hover:not(:disabled),
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:hover:not(:disabled) {
            transform: translateY(-1px) !important;
            border-color: rgba(99, 102, 241, 0.45) !important;
            box-shadow: 0 13px 26px rgba(79, 70, 229, 0.15) !important;
        }

        [data-testid="stMain"] button:focus-visible,
        [data-testid="stMain"] input:focus-visible,
        [data-testid="stMain"] textarea:focus-visible {
            outline: none !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.14) !important;
        }

        [data-testid="stMain"] [data-testid="stTabs"] {
            margin-top: 0.35rem !important;
        }

        [data-testid="stMain"] [data-testid="stTabs"] [role="tablist"] {
            gap: 0.35rem !important;
            padding: 0.3rem 0.35rem 0 !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }

        [data-testid="stMain"] [data-testid="stTabs"] [role="tab"] {
            min-height: 42px !important;
            padding: 0.55rem 0.75rem 0.65rem !important;
            color: #64748B !important;
            font-weight: 700 !important;
            border-radius: 10px 10px 0 0 !important;
        }

        [data-testid="stMain"] [data-testid="stTabs"] [role="tab"]:hover {
            color: #4338CA !important;
            background: rgba(238, 242, 255, 0.72) !important;
        }

        [data-testid="stMain"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: #3730A3 !important;
            background: linear-gradient(180deg, rgba(238, 242, 255, 0.82), rgba(255, 255, 255, 0)) !important;
        }

        [data-testid="stMain"] [data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
            height: 3px !important;
            border-radius: 999px 999px 0 0 !important;
            background: linear-gradient(90deg, #2563EB, #7C3AED) !important;
        }

        [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
            min-height: 132px !important;
            padding: 1.15rem !important;
            border-color: rgba(96, 165, 250, 0.72) !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%) !important;
        }

        [data-testid="stMain"] [data-testid="stExpander"] details {
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            border-radius: 14px !important;
            background: #FFFFFF !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04) !important;
            overflow: hidden !important;
        }

        [data-testid="stMain"] [data-testid="stAlert"] {
            border-radius: 14px !important;
            border-width: 1px !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035) !important;
        }

        .sg-empty-state {
            min-height: 170px !important;
            justify-content: center !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #FAFBFF 100%) !important;
        }

        .sg-footer {
            margin-top: 2rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1.25rem !important;
            border-top-color: rgba(148, 163, 184, 0.3) !important;
        }

        /* Admin panel segmented tab navigation */
        .st-key-admin_panel_tabs[data-testid="stTabs"],
        .st-key-admin_panel_tabs [data-testid="stTabs"] {
            position: relative !important;
            max-width: 100% !important;
            margin: 0.75rem 0 1rem !important;
            padding: 0.34rem !important;
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            border-radius: 16px !important;
            background: linear-gradient(135deg, rgba(248, 250, 252, 0.98), rgba(238, 242, 255, 0.82)) !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055) !important;
            overflow: hidden !important;
        }

        .st-key-admin_panel_tabs [data-baseweb="tab-list"] {
            gap: 0.28rem !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            background: transparent !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            scrollbar-width: thin !important;
            scrollbar-color: rgba(99, 102, 241, 0.26) transparent !important;
            -webkit-overflow-scrolling: touch !important;
        }

        .st-key-admin_panel_tabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            height: 4px !important;
        }

        .st-key-admin_panel_tabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
            background: transparent !important;
        }

        .st-key-admin_panel_tabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
            border-radius: 999px !important;
            background: rgba(99, 102, 241, 0.25) !important;
        }

        .st-key-admin_panel_tabs [role="tab"] {
            flex: 0 0 auto !important;
            min-width: max-content !important;
            min-height: 42px !important;
            padding: 0.56rem 0.9rem !important;
            border: 1px solid transparent !important;
            border-radius: 11px !important;
            background: transparent !important;
            color: #64748B !important;
            font-size: 0.89rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            transition: color 140ms ease, background 140ms ease, border-color 140ms ease, box-shadow 140ms ease !important;
        }

        .st-key-admin_panel_tabs [role="tab"]:hover {
            color: #4338CA !important;
            background: rgba(255, 255, 255, 0.78) !important;
            border-color: rgba(129, 140, 248, 0.18) !important;
        }

        .st-key-admin_panel_tabs [role="tab"][aria-selected="true"] {
            color: #312E81 !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #EEF2FF 100%) !important;
            border-color: rgba(99, 102, 241, 0.28) !important;
            box-shadow: 0 7px 18px rgba(79, 70, 229, 0.12), inset 0 -2px 0 #6366F1 !important;
        }

        .st-key-admin_panel_tabs [data-baseweb="tab-highlight"],
        .st-key-admin_panel_tabs [data-baseweb="tab-border"] {
            display: none !important;
            height: 0 !important;
            background: transparent !important;
        }

        .st-key-admin_panel_tabs [data-testid="stTabsScrollLeft"],
        .st-key-admin_panel_tabs [data-testid="stTabsScrollRight"] {
            top: 0.34rem !important;
            bottom: auto !important;
            width: 38px !important;
            height: 42px !important;
            min-width: 38px !important;
            border: 1px solid rgba(99, 102, 241, 0.24) !important;
            border-radius: 11px !important;
            background: rgba(255, 255, 255, 0.96) !important;
            color: #4F46E5 !important;
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.10) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
        }

        .st-key-admin_panel_tabs [data-testid="stTabsScrollLeft"]:hover,
        .st-key-admin_panel_tabs [data-testid="stTabsScrollRight"]:hover {
            background: #EEF2FF !important;
            border-color: rgba(99, 102, 241, 0.42) !important;
            color: #3730A3 !important;
        }

        .st-key-admin_panel_tabs [data-testid="stTabsScrollLeft"] svg,
        .st-key-admin_panel_tabs [data-testid="stTabsScrollRight"] svg {
            width: 20px !important;
            height: 20px !important;
            color: currentColor !important;
            fill: currentColor !important;
        }

        .st-key-admin_panel_tabs [role="tabpanel"] {
            padding-top: 1.05rem !important;
        }

        /* Subscription pricing cards */
        .st-key-subscription_payment_options {
            margin: 0.8rem 0 1.15rem !important;
            padding: 1.15rem 1.2rem 1.2rem !important;
            border: 1px solid rgba(99, 102, 241, 0.20) !important;
            border-radius: 20px !important;
            background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFF 72%, #F5F3FF 100%) !important;
            box-shadow: 0 15px 34px rgba(37, 99, 235, 0.07) !important;
        }

        .st-key-subscription_payment_options > div[data-testid="stVerticalBlock"] {
            gap: 0.8rem !important;
        }

        .subscription-payment-heading {
            display: flex !important;
            align-items: flex-start !important;
            gap: 0.75rem !important;
        }

        .subscription-payment-icon {
            display: grid !important;
            place-items: center !important;
            flex: 0 0 38px !important;
            width: 38px !important;
            height: 38px !important;
            border: 1px solid rgba(99, 102, 241, 0.24) !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.11), rgba(124, 58, 237, 0.12)) !important;
            color: #4F46E5 !important;
            font-size: 1rem !important;
            font-weight: 900 !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        }

        .subscription-payment-title {
            margin: 0 !important;
            color: #0F172A !important;
            font-size: 1rem !important;
            font-weight: 850 !important;
            line-height: 1.25 !important;
            letter-spacing: -0.015em !important;
        }

        .subscription-payment-helper {
            margin: 0.22rem 0 0 !important;
            color: #64748B !important;
            font-size: 0.82rem !important;
            line-height: 1.45 !important;
        }

        .st-key-subscription_payment_options [data-testid="stHorizontalBlock"] {
            align-items: flex-end !important;
            gap: 0.85rem !important;
        }

        .st-key-subscription_payment_options label,
        .st-key-subscription_payment_options [data-testid="stWidgetLabel"] p {
            color: #334155 !important;
            font-size: 0.82rem !important;
            font-weight: 750 !important;
            line-height: 1.3 !important;
        }

        .st-key-subscription_payment_options div[data-baseweb="select"] > div,
        .st-key-subscription_payment_options div[data-testid="stTextInput"] div[data-baseweb="input"] {
            min-height: 46px !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 13px !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045) !important;
            transition: border-color 170ms ease, box-shadow 170ms ease, background 170ms ease !important;
        }

        .st-key-subscription_payment_options div[data-baseweb="select"] > div:hover,
        .st-key-subscription_payment_options div[data-testid="stTextInput"] div[data-baseweb="input"]:hover {
            border-color: rgba(99, 102, 241, 0.48) !important;
            background: #FFFFFF !important;
        }

        .st-key-subscription_payment_options div[data-baseweb="select"] > div:focus-within,
        .st-key-subscription_payment_options div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
            border-color: #6366F1 !important;
            background: #FFFFFF !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.12), 0 9px 20px rgba(79, 70, 229, 0.08) !important;
        }

        .st-key-subscription_payment_options input,
        .st-key-subscription_payment_options div[data-baseweb="select"] span {
            color: #0F172A !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] input,
        .st-key-manual_payment_select div[data-baseweb="select"] input:focus {
            position: absolute !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            opacity: 0 !important;
            color: transparent !important;
            caret-color: transparent !important;
            text-shadow: none !important;
            pointer-events: none !important;
            overflow: hidden !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] [contenteditable="true"] {
            caret-color: transparent !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] {
            width: 100% !important;
            cursor: pointer !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] > div {
            display: flex !important;
            align-items: center !important;
            min-height: 48px !important;
            height: 48px !important;
            padding: 0 12px 0 14px !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
            overflow: hidden !important;
            transition: border-color 160ms ease, box-shadow 160ms ease !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] > div:hover {
            border-color: #94A3B8 !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] > div:focus-within,
        .st-key-manual_payment_select [role="combobox"][aria-expanded="true"] {
            border-color: #6366F1 !important;
            background: #FFFFFF !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] span {
            display: inline-flex !important;
            align-items: center !important;
            min-height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            color: #0F172A !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            line-height: 1.2 !important;
        }

        .st-key-manual_payment_select div[data-baseweb="select"] svg {
            display: block !important;
            align-self: center !important;
            flex: 0 0 auto !important;
            width: 18px !important;
            height: 18px !important;
            margin: 0 !important;
            color: #64748B !important;
            fill: #64748B !important;
            opacity: 1 !important;
        }

        .st-key-subscription_payment_options input::placeholder {
            color: #7C8AA0 !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }

        .st-key-subscription_payment_options div[data-baseweb="select"] svg {
            color: #64748B !important;
            fill: #64748B !important;
        }

        .st-key-subscription_pricing_grid > div[data-testid="stVerticalBlock"] {
            gap: 1rem !important;
        }

        .st-key-subscription_pricing_grid [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
            gap: 1rem !important;
        }

        .st-key-subscription_pricing_grid [data-testid="column"] {
            display: flex !important;
            min-width: 0 !important;
        }

        .st-key-subscription_pricing_grid [data-testid="column"] > div,
        [class*="st-key-pricing_plan_"] {
            width: 100% !important;
            height: 100% !important;
        }

        [class*="st-key-pricing_plan_"] > div[data-testid="stVerticalBlock"] {
            height: 100% !important;
            gap: 0.7rem !important;
        }

        .pricing-card {
            position: relative !important;
            display: flex !important;
            flex-direction: column !important;
            min-height: 545px !important;
            height: 100% !important;
            padding: 1.25rem 1.2rem 1.15rem !important;
            border: 1px solid rgba(148, 163, 184, 0.30) !important;
            border-radius: 22px !important;
            background: linear-gradient(160deg, #FFFFFF 0%, #FAFBFF 100%) !important;
            box-shadow: 0 15px 34px rgba(15, 23, 42, 0.065) !important;
            overflow: hidden !important;
            transition: transform 190ms ease, box-shadow 190ms ease, border-color 190ms ease !important;
        }

        .pricing-card::before {
            content: "" !important;
            position: absolute !important;
            inset: 0 0 auto !important;
            height: 3px !important;
            background: linear-gradient(90deg, rgba(37, 99, 235, 0.22), rgba(124, 58, 237, 0.26)) !important;
        }

        .pricing-card.recommended {
            border-color: rgba(99, 102, 241, 0.42) !important;
            background: linear-gradient(155deg, #FFFFFF 0%, #F5F3FF 100%) !important;
            box-shadow: 0 20px 42px rgba(79, 70, 229, 0.13) !important;
        }

        .pricing-card.recommended::before {
            height: 4px !important;
            background: linear-gradient(90deg, #2563EB, #7C3AED) !important;
        }

        .pricing-card.current {
            border-color: rgba(16, 185, 129, 0.55) !important;
            background: linear-gradient(180deg, rgba(209, 250, 229, 0.68) 0%, #FFFFFF 38%, #F0FFF7 100%) !important;
            box-shadow: 0 20px 50px rgba(16, 185, 129, 0.18), 0 0 0 1px rgba(16, 185, 129, 0.08), 0 4px 12px rgba(16, 185, 129, 0.06) !important;
        }

        .pricing-card.current::before {
            height: 4px !important;
            background: linear-gradient(90deg, #059669, #10B981, #34D399) !important;
        }

        @media (hover: hover) and (pointer: fine) {
            .pricing-card:hover {
                transform: translateY(-3px) !important;
                border-color: rgba(99, 102, 241, 0.38) !important;
                box-shadow: 0 22px 44px rgba(79, 70, 229, 0.12) !important;
            }

            .pricing-card.current:hover {
                border-color: rgba(16, 185, 129, 0.62) !important;
                box-shadow: 0 24px 52px rgba(16, 185, 129, 0.20), 0 0 0 1px rgba(16, 185, 129, 0.10) !important;
            }
        }

        .pricing-card-topline {
            display: flex !important;
            align-items: flex-start !important;
            justify-content: space-between !important;
            flex-wrap: wrap !important;
            gap: 0.55rem !important;
        }

        .pricing-card-name {
            color: #0F172A !important;
            font-size: 1.18rem !important;
            font-weight: 850 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.025em !important;
        }

        .pricing-card-badges {
            display: flex !important;
            flex-wrap: wrap !important;
            justify-content: flex-end !important;
            gap: 0.35rem !important;
        }

        .pricing-status-badge {
            display: inline-flex !important;
            align-items: center !important;
            min-height: 24px !important;
            padding: 0.25rem 0.55rem !important;
            border-radius: 999px !important;
            font-size: 0.68rem !important;
            font-weight: 800 !important;
            line-height: 1 !important;
            letter-spacing: 0.025em !important;
            white-space: nowrap !important;
        }

        .pricing-status-badge.current {
            color: #047857 !important;
            background: rgba(16, 185, 129, 0.14) !important;
            border: 1px solid rgba(16, 185, 129, 0.30) !important;
            box-shadow: 0 5px 14px rgba(16, 185, 129, 0.09) !important;
            font-size: 0.71rem !important;
            padding: 0.22rem 0.65rem !important;
        }

        .pricing-card.current .pricing-card-limits span {
            color: #065F46 !important;
            background: rgba(209, 250, 229, 0.62) !important;
            border-color: rgba(16, 185, 129, 0.30) !important;
        }

        .pricing-status-badge.recommended {
            color: #4338CA !important;
            background: rgba(99, 102, 241, 0.11) !important;
            border: 1px solid rgba(99, 102, 241, 0.27) !important;
        }

        .pricing-status-badge.muted {
            color: #475569 !important;
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
        }

        .pricing-card-price {
            margin-top: 1rem !important;
            color: #111827 !important;
            font-size: clamp(1.8rem, 3vw, 2.25rem) !important;
            font-weight: 900 !important;
            line-height: 1.05 !important;
            letter-spacing: -0.045em !important;
        }

        .pricing-card-limits {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.4rem !important;
            margin-top: 0.7rem !important;
        }

        .pricing-card-limits span {
            display: inline-flex !important;
            padding: 0.3rem 0.55rem !important;
            border-radius: 999px !important;
            color: #334155 !important;
            background: rgba(238, 242, 255, 0.78) !important;
            border: 1px solid rgba(165, 180, 252, 0.26) !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }

        .pricing-card-description {
            min-height: 2.8rem !important;
            margin: 0.85rem 0 0 !important;
            color: #64748B !important;
            font-size: 0.88rem !important;
            line-height: 1.5 !important;
        }

        .pricing-feature-title {
            margin-top: 1rem !important;
            padding-top: 0.9rem !important;
            border-top: 1px solid rgba(148, 163, 184, 0.20) !important;
            color: #334155 !important;
            font-size: 0.76rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
        }

        .pricing-feature-list {
            display: flex !important;
            flex-direction: column !important;
            gap: 0.52rem !important;
            margin: 0.75rem 0 0 !important;
            padding: 0 !important;
            list-style: none !important;
        }

        .pricing-feature-list li {
            display: flex !important;
            align-items: flex-start !important;
            gap: 0.5rem !important;
            color: #334155 !important;
            font-size: 0.82rem !important;
            line-height: 1.35 !important;
        }

        .pricing-feature-list li.disabled {
            color: #94A3B8 !important;
        }

        .pricing-feature-icon {
            display: inline-grid !important;
            place-items: center !important;
            flex: 0 0 1.15rem !important;
            width: 1.15rem !important;
            height: 1.15rem !important;
            margin-top: 0.03rem !important;
            border-radius: 999px !important;
            color: #047857 !important;
            background: rgba(16, 185, 129, 0.10) !important;
            font-size: 0.68rem !important;
            font-weight: 900 !important;
        }

        .pricing-feature-list li.disabled .pricing-feature-icon {
            color: #94A3B8 !important;
            background: rgba(148, 163, 184, 0.10) !important;
        }

        [class*="st-key-pricing_plan_"] .stButton,
        [class*="st-key-pricing_plan_"] .stLinkButton {
            margin-top: auto !important;
        }

        [class*="st-key-pricing_plan_"] .stButton > button,
        [class*="st-key-pricing_plan_"] .stLinkButton > a {
            width: 100% !important;
            min-height: 44px !important;
            border-radius: 13px !important;
            font-weight: 800 !important;
        }

        [class*="st-key-pricing_plan_"]:has(.pricing-card.current) .stButton > button:disabled {
            color: #047857 !important;
            background: rgba(209, 250, 229, 0.85) !important;
            border: 1px solid rgba(16, 185, 129, 0.40) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.90), 0 6px 16px rgba(16, 185, 129, 0.08) !important;
            opacity: 1 !important;
            cursor: default !important;
            font-weight: 800 !important;
        }

        .st-key-subscription_payment_request .sg-card {
            border-color: rgba(245, 158, 11, 0.28) !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #FFFBEB 100%) !important;
            box-shadow: 0 14px 32px rgba(245, 158, 11, 0.08) !important;
        }

        /* Lightweight premium motion system */
        @keyframes sgFadeSlideUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes sgSoftFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .sg-page-header,
        .sg-saas-card,
        .sg-section-card,
        .sg-empty-state,
        .locked-card,
        .admin-hero,
        .st-key-result_export_actions,
        .st-key-admin_panel_tabs,
        [data-testid="stMain"] [data-testid="stFileUploader"] {
            animation: sgFadeSlideUp 300ms cubic-bezier(0.22, 1, 0.36, 1) both !important;
            will-change: opacity, transform !important;
        }

        .sg-metric-card,
        .admin-metric-card,
        .sg-action-card,
        .admin-plan-card,
        .admin-user-card,
        .status-card {
            animation: sgSoftFadeIn 280ms ease-out both !important;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease !important;
        }

        @media (hover: hover) and (pointer: fine) {
            .sg-metric-card:hover,
            .admin-metric-card:hover,
            .sg-action-card:hover,
            .admin-plan-card:hover,
            .status-card:hover {
                transform: translateY(-2px) !important;
                border-color: rgba(99, 102, 241, 0.24) !important;
                box-shadow: 0 17px 34px rgba(79, 70, 229, 0.09) !important;
            }
        }

        [data-testid="stMain"] .stButton > button,
        [data-testid="stMain"] .stDownloadButton > button,
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button,
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .sg-nav-active {
            transition: transform 170ms ease, box-shadow 170ms ease, border-color 170ms ease, background 170ms ease, color 170ms ease, filter 170ms ease !important;
        }

        [data-testid="stMain"] .stButton > button:active:not(:disabled),
        [data-testid="stMain"] .stDownloadButton > button:active:not(:disabled),
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:active:not(:disabled) {
            transform: translateY(0) scale(0.985) !important;
            filter: brightness(0.99) !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .sg-nav-active {
            transition-duration: 160ms !important;
        }

        [data-testid="stMain"] [role="tab"],
        .st-key-admin_panel_tabs [role="tab"],
        [data-testid="stTabsScrollLeft"],
        [data-testid="stTabsScrollRight"] {
            transition: color 160ms ease, background 160ms ease, border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease !important;
        }

        .sg-usage-progress-fill {
            transition: width 420ms cubic-bezier(0.22, 1, 0.36, 1), background 180ms ease !important;
        }

        [data-testid="stProgressBar"] > div > div,
        [role="progressbar"] > div {
            transition: width 360ms cubic-bezier(0.22, 1, 0.36, 1) !important;
        }

        [data-testid="stSidebar"] {
            transition: transform 260ms cubic-bezier(0.22, 1, 0.36, 1), min-width 260ms ease, max-width 260ms ease, box-shadow 220ms ease !important;
        }

        .st-key-result_export_actions .st-key-download_clean_zip_primary button {
            transition: transform 170ms ease, box-shadow 170ms ease, filter 170ms ease, background 170ms ease !important;
        }

        @media (prefers-reduced-motion: reduce) {
            html:focus-within {
                scroll-behavior: auto !important;
            }

            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 0.9rem !important;
                padding-right: 0.9rem !important;
            }

            [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
                gap: 0.8rem !important;
            }

            .sg-page-header,
            .sg-card,
            .sg-section-card,
            .locked-card,
            .admin-hero {
                padding: 1rem !important;
            }

            .sg-metric-card,
            .admin-metric-card,
            .sg-action-card,
            .admin-plan-card {
                min-height: auto !important;
                padding: 1rem !important;
            }

            .st-key-result_export_actions {
                padding: 1rem !important;
                border-radius: 17px !important;
            }

            .st-key-result_export_actions [data-testid="stCaptionContainer"] {
                min-height: auto !important;
            }

            [data-testid="stMain"] [data-testid="stTabs"] [role="tab"] {
                padding-left: 0.55rem !important;
                padding-right: 0.55rem !important;
                font-size: 0.88rem !important;
            }
        }

        @media (max-width: 1024px) {
            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"],
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] * {
                box-sizing: border-box !important;
            }

            header[data-testid="stHeader"] {
                display: flex !important;
                visibility: visible !important;
                height: 3rem !important;
                min-height: 3rem !important;
                background: transparent !important;
                pointer-events: none !important;
                z-index: 999990 !important;
            }

            [data-testid="stToolbar"] {
                display: flex !important;
                visibility: visible !important;
                width: auto !important;
                min-width: 0 !important;
                background: transparent !important;
                pointer-events: auto !important;
            }

            [data-testid="collapsedControl"],
            [data-testid="stExpandSidebarButton"],
            button[data-testid="stExpandSidebarButton"][kind="header"],
            [data-testid="stSidebarCollapseButton"],
            [data-testid="stSidebarNavCollapseButton"],
            button[aria-label="Open sidebar"],
            button[title="Open sidebar"] {
                display: inline-flex !important;
                visibility: visible !important;
                align-items: center !important;
                justify-content: center !important;
                width: 40px !important;
                height: 40px !important;
                min-width: 40px !important;
                min-height: 40px !important;
                margin: 0.35rem !important;
                padding: 0 !important;
                border: 1px solid rgba(99, 102, 241, 0.22) !important;
                border-radius: 12px !important;
                background: rgba(255, 255, 255, 0.94) !important;
                color: #334155 !important;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10) !important;
                pointer-events: auto !important;
            }

            [data-testid="stExpandSidebarButton"],
            button[data-testid="stExpandSidebarButton"][kind="header"] {
                position: fixed !important;
                top: 12px !important;
                left: 12px !important;
                z-index: 999999 !important;
                width: 44px !important;
                height: 44px !important;
                min-width: 44px !important;
                min-height: 44px !important;
                margin: 0 !important;
                padding: 0 !important;
                gap: 0 !important;
                border: 1px solid rgba(255, 255, 255, 0.34) !important;
                border-radius: 15px !important;
                background: linear-gradient(135deg, #2563EB 0%, #6366F1 48%, #7C3AED 100%) !important;
                color: #FFFFFF !important;
                box-shadow: 0 11px 28px rgba(37, 99, 235, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
                backdrop-filter: blur(10px) !important;
                -webkit-backdrop-filter: blur(10px) !important;
                overflow: hidden !important;
                transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease !important;
            }

            [data-testid="stExpandSidebarButton"]::after,
            button[data-testid="stExpandSidebarButton"][kind="header"]::after {
                content: none !important;
                display: none !important;
            }

            [data-testid="stExpandSidebarButton"] svg,
            button[data-testid="stExpandSidebarButton"][kind="header"] svg {
                width: 23px !important;
                height: 23px !important;
                color: #FFFFFF !important;
                fill: #FFFFFF !important;
                filter: drop-shadow(0 1px 2px rgba(30, 41, 59, 0.2)) !important;
            }

            [data-testid="stExpandSidebarButton"]:hover,
            button[data-testid="stExpandSidebarButton"][kind="header"]:hover {
                background: linear-gradient(135deg, #1D4ED8 0%, #4F46E5 48%, #6D28D9 100%) !important;
                border-color: rgba(255, 255, 255, 0.5) !important;
                box-shadow: 0 14px 30px rgba(79, 70, 229, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
                filter: brightness(1.05) !important;
                transform: translateY(-1px) !important;
            }

            [data-testid="stExpandSidebarButton"]:active,
            button[data-testid="stExpandSidebarButton"][kind="header"]:active {
                transform: translateY(0) scale(0.97) !important;
                filter: brightness(0.98) !important;
                box-shadow: 0 8px 20px rgba(79, 70, 229, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
                display: flex !important;
                visibility: visible !important;
                align-items: center !important;
                justify-content: flex-end !important;
                width: 100% !important;
                min-width: 100% !important;
                max-width: none !important;
                height: 46px !important;
                min-height: 46px !important;
                max-height: 46px !important;
                padding: 0.25rem 0.45rem !important;
                margin: 0 !important;
                overflow: visible !important;
            }

            [data-testid="stSidebar"] button[aria-label="Close sidebar"],
            [data-testid="stSidebar"] button[title="Close sidebar"],
            [data-testid="stSidebarCollapseButton"] button,
            [data-testid="stSidebarCollapseButton"] button[kind="header"] {
                display: inline-flex !important;
                visibility: visible !important;
                align-items: center !important;
                justify-content: center !important;
                width: 38px !important;
                height: 38px !important;
                min-width: 38px !important;
                min-height: 38px !important;
                padding: 0 !important;
                border-radius: 13px !important;
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.30), rgba(124, 58, 237, 0.28)) !important;
                color: #FFFFFF !important;
                border: 1px solid rgba(165, 180, 252, 0.32) !important;
                box-shadow: 0 8px 20px rgba(2, 6, 23, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
                transition: background 140ms ease, border-color 140ms ease, transform 140ms ease !important;
            }

            [data-testid="stSidebarCollapseButton"] button:hover,
            [data-testid="stSidebarCollapseButton"] button[kind="header"]:hover {
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.44), rgba(124, 58, 237, 0.42)) !important;
                border-color: rgba(191, 219, 254, 0.48) !important;
                transform: translateY(-1px) !important;
            }

            [data-testid="stSidebarCollapseButton"] svg {
                width: 21px !important;
                height: 21px !important;
                color: #FFFFFF !important;
                fill: #FFFFFF !important;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                height: calc(100% - 46px) !important;
            }

            [data-testid="stMainBlockContainer"] {
                max-width: 100% !important;
                padding-left: clamp(0.9rem, 2.5vw, 1.35rem) !important;
                padding-right: clamp(0.9rem, 2.5vw, 1.35rem) !important;
            }

            .sg-workflow-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }

            .st-key-subscription_pricing_grid [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 calc(50% - 0.75rem) !important;
                width: auto !important;
                min-width: min(100%, 280px) !important;
            }

            .sg-table-wrap,
            .sg-light-table-shell,
            [data-testid="stDataFrame"],
            [data-testid="stTable"] {
                max-width: 100% !important;
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch !important;
            }

            img,
            [data-testid="stImage"] img {
                max-width: 100% !important;
                height: auto !important;
            }
        }

        @media (max-width: 768px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 4rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-bottom: 2rem !important;
            }

            .st-key-subscription_payment_options {
                padding: 1rem !important;
                border-radius: 17px !important;
            }

            .st-key-subscription_payment_options [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.7rem !important;
            }

            .st-key-subscription_payment_options [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }

            [data-testid="stSidebar"] {
                min-width: 0 !important;
                max-width: 340px !important;
                flex: 0 0 auto !important;
            }

            [data-testid="stSidebar"][aria-expanded="true"] {
                width: min(86vw, 340px) !important;
                min-width: min(86vw, 340px) !important;
                box-shadow: 18px 0 42px rgba(2, 6, 23, 0.28) !important;
                z-index: 1000000 !important;
            }

            [data-testid="stSidebar"][aria-expanded="false"] {
                width: 0 !important;
                min-width: 0 !important;
                max-width: 0 !important;
                box-shadow: none !important;
                overflow: hidden !important;
            }

            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            main,
            .block-container {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
                overflow-x: hidden !important;
            }

            [data-testid="stMain"] {
                flex: 1 1 100% !important;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
                padding-left: 0.7rem !important;
                padding-right: 0.7rem !important;
                padding-bottom: 0.8rem !important;
            }

            [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.75rem !important;
            }

            [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 calc(50% - 0.75rem) !important;
                width: auto !important;
                min-width: min(100%, 250px) !important;
            }

            .st-key-result_export_actions [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }

            .st-key-result_export_actions .stDownloadButton > button,
            .st-key-result_export_actions .stButton > button {
                width: 100% !important;
                min-height: 46px !important;
            }

            [data-testid="stMain"] [data-testid="stTabs"] {
                max-width: 100% !important;
                overflow: hidden !important;
            }

            [data-testid="stMain"] [data-testid="stTabs"] [role="tablist"] {
                display: flex !important;
                flex-wrap: nowrap !important;
                max-width: 100% !important;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                scrollbar-width: thin !important;
                -webkit-overflow-scrolling: touch !important;
            }

            [data-testid="stMain"] [data-testid="stTabs"] [role="tab"] {
                flex: 0 0 auto !important;
                white-space: nowrap !important;
                min-width: max-content !important;
            }

            .sg-page-header,
            .sg-saas-card,
            .sg-card,
            .sg-section-card,
            .sg-empty-state,
            .locked-card,
            .admin-hero {
                padding: 0.95rem !important;
                border-radius: 16px !important;
            }

            .sg-metric-card,
            .admin-metric-card,
            .sg-action-card,
            .admin-plan-card {
                min-height: auto !important;
                padding: 0.95rem !important;
            }

            .sg-page-title,
            .sg-page-title h1 {
                font-size: clamp(1.55rem, 7vw, 1.95rem) !important;
                overflow-wrap: anywhere !important;
            }

            .sg-page-subtitle,
            .sg-muted,
            .sg-card-title,
            [data-testid="stMain"] p,
            [data-testid="stMain"] label {
                overflow-wrap: anywhere !important;
                word-break: normal !important;
            }

            [data-testid="stMain"] .stButton > button,
            [data-testid="stMain"] .stDownloadButton > button,
            [data-testid="stMain"] [data-testid="stFormSubmitButton"] button {
                min-height: 44px !important;
            }

            [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
                min-height: 118px !important;
                padding: 0.9rem !important;
            }

            .sg-light-table,
            .sg-light-table-shell table,
            .sg-table-wrap table {
                min-width: 700px !important;
            }

            .st-key-admin_panel_tabs[data-testid="stTabs"],
            .st-key-admin_panel_tabs [data-testid="stTabs"] {
                margin-top: 0.55rem !important;
                padding: 0.28rem !important;
                border-radius: 14px !important;
            }

            .st-key-admin_panel_tabs [role="tab"] {
                min-height: 40px !important;
                padding: 0.52rem 0.72rem !important;
                font-size: 0.84rem !important;
            }

            .st-key-admin_panel_tabs [data-testid="stTabsScrollLeft"],
            .st-key-admin_panel_tabs [data-testid="stTabsScrollRight"] {
                top: 0.28rem !important;
                width: 36px !important;
                height: 40px !important;
                min-width: 36px !important;
            }

            .st-key-subscription_pricing_grid [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 calc(50% - 0.75rem) !important;
                width: auto !important;
                min-width: min(100%, 280px) !important;
            }

            .pricing-card {
                min-height: 510px !important;
            }
        }

        @media (max-width: 480px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 0.6rem !important;
                padding-right: 0.6rem !important;
            }

            [data-testid="stExpandSidebarButton"],
            button[data-testid="stExpandSidebarButton"][kind="header"] {
                top: 10px !important;
                left: 10px !important;
                width: 42px !important;
                height: 42px !important;
                min-width: 42px !important;
                min-height: 42px !important;
                border-radius: 14px !important;
            }

            [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }

            .sg-workflow-grid {
                grid-template-columns: 1fr !important;
                gap: 0.55rem !important;
            }

            .st-key-subscription_pricing_grid [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }

            .pricing-card {
                min-height: auto !important;
                padding: 1.05rem !important;
                border-radius: 18px !important;
            }

            .pricing-card-description {
                min-height: auto !important;
            }

            .sg-page-header,
            .sg-saas-card,
            .sg-card,
            .sg-section-card,
            .sg-empty-state,
            .locked-card,
            .sg-metric-card,
            .admin-metric-card,
            .sg-action-card,
            .admin-plan-card,
            .admin-hero,
            .st-key-result_export_actions {
                padding: 0.85rem !important;
                border-radius: 15px !important;
            }

            [data-testid="stMain"] h1 {
                font-size: 1.7rem !important;
            }

            [data-testid="stMain"] h2 {
                font-size: 1.3rem !important;
            }

            [data-testid="stMain"] [data-testid="stTabs"] [role="tab"] {
                min-height: 40px !important;
                padding: 0.5rem 0.6rem 0.6rem !important;
                font-size: 0.84rem !important;
            }

            [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {
                min-height: 106px !important;
                padding: 0.75rem !important;
            }

            .sg-footer {
                font-size: 0.82rem !important;
                line-height: 1.45 !important;
            }
        }

        /* Footer polishing */
        .sg-footer {
            border-top: 1px solid #E2E8F0 !important;
            margin-top: 32px !important;
            padding: 0.75rem 0.15rem 24px 0.15rem !important;
            color: #64748B !important;
            font-size: 0.92rem !important;
            line-height: 1.5 !important;
            text-align: left !important;
            display: block !important;
        }

        </style>
        <script>
        // Inject a single footer into the main content if not already present.
        (function(){
            try{
                function ensureFooter(){
                    const main = document.querySelector('[data-testid="stMain"]');
                    if(!main) return;
                    // Remove any existing injected footer to avoid duplicates
                    const existing = main.querySelector('.sg-footer[data-injected="true"]');
                    if(existing) existing.remove();
                    const footer = document.createElement('div');
                    footer.className = 'sg-footer';
                    footer.setAttribute('data-injected','true');
                    footer.innerHTML = `<div>© 2026 StockGuard AI. A product by BEE CLUSTER.<br/>StockGuard AI is an independent tool and is not affiliated with Adobe.</div>`;
                    main.appendChild(footer);
                }
                if(document.readyState === 'complete' || document.readyState === 'interactive'){
                    setTimeout(ensureFooter, 120);
                } else {
                    window.addEventListener('DOMContentLoaded', function(){ setTimeout(ensureFooter, 120); });
                }
                // Also guard for dynamic Streamlit rerenders
                const obs = new MutationObserver(() => { setTimeout(ensureFooter, 120); });
                const root = document.querySelector('[data-testid="stMain"]');
                if(root) obs.observe(root, {childList:true, subtree:true});
            }catch(e){console && console.warn && console.warn('footer inject failed', e);} 
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )



def get_device() -> torch.device:
    """Use CUDA when available, otherwise use CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model(device_name: str) -> torch.nn.Module:
    """Load ResNet50 and remove the final classification layer."""

    device = torch.device(device_name)
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


@st.cache_resource(show_spinner=False)
def get_preprocess_pipeline() -> transforms.Compose:
    """Create the standard ImageNet preprocessing used by pretrained ResNet50."""

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Convert image bytes into a preprocessed tensor."""

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preprocess = get_preprocess_pipeline()
    return preprocess(image)


def risk_label(similarity_percent: float) -> str:
    if similarity_percent >= 95.0:
        return "Very High Risk / Near Duplicate"
    if similarity_percent >= 90.0:
        return "High Risk"
    if similarity_percent >= 80.0:
        return "Possible Similar Content"
    return "Below Review Threshold"


def recommended_action(similarity_percent: float) -> str:
    if similarity_percent >= 95.0:
        return "Keep only the strongest image from this pair before submitting."
    if similarity_percent >= 90.0:
        return "Review carefully; choose one unless the images have clearly different value."
    if similarity_percent >= 80.0:
        return "Possible issue; compare captions, subject, crop, and final use case."
    return "Usually lower priority at the current threshold."


def sanitize_filename(filename: str) -> str:
    """Return a ZIP/report-safe filename with no path traversal or formula prefix."""

    raw_name = os.path.basename(str(filename or "")).strip()
    raw_name = raw_name.replace("\\", "_").replace("/", "_")
    raw_name = re.sub(r"[^A-Za-z0-9._ -]", "_", raw_name)
    raw_name = re.sub(r"\s+", " ", raw_name).strip(" .")
    if not raw_name:
        raw_name = "uploaded_image.jpg"
    if raw_name[0] in {"=", "+", "-", "@"}:
        raw_name = f"_{raw_name}"
    stem, extension = os.path.splitext(raw_name)
    extension = extension.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        extension = ".jpg"
    stem = stem[:120] or "uploaded_image"
    return f"{stem}{extension}"


def upload_signature(uploaded_file) -> Tuple[str, int, str]:
    """Create a stable signature for a file upload, including file contents."""

    name = str(getattr(uploaded_file, "name", "") or "")
    size = int(getattr(uploaded_file, "size", 0) or 0)
    try:
        content = uploaded_file.getvalue()
    except Exception:
        content = b""

    digest = hashlib.sha256(content).hexdigest() if isinstance(content, (bytes, bytearray, memoryview)) else ""
    return name, size, digest


def make_unique_filename(filename: str, seen_names: Dict[str, int]) -> Tuple[str, bool]:
    """Return a safe unique filename for reports and ZIP output.

    Duplicate filenames cause report conflicts because two rows can point to the
    same name. We keep the original filename separately and rename only the
    internal scan name when needed.
    """

    clean_name = sanitize_filename(filename)
    if clean_name not in seen_names:
        seen_names[clean_name] = 1
        return clean_name, False

    seen_names[clean_name] += 1
    stem, extension = os.path.splitext(clean_name)
    return f"{stem}_{seen_names[clean_name]}{extension}", True


def validate_uploaded_image(uploaded_file) -> Dict:
    """Strictly validate image bytes before AI processing."""

    original_name = getattr(uploaded_file, "name", "uploaded_image")
    safe_filename = sanitize_filename(original_name)
    extension = os.path.splitext(os.path.basename(str(original_name)))[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return {"is_valid": False, "error": "unsupported file extension", "safe_filename": safe_filename}

    image_bytes = uploaded_file.getvalue()
    size_mb = len(image_bytes) / (1024 * 1024)
    if not image_bytes:
        return {"is_valid": False, "error": "empty file", "safe_filename": safe_filename}
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return {
            "is_valid": False,
            "error": f"file is larger than {MAX_IMAGE_SIZE_MB}MB",
            "safe_filename": safe_filename,
        }

    try:
        with Image.open(io.BytesIO(image_bytes)) as verify_image:
            image_format = verify_image.format
            verify_image.verify()
        if image_format not in ALLOWED_IMAGE_FORMATS:
            return {
                "is_valid": False,
                "error": f"actual image format {image_format or 'unknown'} is not supported",
                "safe_filename": safe_filename,
            }
        with Image.open(io.BytesIO(image_bytes)) as reopened_image:
            pil_image = reopened_image.convert("RGB")
        return {
            "is_valid": True,
            "error": "",
            "safe_filename": safe_filename,
            "image_bytes": image_bytes,
            "pil_image": pil_image,
            "format": image_format,
            "size_mb": size_mb,
        }
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return {
            "is_valid": False,
            "error": f"corrupted or unsupported image ({error})",
            "safe_filename": safe_filename,
        }


def load_safe_image(uploaded_file) -> Dict:
    """Beginner-friendly wrapper name for strict upload validation."""

    return validate_uploaded_image(uploaded_file)


def get_image_original_bytes(uploaded_file) -> bytes:
    """Read original uploaded bytes once.

    These bytes are preserved for the Clean ZIP export so the customer's final
    download keeps the original image file quality.
    """

    return uploaded_file.getvalue()


def create_thumbnail(pil_image: Image.Image, max_size: int = 300) -> Image.Image:
    """Create a small preview image for the Streamlit UI."""

    thumbnail = pil_image.copy()
    thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return thumbnail.convert("RGB")


def create_model_input_image(pil_image: Image.Image, size: int = 224) -> Image.Image:
    """Create the small image copy used for ResNet50 similarity analysis."""

    width, height = pil_image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    cropped = pil_image.crop((left, top, left + crop_size, top + crop_size))
    return cropped.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")


def load_image_safely(uploaded_file, seen_names: Dict[str, int]) -> Tuple[UploadedImage | None, str | None, bool]:
    """Load one uploaded image with memory-safe optimized copies."""

    validation = load_safe_image(uploaded_file)
    if not validation["is_valid"]:
        return None, f"{uploaded_file.name}: {validation['error']}", False

    image = validation["pil_image"]
    bytes_data = validation["image_bytes"]
    original_width, original_height = image.size
    megapixels = (original_width * original_height) / 1_000_000
    status_notes = [f"OK ({validation['format']}, {validation['size_mb']:.2f}MB)"]
    if megapixels > 50:
        status_notes.append("Very large image optimized for scan")

    internal_name, renamed = make_unique_filename(validation["safe_filename"], seen_names)
    thumbnail = create_thumbnail(image)
    model_image = create_model_input_image(image)

    return (
        UploadedImage(
            name=internal_name,
            original_name=uploaded_file.name,
            image=thumbnail,
            thumbnail=thumbnail,
            model_image=model_image,
            bytes_data=bytes_data,
            original_width=original_width,
            original_height=original_height,
            processing_status="; ".join(status_notes),
        ),
        None,
        renamed,
    )


def read_uploaded_images(uploaded_files) -> Tuple[List[UploadedImage], List[str]]:
    """Read uploaded files and skip corrupted images with a warning message."""

    valid_images: List[UploadedImage] = []
    errors: List[str] = []
    seen_names: Dict[str, int] = {}
    renamed_duplicates = False

    for uploaded_file in uploaded_files:
        image, error_name, renamed = load_image_safely(uploaded_file, seen_names)
        if image:
            valid_images.append(image)
            renamed_duplicates = renamed_duplicates or renamed
        elif error_name:
            errors.append(error_name)

    if renamed_duplicates:
        st.warning("Some duplicate filenames were renamed internally to avoid report conflicts.")

    return valid_images, errors


def preprocess_model_image(image: Image.Image) -> torch.Tensor:
    """Normalize a 224px RGB image for pretrained ResNet50."""

    preprocess = get_preprocess_pipeline()
    return preprocess(image)


def create_embeddings(
    images: List[UploadedImage],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    tensors = [preprocess_model_image(item.model_image) for item in images]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        embeddings = model(batch)

    return embeddings.cpu().numpy()


def compare_images(
    images: List[UploadedImage],
    embeddings: np.ndarray,
    threshold_percent: float,
) -> Tuple[List[Dict], List[Dict], float]:
    """Compare every image pair once and return risky pairs plus all pairs."""

    similarity_matrix = cosine_similarity(embeddings)
    risky_pairs: List[Dict] = []
    all_pairs: List[Dict] = []
    highest_similarity = 0.0

    for first_index in range(len(images)):
        for second_index in range(first_index + 1, len(images)):
            similarity_percent = float(similarity_matrix[first_index, second_index] * 100)
            similarity_percent = max(0.0, min(100.0, similarity_percent))
            highest_similarity = max(highest_similarity, similarity_percent)

            pair = {
                "image_1_index": first_index,
                "image_2_index": second_index,
                "image_1": images[first_index].name,
                "image_2": images[second_index].name,
                "similarity_percent": round(similarity_percent, 2),
                "risk_level": risk_label(similarity_percent),
                "recommended_action": recommended_action(similarity_percent),
            }
            all_pairs.append(pair)

            if similarity_percent >= threshold_percent:
                risky_pairs.append(pair)

    return risky_pairs, all_pairs, highest_similarity


def build_groups(number_of_images: int, risky_pairs: List[Dict]) -> List[List[int]]:
    union_find = UnionFind(number_of_images)

    for pair in risky_pairs:
        union_find.union(pair["image_1_index"], pair["image_2_index"])

    grouped_indexes: Dict[int, List[int]] = {}
    for index in range(number_of_images):
        root = union_find.find(index)
        grouped_indexes.setdefault(root, []).append(index)

    return [indexes for indexes in grouped_indexes.values() if len(indexes) > 1]


def group_highest_similarity(group: List[int], risky_pairs: List[Dict]) -> float:
    group_members = set(group)
    scores = [
        pair["similarity_percent"]
        for pair in risky_pairs
        if pair["image_1_index"] in group_members and pair["image_2_index"] in group_members
    ]
    return max(scores) if scores else 0.0


def assign_group_numbers(groups: List[List[int]], risky_pairs: List[Dict]) -> List[Dict]:
    export_rows: List[Dict] = []

    for group_number, group in enumerate(groups, start=1):
        group_members = set(group)
        for pair in risky_pairs:
            if (
                pair["image_1_index"] in group_members
                and pair["image_2_index"] in group_members
            ):
                export_rows.append(
                    {
                        "group_number": group_number,
                        "image_1": pair["image_1"],
                        "image_2": pair["image_2"],
                        "similarity_percent": pair["similarity_percent"],
                        "risk_level": pair["risk_level"],
                        "recommended_action": pair["recommended_action"],
                    }
                )

    return export_rows


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV RGB array."""

    return np.array(image.convert("RGB"))


def calculate_sharpness(cv_image: np.ndarray) -> float:
    """Use OpenCV Laplacian variance. Higher values usually mean a sharper image."""

    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def calculate_brightness(cv_image: np.ndarray) -> float:
    """Average grayscale brightness from 0 dark to 255 bright."""

    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    return float(gray_image.mean())


def brightness_status(brightness: float) -> str:
    if brightness < 70:
        return "Too Dark"
    if brightness > 190:
        return "Too Bright"
    return "Good Exposure"


def get_quality_score(
    megapixels: float,
    sharpness: float,
    brightness: float,
    file_size_kb: float,
) -> Tuple[float, float, float, float]:
    """Return resolution, sharpness, brightness, and overall quality scores.

    Simple MVP scoring:
    - Resolution contributes 35 points. 4MP+ receives full resolution credit.
    - Sharpness contributes 45 points. Laplacian variance near 500+ receives full credit.
    - Brightness contributes 20 points. 70-190 average brightness is treated as good exposure.
    - Very tiny files get a small penalty because they can indicate compression or low detail.
    """

    if megapixels >= 4:
        resolution_points = 35.0
    elif megapixels >= 2:
        resolution_points = 20.0 + ((megapixels - 2.0) / 2.0) * 14.0
    else:
        resolution_points = 5.0 + (megapixels / 2.0) * 14.0

    if sharpness >= 250:
        sharpness_points = 45.0
    elif sharpness >= 100:
        sharpness_points = 25.0 + ((sharpness - 100.0) / 150.0) * 19.0
    else:
        sharpness_points = 5.0 + (sharpness / 100.0) * 19.0

    if 70 <= brightness <= 190:
        brightness_points = 20.0
    elif 50 <= brightness < 70 or 190 < brightness <= 210:
        brightness_points = 12.0
    else:
        brightness_points = 5.0

    tiny_file_penalty = 12 if file_size_kb < 120 else 0
    overall_score = clamp_score(resolution_points + sharpness_points + brightness_points - tiny_file_penalty)
    return (
        clamp_score((resolution_points / 35.0) * 100.0),
        clamp_score((sharpness_points / 45.0) * 100.0),
        clamp_score((brightness_points / 20.0) * 100.0),
        overall_score,
    )


def calculate_single_image_quality(
    image: Image.Image,
    filename: str,
    bytes_data: bytes,
    index: int = 0,
    original_size: Tuple[int, int] | None = None,
    original_filename: str | None = None,
    processing_status: str = "OK",
) -> Dict:
    """Calculate beginner-friendly quality metrics for one uploaded image."""

    width, height = original_size or image.size
    megapixels = (width * height) / 1_000_000
    file_size_kb = len(bytes_data) / 1024

    cv_image = pil_to_cv2(image)
    sharpness = calculate_sharpness(cv_image)
    brightness = calculate_brightness(cv_image)
    resolution_score, sharpness_score, brightness_score, overall_score = get_quality_score(
        megapixels=megapixels,
        sharpness=sharpness,
        brightness=brightness,
        file_size_kb=file_size_kb,
    )

    warnings = []
    if sharpness < 100:
        warnings.append("Blurry")
    if brightness < 70:
        warnings.append("Too Dark")
    elif brightness > 190:
        warnings.append("Too Bright")
    if file_size_kb < 120:
        warnings.append("Very small file")
    if megapixels < 2:
        warnings.append("Low resolution")
    if megapixels > 50:
        warnings.append("Very large image")

    return {
        "index": index,
        "filename": filename,
        "original_filename": original_filename or filename,
        "internal_filename": filename,
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 2),
        "file_size_kb": round(file_size_kb, 2),
        "resolution_score": resolution_score,
        "sharpness_score": sharpness_score,
        "raw_sharpness": round(sharpness, 2),
        "brightness_score": brightness_score,
        "raw_brightness": round(brightness, 2),
        "brightness_status": brightness_status(brightness),
        "quality_score": overall_score,
        "quality_warnings": warnings,
        "warnings": warnings,
        "processing_status": processing_status,
    }


def calculate_image_quality(images) -> List[Dict] | Dict:
    """Calculate quality metrics for one PIL image or a list of UploadedImage objects."""

    if isinstance(images, Image.Image):
        return calculate_single_image_quality(images, "image", b"", 0)

    quality_rows: List[Dict] = []

    for index, item in enumerate(images):
        quality_rows.append(
            calculate_single_image_quality(
                item.image,
                item.name,
                item.bytes_data,
                index,
                original_size=(item.original_width, item.original_height),
                original_filename=item.original_name,
                processing_status=item.processing_status,
            )
        )

    return quality_rows


def build_similarity_summary(
    images: List[UploadedImage],
    all_pairs: List[Dict],
) -> Dict[int, Dict]:
    summary = {
        index: {
            "highest_similarity_percent": 0.0,
            "matched_with": "",
            "similarity_risk": "Low",
        }
        for index in range(len(images))
    }

    for pair in all_pairs:
        score = pair["similarity_percent"]
        first_index = pair["image_1_index"]
        second_index = pair["image_2_index"]

        if score > summary[first_index]["highest_similarity_percent"]:
            summary[first_index] = {
                "highest_similarity_percent": score,
                "matched_with": pair["image_2"],
                "similarity_risk": risk_label(score),
            }
        if score > summary[second_index]["highest_similarity_percent"]:
            summary[second_index] = {
                "highest_similarity_percent": score,
                "matched_with": pair["image_1"],
                "similarity_risk": risk_label(score),
            }

    return summary


def get_highest_similarity_for_image(filename: str, pair_results: List[Dict]) -> float:
    highest = 0.0
    for pair in pair_results:
        if pair["image_1"] == filename or pair["image_2"] == filename:
            highest = max(highest, pair["similarity_percent"])
    return highest


def get_matched_with_for_image(filename: str, pair_results: List[Dict]) -> str:
    best_match = ""
    highest = 0.0
    for pair in pair_results:
        if pair["image_1"] == filename or pair["image_2"] == filename:
            if pair["similarity_percent"] > highest:
                highest = pair["similarity_percent"]
                best_match = pair["image_2"] if pair["image_1"] == filename else pair["image_1"]
    return best_match


def get_best_shot_badge(filename: str, best_filename: str) -> str:
    return "Recommended Keep" if filename == best_filename else "Consider Removing"


def normalize_score(value: float, min_value: float, max_value: float) -> float:
    """Normalize any number into a simple 0-100 score."""

    if max_value <= min_value:
        return 0.0
    return clamp_score(((value - min_value) / (max_value - min_value)) * 100.0)


def calculate_best_shot_score(filename: str, quality_result: Dict, similarity_info: Dict) -> float:
    """Calculate the improved Best Shot score out of 100."""

    quality_component = quality_result["quality_score"] * 0.50
    sharpness_component = quality_result["sharpness_score"] * 0.20
    resolution_component = normalize_score(quality_result["megapixels"], 0.0, 4.0) * 0.15
    exposure_component = quality_result["brightness_score"] * 0.10
    similarity_component = (100.0 - similarity_info["highest_similarity_percent"]) * 0.05

    penalty = 0.0
    warnings = set(quality_result.get("warnings", []))
    if "Blurry" in warnings:
        penalty += 8.0
    if "Too Dark" in warnings or "Too Bright" in warnings:
        penalty += 6.0
    if similarity_info["highest_similarity_percent"] >= 95:
        penalty += 3.0

    return clamp_score(
        quality_component
        + sharpness_component
        + resolution_component
        + exposure_component
        + similarity_component
        - penalty
    )


def build_best_shot_reason(
    image_index: int,
    best_index: int,
    group: List[int],
    quality_by_index: Dict[int, Dict],
    best_score: float,
    similarity_summary: Dict[int, Dict],
) -> str:
    """Create a plain-language explanation for the recommendation."""

    quality = quality_by_index[image_index]
    best_quality = quality_by_index[best_index]

    if image_index == best_index:
        group_quality_scores = [quality_by_index[index]["quality_score"] for index in group]
        reasons = []
        if quality["quality_score"] >= max(group_quality_scores) - 5:
            reasons.append("best overall quality")
        if quality["sharpness_score"] >= best_quality["sharpness_score"] - 1:
            reasons.append("strong sharpness")
        if quality["brightness_status"] == "Good Exposure":
            reasons.append("good exposure")
        if quality["megapixels"] >= best_quality["megapixels"]:
            reasons.append("strong resolution")
        reason_text = ", ".join(reasons[:3]) or "the strongest combined score"
        return f"Recommended because it has {reason_text} in this similar group. Best Shot score: {best_score:.2f}/100."

    reasons = []
    if quality["quality_score"] + 5 < best_quality["quality_score"]:
        reasons.append("lower quality score")
    if quality["sharpness_score"] + 5 < best_quality["sharpness_score"]:
        reasons.append("lower sharpness")
    if quality["megapixels"] < best_quality["megapixels"]:
        reasons.append("lower resolution")
    if similarity_summary[image_index]["highest_similarity_percent"] >= 95:
        reasons.append("near duplicate risk")
    reasons.extend(quality.get("warnings", []))
    reason_text = ", ".join(dict.fromkeys(reasons)) or "it is similar to the recommended image"
    return f"Consider removing because of {reason_text}."


def choose_best_shot(
    group_files: List[int],
    quality_results: Dict[int, Dict],
    pair_results: Dict[int, Dict],
) -> Dict:
    """Choose the strongest image using score plus clear tie breakers."""

    scored_rows = []
    for image_index in group_files:
        quality = quality_results[image_index]
        similarity_info = pair_results[image_index]
        score = calculate_best_shot_score(quality["filename"], quality, similarity_info)
        scored_rows.append(
            {
                "index": image_index,
                "filename": quality["filename"],
                "score": score,
                "quality_score": quality["quality_score"],
                "sharpness_score": quality["sharpness_score"],
                "megapixels": quality["megapixels"],
                "similarity": similarity_info["highest_similarity_percent"],
            }
        )

    highest_quality = max(row["quality_score"] for row in scored_rows)
    close_quality_candidates = [
        row for row in scored_rows if highest_quality - row["quality_score"] <= 5
    ]
    best_row = max(
        close_quality_candidates,
        key=lambda row: (
            row["score"],
            row["sharpness_score"],
            row["megapixels"],
            -row["similarity"],
        ),
    )
    return {
        "best_index": best_row["index"],
        "best_filename": best_row["filename"],
        "score": best_row["score"],
        "scores": {row["index"]: row["score"] for row in scored_rows},
    }


def get_best_shot_for_group(
    group: List[int],
    quality_by_index: Dict[int, Dict],
    similarity_summary: Dict[int, Dict],
) -> int:
    """Backward-friendly wrapper around the smarter Best Shot chooser."""

    return choose_best_shot(group, quality_by_index, similarity_summary)["best_index"]


def build_best_shot_details(
    groups: List[List[int]],
    quality_rows: List[Dict],
    similarity_summary: Dict[int, Dict],
) -> Dict[int, Dict]:
    """Return label, score, and reason for each image in similar groups."""

    quality_by_index = {row["index"]: row for row in quality_rows}
    details: Dict[int, Dict] = {}

    for group in groups:
        choice = choose_best_shot(group, quality_by_index, similarity_summary)
        best_index = choice["best_index"]
        best_score = choice["score"]
        for image_index in group:
            details[image_index] = {
                "label": "Recommended Keep" if image_index == best_index else "Consider Removing",
                "best_shot_score": choice["scores"][image_index],
                "best_shot_reason": build_best_shot_reason(
                    image_index=image_index,
                    best_index=best_index,
                    group=group,
                    quality_by_index=quality_by_index,
                    best_score=best_score,
                    similarity_summary=similarity_summary,
                ),
            }

    return details


def build_best_shot_recommendations(
    groups: List[List[int]],
    quality_rows: List[Dict],
    similarity_summary: Dict[int, Dict],
) -> Dict[int, str]:
    details = build_best_shot_details(groups, quality_rows, similarity_summary)
    return {image_index: data["label"] for image_index, data in details.items()}


def upload_readiness_status(
    quality_score: float,
    highest_similarity: float,
    user_decision: str,
    best_shot_recommendation: str,
) -> str:
    # User removal is always respected.
    if user_decision == "Remove":
        return "Remove Recommended"

    # Very low quality or strong near-duplicate non-best shots should be removed.
    if quality_score < 35:
        return "Remove Recommended"
    if highest_similarity >= 95 and best_shot_recommendation != "Recommended Keep":
        return "Remove Recommended"

    # Risky similarity, non-best shots, undecided decisions, or weak quality need review.
    if quality_score < 50:
        return "Review Needed"
    if best_shot_recommendation == "Consider Removing":
        return "Review Needed" if highest_similarity < 90 else "Remove Recommended"
    if user_decision == "Undecided" or highest_similarity >= 80 or quality_score < 70:
        return "Review Needed"
    return "Ready to Upload"


def get_upload_readiness_status(
    filename: str,
    quality_result: Dict,
    pair_results: List[Dict],
    best_shot_map: Dict[str, str],
    user_decisions: Dict[str, str],
) -> str:
    """Convenience wrapper using filename-based data for future report pages."""

    user_decision = user_decisions.get(filename, "Keep")
    best_recommendation = best_shot_map.get(filename, "No Similar Group")
    return upload_readiness_status(
        quality_score=quality_result["quality_score"],
        highest_similarity=get_highest_similarity_for_image(filename, pair_results),
        user_decision=user_decision,
        best_shot_recommendation=best_recommendation,
    )


def build_upload_readiness_rows(
    project_name: str,
    batch_name: str,
    images: List[UploadedImage],
    quality_rows: List[Dict],
    all_pairs: List[Dict],
    groups: List[List[int]],
    remove_indexes: List[int],
    best_shot_decisions: Dict[int, str],
    metadata_summary: Dict[str, Dict] | None = None,
) -> List[Dict]:
    images = safe_sequence(images)
    quality_rows = safe_sequence(quality_rows)
    all_pairs = safe_sequence(all_pairs)
    groups = safe_sequence(groups)
    metadata_summary = safe_mapping(metadata_summary)
    similarity_summary = build_similarity_summary(images, all_pairs)
    best_shot_recommendations = build_best_shot_recommendations(
        groups=groups,
        quality_rows=quality_rows,
        similarity_summary=similarity_summary,
    )
    best_shot_details = build_best_shot_details(
        groups=groups,
        quality_rows=quality_rows,
        similarity_summary=similarity_summary,
    )
    remove_set = set(remove_indexes)
    rows: List[Dict] = []

    for quality in quality_rows:
        index = quality["index"]
        best_recommendation = best_shot_decisions.get(index)
        if best_recommendation == "Keep":
            best_recommendation = "Recommended Keep"
        elif best_recommendation == "Remove":
            best_recommendation = "Consider Removing"
        elif best_recommendation == "Undecided":
            best_recommendation = "Review Needed"
        elif not best_recommendation:
            best_recommendation = best_shot_recommendations.get(index, "No Similar Group")

        best_detail = best_shot_details.get(
            index,
            {
                "best_shot_score": "",
                "best_shot_reason": "Not part of a similar group.",
            },
        )
        user_decision = best_shot_decisions.get(index, "Keep")
        if index in remove_set:
            user_decision = "Remove"

        similarity = similarity_summary[index]
        metadata = (metadata_summary or {}).get(quality["filename"], {})
        status = upload_readiness_status(
            quality_score=quality["quality_score"],
            highest_similarity=similarity["highest_similarity_percent"],
            user_decision=user_decision,
            best_shot_recommendation=best_recommendation,
        )
        if status != "Remove Recommended" and float(metadata.get("metadata_similarity_percent") or 0) >= 90:
            status = "Review Needed"

        reasons = []
        if user_decision == "Remove":
            reasons.append("User marked remove")
        if similarity["highest_similarity_percent"] >= 95:
            reasons.append("Near duplicate")
        elif similarity["highest_similarity_percent"] >= 80:
            reasons.append("Risky similar content")
        if quality["quality_score"] < 35:
            reasons.append("Very low quality")
        elif quality["quality_score"] < 50:
            reasons.append("Low quality")
        if best_recommendation == "Consider Removing":
            reasons.append("Not best shot in similar group")
        if float(metadata.get("metadata_similarity_percent") or 0) >= 90:
            if similarity["highest_similarity_percent"] >= 80:
                reasons.append("High Similar Content Risk")
            else:
                reasons.append("Metadata is too similar even though image is visually different")
            reasons.append("Metadata is highly similar to another image")
        reasons.extend(quality["warnings"])

        recommended_action_text = "Ready for upload review."
        if status == "Remove Recommended":
            recommended_action_text = "Remove this image from the upload batch."
        elif status == "Review Needed":
            recommended_action_text = "Review quality and similarity before uploading."

        rows.append(
            {
                "filename": quality["filename"],
                "auto_status": status,
                "auto_reason": "; ".join(dict.fromkeys(reasons)) or "Good quality and low similarity risk",
                "final_status": status,
                "included_in_auto_clean_zip": (status == "Ready to Upload" or user_decision == "Keep") and status != "Remove Recommended",
                "original_filename": quality["original_filename"],
                "internal_filename": quality["internal_filename"],
                "project_name": project_name,
                "batch_name": batch_name,
                "image_width": quality["width"],
                "image_height": quality["height"],
                "megapixels": quality["megapixels"],
                "quality_score": quality["quality_score"],
                "sharpness_score": quality["sharpness_score"],
                "brightness_score": quality["brightness_score"],
                "brightness_status": quality["brightness_status"],
                "highest_similarity_percent": similarity["highest_similarity_percent"],
                "matched_with": similarity["matched_with"],
                "similarity_risk_level": similarity["similarity_risk"],
                "best_shot_recommendation": best_recommendation,
                "best_shot_score": best_detail["best_shot_score"],
                "best_shot_reason": best_detail["best_shot_reason"],
                "user_decision": user_decision,
                "upload_readiness_status": status,
                "recommended_action": recommended_action_text,
                "reason": "; ".join(dict.fromkeys(reasons)) or "No major issue found",
                "quality_warnings": "; ".join(quality["warnings"]) or "None",
                "title": metadata.get("title", ""),
                "keywords": metadata.get("keywords", ""),
                "metadata_similarity_percent": metadata.get("metadata_similarity_percent", 0.0),
                "metadata_matched_with": metadata.get("metadata_matched_with", ""),
                "title_similarity_percent": metadata.get("title_similarity_percent", 0.0),
                "keyword_similarity_percent": metadata.get("keyword_similarity_percent", 0.0),
                "metadata_risk_level": metadata.get("metadata_risk_level", "Low metadata risk"),
                "metadata_recommendation": metadata.get("metadata_recommendation", "Metadata was not checked."),
                "processing_status": quality["processing_status"],
                # Backward-friendly aliases used by the current UI.
                "similarity_risk": similarity["similarity_risk"],
            }
        )

    return rows


def build_auto_decision_rows(
    project_name: str,
    batch_name: str,
    images: List[UploadedImage],
    quality_rows: List[Dict],
    all_pairs: List[Dict],
    groups: List[List[int]],
    remove_indexes: List[int],
    best_shot_decisions: Dict[int, str],
    metadata_summary: Dict[str, Dict] | None = None,
    near_duplicates: List[Dict] | None = None,
) -> List[Dict]:
    """Automatically classify every image while preserving manual overrides."""

    images = safe_sequence(images)
    quality_rows = safe_sequence(quality_rows)
    all_pairs = safe_sequence(all_pairs)
    groups = safe_sequence(groups)
    metadata_summary = safe_mapping(metadata_summary)
    similarity_summary = build_similarity_summary(images, all_pairs)
    best_shot_details = build_best_shot_details(groups, quality_rows, similarity_summary)
    quality_by_index = {row["index"]: row for row in quality_rows}
    best_indexes = set()
    for group in groups:
        if group and quality_by_index:
            best_indexes.add(get_best_shot_for_group(group, quality_by_index, similarity_summary))

    remove_set = set(remove_indexes)
    group_members = {index for group in groups for index in group}
    rows: List[Dict] = []

    for quality in quality_rows:
        index = quality["index"]
        similarity = similarity_summary[index]
        best_detail = best_shot_details.get(
            index,
            {
                "label": "No Similar Group",
                "best_shot_score": "",
                "best_shot_reason": "Not part of a similar group.",
            },
        )
        is_best_shot = index in best_indexes or index not in group_members
        user_decision = best_shot_decisions.get(index, "Undecided")
        if index in remove_set:
            user_decision = "Remove"

        highest_similarity = similarity["highest_similarity_percent"]
        metadata = (metadata_summary or {}).get(quality["filename"], {})
        metadata_similarity = float(metadata.get("metadata_similarity_percent") or 0)
        quality_score = quality["quality_score"]
        brightness = quality["brightness_status"]

        auto_status = "Ready to Upload"
        auto_reason = "Good quality and low similarity risk"
        if highest_similarity >= 95 and not is_best_shot:
            auto_status = "Remove Recommended"
            auto_reason = "Near duplicate and not the best shot"
        elif highest_similarity >= 90 and not is_best_shot:
            auto_status = "Review Needed"
            auto_reason = "High similarity to another image"
        elif quality_score < 35:
            auto_status = "Remove Recommended"
            auto_reason = "Low quality score"
        elif quality_score < 50:
            auto_status = "Review Needed"
            auto_reason = "Quality needs review"
        elif index in group_members and not is_best_shot:
            auto_status = "Review Needed"
            auto_reason = "Similar to recommended image"
        elif brightness in {"Too Dark", "Too Bright"} and quality_score < 60:
            auto_status = "Review Needed"
            auto_reason = f"{brightness} exposure needs review"
        if metadata_similarity >= 90 and auto_status != "Remove Recommended":
            auto_status = "Review Needed"
            if highest_similarity >= 80:
                auto_reason = f"{auto_reason}; High Similar Content Risk"
            else:
                auto_reason = "Metadata is too similar even though image is visually different"

        final_status = auto_status
        if user_decision == "Remove":
            final_status = "Remove Recommended"
            auto_reason = f"{auto_reason}; user marked Remove"
        elif user_decision == "Keep":
            final_status = "Ready to Upload" if quality_score >= 35 else "Review Needed"
        elif user_decision == "Undecided":
            final_status = auto_status

        included_in_auto_clean_zip = (
            final_status == "Ready to Upload"
            or user_decision == "Keep"
        ) and final_status != "Remove Recommended"

        rows.append(
            {
                "filename": quality["filename"],
                "project_name": project_name,
                "batch_name": batch_name,
                "auto_status": auto_status,
                "auto_reason": auto_reason,
                "user_decision": user_decision,
                "final_status": final_status,
                "highest_similarity_percent": highest_similarity,
                "matched_with": similarity["matched_with"],
                "quality_score": quality_score,
                "sharpness_score": quality["sharpness_score"],
                "brightness_status": brightness,
                "is_near_duplicate": highest_similarity >= 95,
                "is_best_shot_in_group": is_best_shot,
                "best_shot_recommendation": "Recommended Keep" if is_best_shot else "Consider Removing",
                "best_shot_score": best_detail["best_shot_score"],
                "best_shot_reason": best_detail["best_shot_reason"],
                "included_in_auto_clean_zip": included_in_auto_clean_zip,
                "image_index": index,
                "title": metadata.get("title", ""),
                "keywords": metadata.get("keywords", ""),
                "metadata_similarity_percent": metadata_similarity,
                "metadata_matched_with": metadata.get("metadata_matched_with", ""),
                "title_similarity_percent": metadata.get("title_similarity_percent", 0.0),
                "keyword_similarity_percent": metadata.get("keyword_similarity_percent", 0.0),
                "metadata_risk_level": metadata.get("metadata_risk_level", "Low metadata risk"),
                "metadata_recommendation": metadata.get("metadata_recommendation", "Metadata was not checked."),
            }
        )

    return rows


def auto_zip_remove_indexes(auto_rows: List[Dict], image_count: int) -> List[int]:
    included = {row["image_index"] for row in auto_rows if row["included_in_auto_clean_zip"]}
    return [index for index in range(image_count) if index not in included]


def render_hero(title: str = PRODUCT_NAME, subtitle: str = "") -> None:
    if not subtitle:
        subtitle = "Premium AI review workspace for safer stock uploads."
    st.markdown(
        f"""
        <div class="sg-hero">
            <div class="sg-badge">Stock Upload Review Assistant</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_custom_css() -> None:
    """Compatibility wrapper for the new UI helper name requested by the redesign."""

    inject_custom_css()


def render_page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render a premium page header with optional badge."""

    badge_html = f"<span class='sg-pill sg-pill-muted'>{escape_html(badge)}</span>" if badge else ""
    st.markdown(
        f"""
        <div class="sg-page-header">
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.55rem; margin-bottom:0.35rem;">{badge_html}</div>
            <div class="sg-page-title">{escape_html(title)}</div>
            <div class="sg-page-subtitle">{escape_html(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, text: str, badge: str = "") -> None:
    badge_html = f"<div class='sg-badge'>{escape_html(badge)}</div>" if badge else ""
    st.markdown(
        f"""
        <div class="sg-card">
            {badge_html}
            <div class="sg-card-title">{escape_html(title)}</div>
            <div class="sg-muted">{escape_html(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str, subtitle: str | None = None, body_html: str | None = None) -> None:
    """Reusable white card wrapper for consistent page sections."""

    subtitle_html = f"<div class='sg-muted' style='margin-top: 0.18rem;'> {escape_html(subtitle)}</div>" if subtitle else ""
    body_html = body_html or ""
    st.markdown(
        f"""
        <div class="sg-section-card">
            <div class="sg-card-title">{escape_html(title)}</div>
            {subtitle_html}
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(icon: str, title: str, description: str) -> None:
    """Render one feature/value proposition card."""

    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{escape_html(clean_icon(icon))}</div>
            <div class="sg-card-title">{escape_html(title)}</div>
            <p class="sg-muted">{escape_html(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, subtitle: str | None = None) -> None:
    """Shared card wrapper for a light SaaS card layout."""

    subtitle_html = f"<div class='sg-card-subtitle'>{escape_html(subtitle)}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="sg-card">
            <div class="sg-card-title">{escape_html(title)}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_light_table(rows: List[Dict], columns: List[str], status_columns: Dict[str, str] | None = None, action_columns: Dict[str, str] | None = None) -> None:
    """Render a light HTML table using consistent SaaS card styling."""

    if not rows:
        st.info("No rows to display.")
        return

    status_columns = status_columns or {}
    action_columns = action_columns or {}

    header_cells = "".join(
        f"<th>{escape_html(str(column))}</th>" for column in columns
    )

    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            text = "" if value is None else str(value)
            if column in status_columns:
                tone = status_columns[column]
                pill = "sg-pill sg-pill-success" if tone == "success" else "sg-pill sg-pill-warning" if tone == "warning" else "sg-pill sg-pill-muted"
                text = f"<span class='{pill}'>{escape_html(text)}</span>"
            elif column in action_columns:
                text = f"<span class='sg-pill sg-pill-muted'>{escape_html(text)}</span>"
            cells.append(f"<td class='sg-cell-truncate' title='{escape_html(text)}'>{text}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    st.markdown(
        """
        <div class="sg-table-wrap">
            <table class="sg-light-table">
                <thead><tr>""" + header_cells + """</tr></thead>
                <tbody>""" + "".join(body_rows) + """</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    icon: str,
    title: str,
    text: str,
    cta_text: str | None = None,
    target_page: str | None = None,
    secondary_cta_text: str | None = None,
    secondary_target_page: str | None = None,
) -> None:
    """Render a premium white-card empty state with optional primary and secondary actions."""

    st.markdown(
        f"""
        <div class="sg-empty-state" style="padding: 1.25rem; text-align: left; border: 1px solid #E2E8F0; border-radius: 18px; background: #FFFFFF; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
            <div class="icon" style="width: 2.5rem; height: 2.5rem; border-radius: 999px; display: grid; place-items: center; background: linear-gradient(135deg, #EFF6FF, #F5F3FF); color: #2563EB; font-size: 1rem; box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.18);">{escape_html(clean_icon(icon))}</div>
            <h3 style="margin: 0.55rem 0 0.3rem 0; color: #0F172A; font-size: 1.08rem;">{escape_html(title)}</h3>
            <p style="margin: 0 0 0.35rem 0; color: #64748B; line-height: 1.5;">{escape_html(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    button_columns = []
    if cta_text and target_page:
        button_columns.append((cta_text, target_page, "primary"))
    if secondary_cta_text and secondary_target_page:
        button_columns.append((secondary_cta_text, secondary_target_page, "secondary"))

    if button_columns:
        cols = st.columns(len(button_columns))
        for column, (label, page_name, kind) in zip(cols, button_columns):
            with column:
                if st.button(label, key=f"empty_state_{title}_{page_name}", type=kind, use_container_width=True):
                    st.session_state["page"] = page_name
                    st.rerun()


def render_light_html_table(df: pd.DataFrame, caption: str | None = None) -> None:
    """Render a light, readable HTML table for scan history and report views."""

    if df.empty:
        st.info("No rows available to display.")
        return

    columns = list(df.columns)
    header_cells = "".join(
        f"<th style='padding: 0.65rem 0.7rem; text-align:left; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; background: #F8FAFC; border-bottom: 1px solid #E2E8F0;'>{escape_html(str(column))}</th>"
        for column in columns
    )

    body_rows = []
    for row in df.to_dict("records"):
        cells = []
        for column in columns:
            value = row.get(column, "")
            text = "" if value is None else str(value)
            text = text.replace("\n", "<br />")
            column_name = str(column).lower()
            cell_class = "sg-cell-truncate" if any(keyword in column_name for keyword in ("file", "email", "name", "reason", "batch", "project", "report", "metadata")) else ""
            cells.append(
                f"<td class='{cell_class}' style='padding: 0.65rem 0.7rem; color: #0F172A; border-bottom: 1px solid #E2E8F0; vertical-align: top; font-size: 0.92rem; max-width: 280px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='{escape_html(text)}'>{escape_html(text)}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    caption_html = f"<div class='sg-muted' style='margin-bottom: 0.35rem;'>{escape_html(caption)}</div>" if caption else ""
    st.markdown(
        f"""
        <div class="sg-light-table-shell" style="max-width: 100%; overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 16px; background: #FFFFFF; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06); padding: 0.2rem;">
            {caption_html}
            <table style="width: 100%; border-collapse: collapse; min-width: 820px; table-layout: fixed; background: #FFFFFF;">
                <thead>
                    <tr>{header_cells}</tr>
                </thead>
                <tbody>
                    {''.join(body_rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_raw_table_expander(expander_label: str, data: pd.DataFrame | List[Dict], caption: str | None = None) -> None:
    """Render a raw-detail expander using the light StockGuard table design instead of dark dataframes."""

    if isinstance(data, pd.DataFrame):
        table_df = data.copy()
    else:
        table_df = pd.DataFrame(data)

    with st.expander(expander_label, expanded=False):
        st.markdown(
            """
            <div class="sg-card" style="padding: 0.75rem 0.85rem; margin-bottom: 0.45rem; border-radius: 16px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);">
                <div class="sg-card-title" style="margin-bottom: 0.15rem;">Raw detail view</div>
                <div class="sg-muted">Light table layout with internal scrolling only.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_light_html_table(table_df, caption=caption)


def render_status_badge(label: str | None = None, color: str = "green", text: str | None = None, tone: str = "default") -> None:
    badge_text = text or label or ""
    tone_map = {
        "ready": "success",
        "success": "success",
        "review": "warning",
        "warning": "warning",
        "remove": "danger",
        "danger": "danger",
        "locked": "muted",
        "default": color,
        "paid": "success",
        "pending": "warning",
        "rejected": "danger",
        "refunded": "muted",
    }
    badge_tone = tone_map.get(str(tone).lower(), color)
    css_class = "sg-pill"
    if badge_tone == "success":
        css_class += " sg-pill-success"
    elif badge_tone == "warning":
        css_class += " sg-pill-warning"
    elif badge_tone == "danger":
        css_class += " sg-pill-danger"
    else:
        css_class += " sg-pill-muted"
    st.markdown(f"<span class='{css_class}'>{escape_html(badge_text)}</span>", unsafe_allow_html=True)


def render_metric_cards(card_data: List[Tuple]) -> None:
    column_count = min(len(card_data), 4)
    columns = st.columns(column_count)
    for index, card in enumerate(card_data):
        if len(card) == 4:
            icon, label, value, hint = card
        else:
            label, value, hint = card
            icon = "Metric"
        column = columns[index % column_count]
        with column:
            st.markdown(
                f"""
                <div class="status-card">
                    <div class="icon">{escape_html(clean_icon(icon))}</div>
                    <div class="label">{escape_html(label)}</div>
                    <div class="value">{escape_html(value)}</div>
                    <div class="hint">{escape_html(hint)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_card(
    label: str,
    value: str,
    description: str | None = None,
    icon: str | None = None,
    tone: str = "default",
) -> None:
    """Render one premium metric card with a clean SaaS layout."""

    display_text = description or ""
    tone_class = f" tone-{tone}" if tone and tone != "default" else ""
    icon_html = f"<div class='sg-metric-icon'>{escape_html(clean_icon(icon))}</div>" if icon else ""
    description_html = f"<div class='sg-muted' style='line-height: 1.4;'> {escape_html(display_text)}</div>" if display_text else ""
    st.markdown(
        f"""
        <div class="sg-metric-card{tone_class}">
            {icon_html}
            <div class="sg-metric-label">{escape_html(label)}</div>
            <div class="sg-metric-value">{escape_html(value)}</div>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_card(number_or_icon: str, title: str, description: str, action_label: str | None = None) -> None:
    """Render a premium workflow/action card with a small badge and optional CTA."""

    action_html = f"<div style='margin-top: 0.5rem;'><span class='sg-pill sg-pill-muted'>{escape_html(action_label)}</span></div>" if action_label else ""
    st.markdown(
        f"""
        <div class="sg-action-card">
            <div class="sg-action-badge">{escape_html(clean_icon(number_or_icon))}</div>
            <div class="sg-card-title" style='margin-top: 0.35rem;'>{escape_html(title)}</div>
            <div class="sg-muted" style='line-height: 1.45;'>{escape_html(description)}</div>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_admin_metric_card(icon: str, label: str, value: str, description: str) -> None:
    """Compact admin metric card that avoids broken text wrapping."""

    st.markdown(
        f"""
        <div class="admin-metric-card">
            <div class="admin-metric-icon">{escape_html(icon)}</div>
            <div class="admin-metric-label">{escape_html(label)}</div>
            <div class="admin-metric-value">{escape_html(value)}</div>
            <div class="admin-metric-desc">{escape_html(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_admin_metric_grid(metrics: List[Tuple[str, str, str, str]]) -> None:
    columns = st.columns(4)
    for column, metric in zip(columns, metrics):
        with column:
            render_admin_metric_card(*metric)


def render_admin_plan_summary_card(plan: Dict) -> None:
    metadata_enabled = bool(plan.get("metadata_checker", plan.get("metadata_checker_enabled", 0)))
    advanced_modes_enabled = bool(plan.get("advanced_scan_modes", plan.get("advanced_scan_modes_enabled", 0)))
    features = [
        ("CSV", bool(plan.get("csv_export", plan.get("csv_export_enabled", 1)))),
        ("ZIP", bool(plan.get("zip_export", plan.get("zip_export_enabled", 0)))),
        ("Readiness", bool(plan.get("readiness_report", plan.get("readiness_report_enabled", 0)))),
        ("Best Shot", bool(plan.get("best_shot", plan.get("best_shot_enabled", 0)))),
        ("Metadata", metadata_enabled),
        ("Advanced Modes", advanced_modes_enabled),
        ("History", bool(plan.get("batch_history", plan.get("scan_history_enabled", 0)))),
        ("Client Folders", bool(plan.get("client_folders", plan.get("client_folders_enabled", 0)))),
    ]
    badges = "".join(
        f"<span class='admin-pill {'green' if enabled else 'red'}'>{escape_html(name)}</span>"
        for name, enabled in features
    )
    active_badge = "Active" if plan["is_active"] else "Inactive"
    active_class = "green" if plan["is_active"] else "red"
    st.markdown(
        f"""
        <div class="admin-plan-card">
            <div class="admin-badges"><span class="admin-pill {active_class}">{active_badge}</span></div>
            <div class="admin-plan-name">{escape_html(plan['plan_name'])}</div>
            <div class="admin-plan-price">{escape_html(plan_price_label(plan))}</div>
            <div class="admin-muted">{plan['monthly_scans']} scans/month</div>
            <div class="admin-muted">{plan['images_per_scan']} images/scan</div>
            <div class="admin-feature-list">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_locked_feature_card(title: str, description: str, plan_required: str) -> None:
    """Render a premium locked feature card for plan-gated pages."""

    st.markdown(
        f"""
        <div class="locked-card">
            <div class="sg-badge">Locked Feature</div>
            <h2>{escape_html(title)}</h2>
            <p class="sg-muted">{escape_html(description)}</p>
            <div class="status-badge purple">{escape_html(plan_required)} required</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plan_badge(plan_name: str) -> None:
    """Render a compact plan badge."""

    st.markdown(
        f"<span class='sg-plan-badge'>{escape_html(plan_name)} Plan</span>",
        unsafe_allow_html=True,
    )


def render_status_card(icon: str, title: str, value: str, description: str, tone: str = "blue") -> None:
    """Render a premium status/metric card with an accent line."""

    st.markdown(
        f"""
        <div class="sg-status-card tone-{tone}">
            <div class="sg-status-icon">{escape_html(clean_icon(icon))}</div>
            <div class="sg-status-value">{escape_html(value)}</div>
            <div class="sg-status-title">{escape_html(title)}</div>
            <p>{escape_html(description)}</p>
            <div class="sg-accent-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_card(title: str, description: str) -> None:
    """Render copy for the upload area while keeping Streamlit uploader active."""

    st.markdown(
        f"""
        <div class="sg-upload-shell">
            <div class="sg-upload-cloud">Upload</div>
            <div class="sg-card-title">{escape_html(title)}</div>
            <p class="sg-muted">{escape_html(description)}</p>
            <p class="sg-upload-helper">Supports JPG, JPEG, PNG • Large batches supported</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_export_card(title: str, description: str, badge: str = "") -> None:
    """Render a premium export/action card."""

    badge_html = f"<span class='status-badge purple'>{escape_html(badge)}</span>" if badge else ""
    st.markdown(
        f"""
        <div class="export-card">
            {badge_html}
            <div class="sg-card-title">{escape_html(title)}</div>
            <p class="sg-muted">{escape_html(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_review_card(
    filename: str,
    quality_score: float,
    similarity_percent: float,
    status: str,
    recommended: bool = False,
) -> None:
    """Render metadata for one image review card."""

    badge = "Recommended Keep" if recommended else "Consider Removing"
    tone = "green" if recommended else "yellow"
    glow = " recommended" if recommended else ""
    st.markdown(
        f"""
        <div class="image-review-card{glow}">
            <span class="status-badge {tone}">{escape_html(badge)}</span>
            <div class="sg-card-title">{escape_html(filename)}</div>
            <p class="sg-muted">Quality {quality_score:.2f}/100 | Similarity {similarity_percent:.2f}%</p>
            <span class="status-badge purple">{escape_html(status)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similar_group_card(group_number: int, image_count: int, reason: str) -> None:
    """Render a premium similar-group header card."""

    st.markdown(
        f"""
        <div class="similar-group-card">
            <div class="sg-status-icon">Group</div>
            <div class="sg-card-title">Similar Group #{group_number}</div>
            <p class="sg-muted">{image_count} images | Best match found</p>
            <span class="status-badge purple">{escape_html(reason)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(user: Dict) -> str:
    """Public helper name for the redesigned sidebar."""

    return sidebar_navigation(user)


def nav_display_label(label: str) -> str:
    """Return the visible sidebar label used in the UI."""

    mapping = {
        "DB": "Dashboard",
        "NS": "New Scan",
        "SH": "My Scans",
        "BS": "Best Shot Selector",
        "UR": "Upload Readiness Report",
        "AS": "Auto Scan Summary",
        "EX": "Downloads",
        "SP": "Scan Profiles",
        "SUB": "Subscription",
        "BH": "Billing History",
        "ADM": "Admin Panel",
        "OUT": "Logout",
        "Scan History": "My Scans",
        "My Exports": "Downloads",
    }
    cleaned = str(label or "").strip()
    return mapping.get(cleaned, cleaned)


def render_topbar(user: Dict, page: str) -> None:
    """Render the light SaaS top header in the main workspace."""

    plan = get_current_plan(user)
    name = user.get("name") or user.get("email", "User")
    role = "Admin" if is_admin_user(user) else "Contributor"
    cols = st.columns([4, 1])
    page_label = nav_display_label(page)
    avatar_html = _render_avatar_html(user, size=36)

    with cols[0]:
        st.markdown(
            f"""
            <div class="sg-topbar">
                <div>
                    <div class="sg-topbar-title">{escape_html(page_label)}</div>
                    <div class="sg-topbar-subtitle">Current workspace · {escape_html(role)} · {escape_html(plan['plan_name'])} plan</div>
                </div>
                <div class="sg-topbar-right">
                    <span class="sg-pill sg-pill-success">{escape_html(plan['plan_name'])} Plan</span>
                    <span class="sg-pill sg-pill-muted">Help</span>
                    <span class="sg-pill sg-pill-muted">Notifications</span>
                    <span class="sg-muted">Hello, {escape_html(name)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        components.html(
            f"""
            <form action="" method="GET" target="_top" style="display:contents">
                <input type="hidden" name="nav" value="profile">
                <button type="submit" style="border:none; background:none; padding:0; cursor:pointer; display:flex; align-items:center; justify-content:center; height:100%; width:100%;">
                    {avatar_html}
                </button>
            </form>
            """,
            height=44,
            width=60,
        )


def render_app_footer() -> None:
    """Subtle product footer for logged-in pages."""

    st.markdown(
        f"""
        <div class="sg-footer">
            <div>{escape_html(LANDING_FOOTER_TEXT)}</div>
            <div style="margin-top: 0.2rem;">StockGuard AI is an independent tool and is not affiliated with Adobe.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Public footer helper for the polished app layout."""

    render_app_footer()


def privacy_policy_page(user: Dict) -> None:
    """Render a simple, customer-friendly Privacy Policy page."""

    st.markdown(
        """
        <div class="sg-saas-card" style="padding: 1.1rem 1rem 1rem 1rem; margin-bottom: 1rem;">
            <div class="sg-card-header-row">
                <div>
                    <h2 style="margin: 0 0 0.2rem 0; color: #0F172A;">Privacy Policy</h2>
                    <p class="sg-muted" style="margin: 0;">StockGuard AI is built to help stock contributors review image batches before upload.</p>
                </div>
                <span class="sg-plan-badge">StockGuard AI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sg-saas-card" style="padding: 1rem 1rem 1.1rem 1rem;">
            <h3 style="margin-top: 0; color: #111827;">StockGuard AI Privacy Policy</h3>
            <p class="sg-muted">This page explains how the app handles images, reports, and account information in plain language.</p>
            <ol style="padding-left: 1rem; line-height: 1.5; color: #334155;">
                <li><strong>Uploaded images</strong> — Uploaded images are processed temporarily for similarity analysis, duplicate detection, weak variation detection, and clean ZIP generation.</li>
                <li><strong>Storage</strong> — StockGuard AI does not permanently store uploaded images after the scan/export process, unless a future feature clearly says otherwise.</li>
                <li><strong>Reports and exports</strong> — CSV reports and clean ZIP files are generated for the user during the scan session. Users should download them before leaving the session unless saved history or downloads are available on their plan.</li>
                <li><strong>Account data</strong> — Basic account and subscription information may be stored to provide login, plan limits, and app access.</li>
                <li><strong>No sale of user files</strong> — StockGuard AI does not sell uploaded images or image data.</li>
                <li><strong>Independent tool</strong> — StockGuard AI is an independent tool and is not affiliated with Adobe.</li>
                <li><strong>Contact</strong> — For privacy questions, contact BEE CLUSTER.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("This privacy page is for general information only and may be updated as the product evolves.")

    col_back, col_home = st.columns([1, 1])
    with col_back:
        if st.button("Back to New Scan", key="privacy_policy_back_to_new_scan", use_container_width=True):
            st.session_state["page"] = "New Scan"
            st.rerun()
    with col_home:
        if st.button("Open Subscription", key="privacy_policy_open_subscription", use_container_width=True):
            st.session_state["page"] = "Subscription"
            st.rerun()


def render_workflow_steps(active_step: int) -> None:
    """Show the simple StockGuard AI workflow without technical wording."""

    steps = [
        ("Upload Images", "Prepare a batch of stock images for review."),
        ("Scan & Review", "Detect duplicates, weak variations, quality issues, and metadata risk."),
        ("Export Clean Batch", "Download only the selected safe images and reports."),
    ]
    columns = st.columns(3, gap="medium")
    for index, (column, (label, description)) in enumerate(zip(columns, steps), start=1):
        with column:
            status = "Active" if index == active_step else "Next"
            render_action_card(str(index), label, description, status)


def render_filtered_image_list(
    title: str,
    rows: List[Dict],
    images: List[UploadedImage],
    empty_text: str,
) -> None:
    """Show filtered readiness images as a compact thumbnail grid."""

    st.subheader(title)
    if not rows:
        render_empty_state("Images", title, empty_text)
        return

    images_by_name = {image.name: image for image in images}
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "Filename": row["filename"],
                "Status": row.get("final_status") or row.get("upload_readiness_status"),
                "Quality": f"{float(row.get('quality_score') or 0):.2f}",
                "Highest similarity": f"{float(row.get('highest_similarity_percent') or 0):.2f}%",
                "Reason": row.get("auto_reason") or row.get("reason") or "",
            }
        )
    render_light_html_table(pd.DataFrame(table_rows), caption="Filtered image summary")

    preview_rows = rows[:24]
    for start in range(0, len(preview_rows), 4):
        columns = st.columns(4)
        for column, row in zip(columns, preview_rows[start : start + 4]):
            with column:
                image = images_by_name.get(row["filename"])
                if image:
                    st.image(image.thumbnail, caption=row["filename"], use_container_width=True)
                st.caption(row.get("auto_reason") or row.get("reason") or "")


def settings_page(user: Dict) -> None:
    """Profile Settings page with display name, avatar upload/crop, and account info."""

    user = get_current_user()
    plan = get_current_plan(user)
    render_page_header(
        "Settings",
        "Manage your account, plan, and privacy controls in one clean view.",
        "Settings",
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_section_card("Profile Photo", "Upload or update your profile picture.")
        avatar_html = _render_avatar_html(user, size=120, font_size="2rem")
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:center; gap:0.75rem; margin-bottom:1rem;">
                {avatar_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_photo = st.file_uploader(
            "Choose a profile photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="profile_photo_upload",
            accept_multiple_files=False,
        )

        if uploaded_photo is not None:
            error = validate_profile_photo(uploaded_photo)
            if error:
                st.error(error)
            else:
                crop_mode = st.selectbox(
                    "Crop mode",
                    options=["center", "top", "bottom", "left", "right"],
                    format_func=lambda x: {"center": "Center crop", "top": "Top crop", "bottom": "Bottom crop", "left": "Left crop", "right": "Right crop"}[x],
                    key="profile_crop_mode",
                )
                st.caption("The image will be cropped to a square based on the selected mode, then resized to 512x512.")
                if st.button("Save Profile Photo", key="save_profile_photo", type="primary", use_container_width=True):
                    try:
                        image_bytes = process_profile_photo(uploaded_photo, crop_mode)
                        delete_profile_photo_file(user["id"], user)
                        file_path = save_profile_photo(user["id"], image_bytes)
                        update_user_profile(user_id=user["id"], profile_photo_path=file_path)
                        st.session_state.pop("profile_photo_upload", None)
                        st.session_state.pop("profile_crop_mode", None)
                        st.success("Profile photo saved!")
                        st.rerun()
                    except Exception:
                        st.error("Failed to save profile photo. Please try again.")

        if user.get("profile_photo_path"):
            if st.button("Remove Profile Photo", key="remove_profile_photo", use_container_width=True):
                try:
                    delete_profile_photo_file(user["id"], user)
                    update_user_profile(user_id=user["id"], profile_photo_path="")
                    st.session_state.pop("profile_photo_upload", None)
                    st.session_state.pop("profile_crop_mode", None)
                    st.success("Profile photo removed.")
                    st.rerun()
                except Exception:
                    st.error("Failed to remove profile photo.")

        with col_right:
            render_section_card("Display Name", "Set your display name shown across the app.")
            current_display = user.get("display_name") or user.get("name") or ""
            new_display = st.text_input("Display name", value=current_display, key="profile_display_name")
            if st.button("Save Profile", key="save_profile_settings", type="primary", use_container_width=True):
                val = new_display.strip()
                if val:
                    try:
                        update_user_profile(user_id=user["id"], display_name=val)
                        st.session_state.pop("profile_display_name", None)
                        st.success("Profile updated!")
                        st.rerun()
                    except Exception:
                        st.error("Failed to save display name.")
                else:
                    st.error("Display name cannot be empty.")

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        render_section_card("Account Details", "Your email and current plan are managed separately.")
        render_metric_card("Email Address", escape_html(user.get("email", "")), "Read-only", "✉️")
        render_metric_card("Current Plan", f"{escape_html(plan['plan_name'])} Plan", "Read-only", "📋")
        render_metric_card("Batch Limit", f"{escape_html(plan['images_per_scan'])} images per scan", "Per-batch upload cap", "📸")

    st.markdown("---")
    st.caption("Profile photos are stored for your account. Scan images are still processed in memory only and are not saved to disk.")


def feature_placeholder_page(user: Dict, title: str, subtitle: str, actions: List[Tuple[str, str]]) -> None:
    """Polished placeholder/shortcut page for workflow sections that live inside scans, handling empty states and redirects."""

    render_page_header(title, subtitle, None)

    if title == "My Exports":
        st.info("Clean ZIP and CSV files are generated after each scan. Open your latest scan result to download them.")

        latest_exports = st.session_state.get("latest_scan_exports") or {}
        latest_zip = latest_exports.get("clean_zip")
        latest_csv = latest_exports.get("csv_report")
        if latest_zip or latest_csv:
            st.caption(f"Latest temporary exports: {latest_exports.get('batch_name') or 'Recent scan'}")
            zip_column, csv_column = st.columns(2, gap="medium")
            with zip_column:
                st.download_button(
                    "Download Latest Clean ZIP",
                    data=latest_zip or b"",
                    file_name="clean_stockguard_client_batch.zip",
                    mime="application/zip",
                    type="primary",
                    disabled=not latest_zip,
                    key="downloads_page_clean_zip",
                    use_container_width=True,
                )
            with csv_column:
                st.download_button(
                    "Download Latest CSV Report",
                    data=latest_csv or b"",
                    file_name="upload_readiness_report.csv",
                    mime="text/csv",
                    disabled=not latest_csv,
                    key="downloads_page_csv_report",
                    use_container_width=True,
                )

    result = st.session_state.get("last_scan_result")
    has_active_scan = bool(result)

    if has_active_scan:
        batch_name = result.get("batch_name", "Active Batch")
        project_name = result.get("project_name", "Default Project")
        render_empty_state(
            "⚡",
            "Active Scan Loaded",
            f"An active scan for {project_name} / {batch_name} is currently loaded. You can access the {title} workflow directly from the latest scan results.",
            cta_text="Go to Scan Results",
            target_page="New Scan",
        )
    else:
        render_empty_state(
            "✨",
            "No active scan data found",
            "This workflow requires an active batch scan. Upload a new batch or open a previous report to activate this view.",
            cta_text="Start New Scan",
            target_page="New Scan",
            secondary_cta_text="Open Scan History",
            secondary_target_page="Scan History",
        )

    st.markdown(
        """
        <div class="sg-card" style="margin-top: 1rem; padding: 0.95rem 1rem; border-radius: 18px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
            <div class="sg-card-title" style="margin-bottom: 0.25rem;">Workflow status</div>
            <div class="sg-muted">This view stays lightweight and ready for the next scan or report. The same logic path is preserved; only the presentation is being refined.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_columns = st.columns(3, gap="medium")
    with status_columns[0]:
        render_info_card(
            "Current state",
            "Active scan loaded" if has_active_scan else "Waiting for a new batch scan",
            "Ready" if has_active_scan else "Pending",
        )
    with status_columns[1]:
        render_info_card(
            "Primary action",
            "Open the latest report to continue your review flow." if has_active_scan else "Start a new batch to generate your next report.",
            "Next step",
        )
    with status_columns[2]:
        render_info_card(
            "Best use",
            "Review report-ready batches and export outputs from this workspace." if title == "My Exports" else "Use this section to keep the report workflow visible and easy to revisit.",
            "Helpful",
        )

    if title == "My Exports":
        st.markdown(
            """
            <div class="sg-card" style="margin-top: 1rem; padding: 1rem; border: 1px solid #E2E8F0; border-radius: 18px; background: #FFFFFF; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
                <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">Data Privacy & Export Notes</h4>
                <p style="margin: 0; color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    Your ZIP and CSV exports are generated in-memory. We do not store your original images or generated ZIP archives on our servers. Once you close this tab or log out, temporary files are destroyed and cannot be recovered.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def scan_profiles_page(user: Dict) -> None:
    render_page_header(
        "Scan Profiles",
        "Choose a preset that fits your stock upload workflow. Each option keeps the same scanning logic and changes only the visible review mode.",
        "Profiles",
    )

    render_section_card("Active Presets", "Use the preset that best matches your current submission goals.")

    profiles = [
        ("Broad Review", "75%", "Finds more possible matches. Best for checking large AI batches.", "Broad Review"),
        ("Balanced / Recommended", "85%", "Recommended for most stock upload batches.", "Balanced Recommended"),
        ("Strict / High Sensitivity", "90%", "Safer for stock uploads. Highlights strong similar-content risk.", "Strict"),
        ("Near Duplicate Only", "95%", "Only catches almost identical or near-duplicate images.", "Near Duplicate Only"),
    ]

    columns = st.columns(2, gap="medium")
    for index, (title, threshold, description, mode_key) in enumerate(profiles):
        with columns[index % 2]:
            st.markdown(
                f"""
                <div class="sg-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between; padding: 1rem; border: 1px solid #E2E8F0; border-radius: 18px; background: #FFFFFF; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <div class="sg-card-title" style="font-size: 1.02rem; margin: 0;">{escape_html(title)}</div>
                            <span class="sg-pill sg-pill-success" style="font-size: 0.82rem; font-weight: 800;">{threshold}</span>
                        </div>
                        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.45; margin: 0 0 0.75rem 0;">{escape_html(description)}</p>
                    </div>
                    <div style="display:flex; justify-content:flex-start;">
                        <span class="sg-pill sg-pill-muted">Preset ready</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Use {title}", key=f"use_profile_{index}", use_container_width=True, type="primary"):
                st.session_state["scan_mode_radio"] = mode_key
                st.session_state["page"] = "New Scan"
                st.success(f"Selected preset: {title}. Redirecting...")
                st.rerun()

    render_section_card(
        "Custom Profiles",
        "Custom Profiles Coming Soon",
        "<div style='display:flex; flex-direction:column; gap:0.35rem;'><div style='font-size:1.35rem; color:#2563EB;'>⚙️</div><div class=\"sg-muted\">In a future update, you will be able to save custom thresholds, metadata weights, and quality presets for repeatable review workflows.</div></div>",
    )


def my_profile_page(user: Dict) -> None:
    plan = get_current_plan(user)
    render_page_header("My Profile", "Review your account information and usage limits.", "Profile")

    user = get_current_user()
    col1, col2 = st.columns([1, 2])
    with col1:
        avatar_html = _render_avatar_html(user, size=96, font_size="1.6rem")
        display_name = user.get("display_name") or user.get("name") or "No name"
        render_section_card("Profile", f"{escape_html(display_name)} • {escape_html(plan['plan_name'])} Plan", f"<div style='display:flex; flex-direction:column; align-items:center; gap:0.45rem;'>{avatar_html}<span class='sg-pill sg-pill-success'>{escape_html(plan['plan_name'])} Plan</span></div>")
    with col2:
        render_section_card("Account Details", "Current account value and plan details.")
        render_metric_card("Email Address", escape_html(user.get("email", "")), "Primary sign-in address", "✉️")
        render_metric_card("Account Role", escape_html(user.get("role", "contributor")).title(), "Current role in the workspace", "🧭")
        render_metric_card("Member Since", escape_html(user.get("created_at", "N/A")), "Account creation date", "🗓️")
        render_metric_card("Batch Image Limit", f"{plan['images_per_scan']} images per scan", "Per-batch upload cap", "📸")


def billing_history_page(user: Dict) -> None:
    render_page_header("Billing History", "Review your payment requests, plan status, and upgrade history in one readable view.", "Billing")

    payments = get_payments(search_query=user["email"], limit=50)
    payments = [p for p in payments if int(p["user_id"] or 0) == int(user["id"])]

    if not payments:
        render_empty_state(
            "Billing",
            "No billing history found",
            "You are on the Free tier right now. Upgrade your plan to unlock paid features and payment history.",
            cta_text="Explore Paid Plans",
            target_page="Subscription",
            secondary_cta_text="Open Scan History",
            secondary_target_page="Scan History",
        )
        return

    render_section_card(
        "Your Payment Requests",
        "Keep manual payment approvals and plan updates in one light, readable table.",
    )

    table_rows = []
    for p in payments:
        status = (p["payment_status"] or "pending").strip().lower()
        if status == "paid":
            status_label = "Paid"
        elif status in ("rejected", "failed"):
            status_label = "Rejected"
        else:
            status_label = "Pending"
        table_rows.append({
            "Payment Ref": p["payment_ref"],
            "Plan": p["plan_name"],
            "Amount": f"{float(p['amount'] or 0):.2f} {p['currency']}",
            "Status": status_label,
            "Method": p.get("payment_method") or "Manual",
            "Date": p["created_at"],
        })

    render_raw_table_expander("View raw billing table", pd.DataFrame(table_rows), caption="Billing history details")

    if st.button("Request New Upgrade", key="billing_request_upgrade", type="primary", use_container_width=True):
        st.session_state["page"] = "Subscription"
        st.rerun()


def api_access_page(user: Dict) -> None:
    render_page_header("API Access", "Future automation access for scan workflows and integrations.", "Developer")

    st.markdown(
        """
        <div class="sg-card" style="padding: 1rem; margin-bottom: 0.9rem;">
            <div class="sg-card-title" style="margin-bottom: 0.25rem;">Coming soon</div>
            <div class="sg-muted">This beta release keeps the contributor workflow in the web app. Future API access will support automation, export hooks, and developer integrations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_columns = st.columns(3)
    with feature_columns[0]:
        render_info_card("Automation", "Use API keys to launch scans and fetch summary data in a future release.", "Planned")
    with feature_columns[1]:
        render_info_card("Exports", "Connect reports, CSV details, and Clean ZIP export workflows to external tooling.", "Planned")
    with feature_columns[2]:
        render_info_card("Billing", "Manage subscription and payment actions in a dedicated developer API layer later on.", "Future")

    render_locked_feature_card(
        "API Access is planned for a future version",
        "The current beta is focused on the web app workflow, manual payments, and contributor-facing reports.",
        "Future",
    )


def render_scan_card(scan: Dict) -> None:
    """Render one scan history/recent scan card."""

    st.markdown(
        f"""
        <div class="scan-card">
            <div class="sg-card-title">{scan["batch_name"]}</div>
            <div class="sg-muted">{scan.get("project_name") or "No Project"} | {scan["scan_datetime"]}</div>
            <p>
                Images: {scan["total_images"]} &nbsp;|&nbsp;
                Risky pairs: {scan["risky_pairs_count"]} &nbsp;|&nbsp;
                Near duplicates: {scan["near_duplicate_count"]} &nbsp;|&nbsp;
                Highest: {scan["highest_similarity_score"]:.2f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_card(scan: Dict) -> None:
    """Render one scan-history card. Kept separate for UI readability."""

    render_scan_card(scan)


def render_scan_result_card(title: str, value: str, description: str, tone: str = "default") -> None:
    """Render one scan-result summary card."""

    render_metric_card(title, value, description=description, tone=tone)


def render_project_card(project: Dict) -> None:
    """Render one project/client folder card."""

    st.markdown(
        f"""
        <div class="project-card">
            <div class="sg-card-title">{project["name"]}</div>
            <div class="sg-muted">Created {project["created_at"]}</div>
            <p>{project["scan_count"]} saved scan(s)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plan_feature_rows(plan: Dict) -> List[Tuple[str, str, bool]]:
    """Build pricing features from SQLite-backed plan flags."""

    enabled_icon = "✅"
    disabled_icon = "❌"
    feature_flags = [
        ("CSV export", bool(plan.get("csv_export"))),
        ("Clean ZIP export", bool(plan.get("zip_export"))),
        ("Upload Readiness Report", bool(plan.get("readiness_report"))),
        ("Best Shot Selector", bool(plan.get("best_shot"))),
        ("Metadata Similarity Checker", bool(plan.get("metadata_checker"))),
        ("Advanced Scan Modes", bool(plan.get("advanced_scan_modes"))),
        ("Full batch history", bool(plan.get("batch_history"))),
        ("Project folders", bool(plan.get("project_folders"))),
        ("Client folders", bool(plan.get("client_folders"))),
    ]
    rows = [
        (enabled_icon, f"{plan['monthly_scans']} scans/month", True),
        (enabled_icon, f"{plan['images_per_scan']} images/scan", True),
    ]
    rows.extend(
        (enabled_icon if enabled else disabled_icon, label, enabled)
        for label, enabled in feature_flags
    )
    return rows


def plan_card_description(plan: Dict) -> str:
    name = plan.get("plan_name", "").lower()
    if name == "free":
        return "Try small batches with basic similarity review and CSV export."
    if name == "starter":
        return "Regular contributor scans with CSV reports and review tools."
    if name == "pro":
        return "Larger batches with Clean ZIP export and saved scan history."
    if name == "agency":
        return "Team-scale workflows with client folders and priority features."
    return "Built for clean, confident stock contributor batch reviews."


def render_plan_card(
    plan_name: str,
    plan: Dict,
    price: str,
    accent: str,
    is_popular: bool = False,
    is_current: bool = False,
) -> None:
    """Render one subscription pricing card."""

    card_classes = ["pricing-card"]
    if is_popular:
        card_classes.append("recommended")
    if is_current:
        card_classes.append("current")

    badges = []
    if is_current:
        badges.append("<span class='pricing-status-badge current'>&#10003; Current Plan</span>")
    elif is_popular:
        badges.append("<span class='pricing-status-badge recommended'>Recommended</span>")
    if accent and accent not in {"Most Popular", "Recommended"}:
        badges.append(f"<span class='pricing-status-badge muted'>{escape_html(accent)}</span>")

    feature_items = plan_feature_rows(plan)
    features_html = "".join(
        f"<li class='{'enabled' if enabled else 'disabled'}'>"
        f"<span class='pricing-feature-icon'>{'&#10003;' if enabled else '&#8212;'}</span>"
        f"<span>{escape_html(label)}</span></li>"
        for _icon, label, enabled in feature_items
    )
    st.markdown(
        f"""
        <div class="{' '.join(card_classes)}">
            <div class="pricing-card-topline">
                <div class="pricing-card-name">{escape_html(plan_name)}</div>
                <div class="pricing-card-badges">{''.join(badges)}</div>
            </div>
            <div class="pricing-card-price">{escape_html(price)}</div>
            <div class="pricing-card-limits">
                <span>{int(plan['monthly_scans'])} scans / month</span>
                <span>{int(plan['images_per_scan'])} images / scan</span>
            </div>
            <p class="pricing-card-description">{escape_html(plan_card_description(plan))}</p>
            <div class="pricing-feature-title">Included features</div>
            <ul class="pricing-feature-list">{features_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scan_metric_cards(
    total_images: int,
    risky_pair_count: int,
    group_count: int,
    highest_similarity: float,
    near_duplicate_count: int,
) -> None:
    render_metric_cards(
        [
            ("Img", "Total uploaded images", str(total_images), "Valid files included in this scan."),
            ("Risk", "Risky pairs", str(risky_pair_count), "Pairs at or above your selected threshold."),
            ("Grp", "Similar groups", str(group_count), "Connected image sets found by grouping."),
            ("Top", "Highest similarity score", f"{highest_similarity:.2f}%", "Strongest match in this batch."),
            ("Near", "Near duplicates count", str(near_duplicate_count), "Pairs at 95% or higher."),
        ]
    )


def render_thumbnails(images: List[UploadedImage]) -> None:
    st.subheader("Uploaded Image Preview")
    preview_limit = 30
    visible_images = images[:preview_limit]
    st.markdown(
        f"""
        <div class="sg-upload-count">
            <strong>{len(images)} images uploaded</strong>
            <span>{'Showing first 30 thumbnails.' if len(images) > preview_limit else 'Showing all uploaded thumbnails.'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns_per_row = 5

    def render_preview_grid(preview_images: List[UploadedImage], key_prefix: str = "main") -> None:
        for start in range(0, len(preview_images), columns_per_row):
            columns = st.columns(columns_per_row)
            for column, item in zip(columns, preview_images[start : start + columns_per_row]):
                with column:
                    size_kb = len(item.bytes_data) / 1024
                    caption = f"{item.name[:36]}{'...' if len(item.name) > 36 else ''}\n{size_kb:.1f} KB"
                    st.image(item.thumbnail, caption=caption, use_container_width=True)

    render_preview_grid(visible_images)
    if len(images) > preview_limit:
        st.caption(f"Showing first {preview_limit} thumbnails out of {len(images)} uploaded images.")
        with st.expander("View more uploaded images"):
            render_preview_grid(images[preview_limit: min(len(images), 120)], "more")
            if len(images) > 120:
                st.info("Preview is limited to 120 thumbnails for performance. All accepted images are still scanned.")


def render_metadata_input(images: List[UploadedImage], plan: Dict) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Collect optional title/keyword metadata and calculate risky metadata pairs."""

    st.subheader("Metadata Similarity Checker")
    st.caption("Check whether titles and keywords are too repetitive across your batch.")
    if not plan.get("metadata_checker"):
        render_locked_feature_card(
            "Metadata Similarity Checker is available on Pro and above",
            "Images with repeated titles and keywords may look like weak variations. Upgrade or ask admin to enable this feature for your plan.",
            "Pro",
        )
        return {}, []

    st.warning(
        "Images with repeated titles and keywords may look like weak variations. Rewrite metadata before upload."
    )
    st.caption("StockGuard AI estimates metadata similarity risk. It does not guarantee acceptance or rejection.")

    metadata_df = pd.DataFrame(
        {
            "filename": [image.name for image in images],
            "title": ["" for _ in images],
            "keywords": ["" for _ in images],
        }
    )
    csv_file = st.file_uploader(
        "Optional metadata CSV",
        type=["csv"],
        key="metadata_csv_upload",
        help="CSV columns required: filename, title, keywords",
    )
    if csv_file is not None:
        try:
            uploaded_metadata = pd.read_csv(csv_file).fillna("")
            required = {"filename", "title", "keywords"}
            if not required.issubset(set(uploaded_metadata.columns)):
                st.error("Metadata CSV must include filename, title, and keywords columns.")
            else:
                by_filename = {
                    str(row["filename"]): {
                        "title": str(row["title"]),
                        "keywords": str(row["keywords"]),
                    }
                    for _, row in uploaded_metadata.iterrows()
                }
                matched_rows = []
                for image in images:
                    values = by_filename.get(image.name, {})
                    matched_rows.append(
                        {
                            "filename": image.name,
                            "title": values.get("title", ""),
                            "keywords": values.get("keywords", ""),
                        }
                    )
                metadata_df = pd.DataFrame(matched_rows)
                unmatched = sorted(set(by_filename) - {image.name for image in images})
                if unmatched:
                    st.warning("Metadata CSV contains filenames not in this upload: " + ", ".join(unmatched[:10]))
        except Exception as exc:
            st.error(f"Could not read metadata CSV: {exc}")

    edited_df = st.data_editor(
        metadata_df,
        use_container_width=True,
        hide_index=True,
        disabled=["filename"],
        column_config={
            "filename": st.column_config.TextColumn("filename"),
            "title": st.column_config.TextColumn("title"),
            "keywords": st.column_config.TextColumn("keywords"),
        },
        key=f"metadata_editor_{len(images)}_{abs(hash('|'.join(image.name for image in images)))}",
    )

    metadata_rows = {}
    missing = []
    for _, row in edited_df.fillna("").iterrows():
        filename = str(row["filename"])
        title = str(row["title"]).strip()
        keywords = str(row["keywords"]).strip()
        metadata_rows[filename] = {"title": title, "keywords": keywords}
        if not title and not keywords:
            missing.append(filename)
    if missing:
        st.warning(f"{len(missing)} image(s) do not have metadata yet.")

    metadata_pairs = build_metadata_similarity_pairs(images, metadata_rows)
    return metadata_rows, metadata_pairs


def render_metadata_similarity_report(
    images: List[UploadedImage],
    metadata_rows: Dict[str, Dict],
    metadata_pairs: List[Dict],
    csv_export_enabled: bool,
) -> None:
    """Render metadata risk summary and optional CSV report."""

    metadata_rows = safe_mapping(metadata_rows)
    metadata_pairs = safe_sequence(metadata_pairs)

    st.subheader("Metadata Similarity Check")
    st.caption("Check whether titles and keywords are too repetitive across your batch.")
    images_with_metadata = sum(
        1
        for image in images
        if metadata_rows.get(image.name, {}).get("title") or metadata_rows.get(image.name, {}).get("keywords")
    )
    missing_metadata = len(images) - images_with_metadata
    high_pairs = [pair for pair in metadata_pairs if float(pair.get("metadata_similarity_percent", 0) or 0) >= 90]
    review_pairs = [pair for pair in metadata_pairs if 75 <= float(pair.get("metadata_similarity_percent", 0) or 0) < 90]

    render_metric_cards(
        [
            ("Meta", "Images with metadata", str(images_with_metadata), "Images with title or keywords."),
            ("Missing", "Missing metadata", str(missing_metadata), "Images without title/keywords."),
            ("High", "High metadata risk", str(len(high_pairs)), "Pairs at 90% or higher."),
            ("Review", "Metadata review needed", str(len(review_pairs)), "Pairs from 75% to 89%."),
        ]
    )
    if images_with_metadata == 0:
        st.info("Metadata similarity was not checked because no titles/keywords were provided.")
        return
    if not metadata_pairs:
        st.success("No risky repeated metadata was found at the review threshold.")
        return

    render_light_html_table(
        pd.DataFrame(
            [
                {
                    "Image 1": pair["image_1"],
                    "Image 2": pair["image_2"],
                    "Title similarity": f"{pair['title_similarity_percent']:.2f}%",
                    "Keyword similarity": f"{pair['keyword_similarity_percent']:.2f}%",
                    "Overall metadata similarity": f"{pair['metadata_similarity_percent']:.2f}%",
                    "Risk": pair["risk_level"],
                    "Recommendation": pair["recommendation"],
                }
                for pair in metadata_pairs
            ]
        ),
        caption="Metadata similarity details",
    )
    if csv_export_enabled:
        st.download_button(
            "Download Metadata Similarity Report",
            data=csv_download_bytes(pd.DataFrame(metadata_pairs)),
            file_name="metadata_similarity_report.csv",
            mime="text/csv",
            key="download_metadata_similarity_report",
        )


def render_action_summary(
    risky_pairs: List[Dict],
    near_duplicates: List[Dict],
    highest_similarity: float,
) -> None:
    if not risky_pairs:
        st.success(
            "No risky pairs were found at the selected threshold. "
            "This batch looks cleaner for similarity risk, but approval is not guaranteed."
        )
        return

    if near_duplicates:
        message = (
            f"Review near duplicates first. This batch has {len(near_duplicates)} pair(s) "
            f"at or above 95%, with a highest score of {highest_similarity:.2f}%."
        )
    else:
        message = (
            f"This batch has {len(risky_pairs)} risky pair(s). "
            "Review the grouped images and keep only clearly different variations."
        )

    st.markdown(
        f"""
        <div class="action-box">
            <strong>Recommended next step:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_near_duplicates(near_duplicates: List[Dict]) -> None:
    st.subheader("Near Duplicates Above 95%")

    if not near_duplicates:
        render_empty_state(
            "OK",
            "No near duplicates found",
            "No image pairs crossed the 95% near-duplicate threshold.",
        )
        return

    st.warning("These pairs are the highest-risk results and should be reviewed first.")
    for pair in near_duplicates[:6]:
        st.markdown(
            f"""
            <div class="sg-card">
                <div class="sg-card-title">{pair["similarity_percent"]:.2f}% near duplicate</div>
                <div class="sg-muted">{pair["image_1"]} vs {pair["image_2"]}</div>
                <p>{pair["recommended_action"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View detailed near duplicate table"):
        display_rows = [
            {
                "Image 1": pair["image_1"],
                "Image 2": pair["image_2"],
                "Similarity": f"{pair['similarity_percent']:.2f}%",
                "Risk": pair["risk_level"],
                "Recommended action": pair["recommended_action"],
            }
            for pair in near_duplicates
        ]
        render_raw_table_expander("Detailed near duplicate rows", pd.DataFrame(display_rows), caption="Detailed near duplicate table")


def render_top_risky_pairs(risky_pairs: List[Dict], images: List[UploadedImage]) -> None:
    st.subheader("Top Risky Pair Review")

    if not risky_pairs:
        render_empty_state(
            "OK",
            "No risky pairs found",
            "This batch has no pairs above the selected similarity threshold.",
        )
        return

    top_pairs = sorted(
        risky_pairs,
        key=lambda pair: pair["similarity_percent"],
        reverse=True,
    )[:5]

    for index, pair in enumerate(top_pairs, start=1):
        with st.container(border=True):
            st.markdown(
                f"**Pair {index}: {pair['similarity_percent']:.2f}%** "
                f"<span class='risk-pill'>{pair['risk_level']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(pair["recommended_action"])
            columns = st.columns(2)
            with columns[0]:
                first = images[pair["image_1_index"]]
                st.image(first.thumbnail, caption=first.name, use_container_width=True)
            with columns[1]:
                second = images[pair["image_2_index"]]
                st.image(second.thumbnail, caption=second.name, use_container_width=True)


def render_best_shot_selector(
    groups: List[List[int]],
    images: List[UploadedImage],
    quality_rows: List[Dict],
    all_pairs: List[Dict],
) -> Dict[int, str]:
    """Suggest the best image in each similar group and allow user override."""

    images = safe_sequence(images)
    quality_rows = safe_sequence(quality_rows)
    all_pairs = safe_sequence(all_pairs)
    groups = safe_sequence(groups)

    st.subheader("Best Shot Selector")

    if not groups:
        render_empty_state(
            "OK",
            "No similar groups to resolve",
            "Best-shot suggestions appear when a similar group is detected.",
        )
        return {}

    quality_by_index = {row["index"]: row for row in quality_rows}
    similarity_summary = build_similarity_summary(images, all_pairs)
    best_shot_details = build_best_shot_details(groups, quality_rows, similarity_summary)
    decisions: Dict[int, str] = {}

    for group_number, group in enumerate(groups, start=1):
        best_index = get_best_shot_for_group(group, quality_by_index, similarity_summary)
        with st.container(border=True):
            st.markdown(f"**Group {group_number}: Best shot suggestion**")
            columns = st.columns(min(len(group), 4))
            for position, image_index in enumerate(group):
                item = images[image_index]
                quality = quality_by_index[image_index]
                default_decision = "Keep" if image_index == best_index else "Remove"
                detail = best_shot_details[image_index]
                badge = detail["label"]
                badge_class = "green" if image_index == best_index else "yellow"

                with columns[position % len(columns)]:
                    st.image(item.thumbnail, caption=item.name, use_container_width=True)
                    st.markdown(
                        f"<span class='status-badge {badge_class}'>{badge}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Best Shot score {detail['best_shot_score']:.2f}/100 | "
                        f"Quality {quality['quality_score']:.2f}/100 | "
                        f"Sharpness {quality['sharpness_score']:.2f} | "
                        f"{quality['brightness_status']} | "
                        f"{quality['width']}x{quality['height']} ({quality['megapixels']}MP)"
                    )
                    st.caption(detail["best_shot_reason"])
                    decision = st.selectbox(
                        "Decision",
                        ["Keep", "Remove", "Undecided"],
                        index=["Keep", "Remove", "Undecided"].index(default_decision),
                        key=f"best_shot_decision_{group_number}_{image_index}_{item.name}",
                    )
                    decisions[image_index] = decision

    return decisions


def render_keep_remove_workflow(
    risky_pairs: List[Dict],
    images: List[UploadedImage],
    default_remove_indexes: List[int] | None = None,
    csv_export_enabled: bool = True,
) -> List[int]:
    """Let the user choose which images to remove before clean ZIP export."""

    st.subheader("Keep / Remove Workflow")

    if not risky_pairs:
        st.info("No risky pairs found, so the cleaned batch ZIP will include all uploaded images.")
        return []

    remove_indexes = set()
    for pair_number, pair in enumerate(risky_pairs, start=1):
        first_name = pair["image_1"]
        second_name = pair["image_2"]
        with st.container(border=True):
            st.markdown(
                f"**Pair {pair_number}: {pair['similarity_percent']:.2f}%** "
                f"<span class='risk-pill'>{pair['risk_level']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(pair["recommended_action"])

            preview_columns = st.columns(2)
            with preview_columns[0]:
                first_image = images[pair["image_1_index"]]
                st.image(first_image.thumbnail, caption=f"Image 1: {first_name}", use_container_width=True)
            with preview_columns[1]:
                second_image = images[pair["image_2_index"]]
                st.image(second_image.thumbnail, caption=f"Image 2: {second_name}", use_container_width=True)

            choice = st.radio(
                "Choose what to remove",
                options=[
                    "Keep both",
                    "Remove image 2",
                    "Remove image 1",
                ],
                captions=[
                    "Use when both images are good enough to keep.",
                    second_name,
                    first_name,
                ],
                key=f"remove_choice_{pair_number}_{first_name}_{second_name}",
                horizontal=True,
            )

        if choice == "Remove image 2":
            remove_indexes.add(pair["image_2_index"])
        elif choice == "Remove image 1":
            remove_indexes.add(pair["image_1_index"])

    suggested_remove_indexes = sorted(set(default_remove_indexes or []) | remove_indexes)
    remove_options = {
        f"{index + 1}. {image.name}": index for index, image in enumerate(images)
    }
    suggested_remove_labels = [
        label
        for label, index in remove_options.items()
        if index in suggested_remove_indexes
    ]

    st.markdown("### Final ZIP Contents Control")
    st.caption(
        "Only images selected in this box will be removed from the cleaned ZIP. "
        "Everything else will be included for the customer."
    )
    final_removed_labels = st.multiselect(
        "Images to remove from cleaned ZIP",
        options=list(remove_options.keys()),
        default=suggested_remove_labels,
        key=f"final_removed_images_for_cleaned_zip_{abs(hash('|'.join(suggested_remove_labels)))}",
        help="Use this final list as the source of truth for the cleaned ZIP.",
    )

    sorted_remove_indexes = sorted(remove_options[label] for label in final_removed_labels)
    remove_indexes = set(sorted_remove_indexes)
    kept_images = [
        image.name for index, image in enumerate(images) if index not in remove_indexes
    ]
    removed_images = [images[index].name for index in sorted_remove_indexes]

    st.markdown("### Keep / Remove Result")
    summary_columns = st.columns(3)
    summary_columns[0].metric("Original images", len(images))
    summary_columns[1].metric("Images to remove", len(removed_images))
    summary_columns[2].metric("Images kept", len(kept_images))

    if removed_images:
        st.warning("These image(s) are currently selected for removal from the cleaned batch ZIP.")
        render_light_html_table(
            pd.DataFrame({"remove_from_cleaned_zip": removed_images}),
            caption="Images to remove from cleaned ZIP",
        )
    else:
        st.success("No images are selected for removal. The cleaned batch ZIP will keep all uploaded images.")

    checklist_df = pd.DataFrame(
        {
            "kept_images": pd.Series(kept_images),
            "removed_images": pd.Series(removed_images),
        }
    )
    if csv_export_enabled:
        st.caption("Optional review aid: this checklist records your keep/remove selections; it is not the Clean ZIP.")
        st.download_button(
            label="Download Keep / Remove Checklist",
            data=csv_download_bytes(checklist_df),
            file_name="keep_remove_checklist.csv",
            mime="text/csv",
            key="download_keep_remove_checklist",
        )
    else:
        st.info("CSV checklist export is not enabled for your current plan.")

    return sorted_remove_indexes


def image_bytes_for_zip(image: UploadedImage) -> bytes:
    """Return export-safe bytes for one image, falling back when original bytes are missing."""

    raw_bytes = getattr(image, "bytes_data", None)
    if isinstance(raw_bytes, (bytes, bytearray, memoryview)) and len(raw_bytes) > 0:
        return bytes(raw_bytes)

    fallback_image = getattr(image, "image", None) or getattr(image, "thumbnail", None) or getattr(image, "model_image", None)
    if fallback_image is None:
        raise ValueError(f"No image bytes available for ZIP export: {getattr(image, 'name', 'unknown')}")

    image_buffer = io.BytesIO()
    try:
        fallback_image.save(image_buffer, format="PNG")
    except Exception:
        fallback_image.convert("RGB").save(image_buffer, format="PNG")

    return image_buffer.getvalue()


def build_cleaned_batch_zip(images: List[UploadedImage], remove_indexes: List[int]) -> bytes:
    buffer = io.BytesIO()
    remove_set = set(remove_indexes)
    seen_names: Dict[str, int] = {}

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for index, image in enumerate(images):
            if index not in remove_set:
                safe_name, _renamed = make_unique_filename(image.name, seen_names)
                zip_file.writestr(safe_name, image_bytes_for_zip(image))

    buffer.seek(0)
    return buffer.getvalue()


def sanitize_csv_cell(value):
    """Prevent spreadsheet formula injection when users open exported CSV files."""

    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return f"'{value}"
    return value


def normalize_text(text) -> str:
    """Normalize metadata text for local/offline comparison."""

    text = str(text or "").lower()
    text = re.sub(r"[^\w\s,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_keywords(keywords_text) -> set:
    """Split keyword/tag text by comma and return a clean unique set."""

    cleaned = normalize_text(keywords_text)
    tokens = []
    for part in cleaned.split(","):
        token = part.strip()
        if token:
            tokens.append(token)
    return set(tokens)


def calculate_title_similarity(title1, title2) -> float:
    title1 = normalize_text(title1)
    title2 = normalize_text(title2)
    if not title1 or not title2:
        return 0.0
    return round(difflib.SequenceMatcher(None, title1, title2).ratio() * 100, 2)


def calculate_keyword_similarity(keywords1, keywords2) -> float:
    first = tokenize_keywords(keywords1)
    second = tokenize_keywords(keywords2)
    if not first or not second:
        return 0.0
    return round((len(first & second) / len(first | second)) * 100, 2)


def calculate_metadata_similarity(title1, keywords1, title2, keywords2) -> Dict:
    """Return weighted metadata similarity scores from title and keywords."""

    title_similarity = calculate_title_similarity(title1, title2)
    keyword_similarity = calculate_keyword_similarity(keywords1, keywords2)
    overall = round((title_similarity * 0.4) + (keyword_similarity * 0.6), 2)
    return {
        "title_similarity_percent": title_similarity,
        "keyword_similarity_percent": keyword_similarity,
        "metadata_similarity_percent": overall,
    }


def metadata_risk_label(score: float) -> str:
    if score >= 90:
        return "High metadata similarity risk"
    if score >= 75:
        return "Review metadata"
    return "Low metadata risk"


def metadata_recommendation(title_score: float, keyword_score: float, overall_score: float) -> str:
    if overall_score >= 90 and keyword_score >= title_score:
        return "Reduce repeated keywords and use unique subject, location, style, or concept keywords."
    if overall_score >= 90:
        return "Rewrite title to be more specific and avoid copying the same metadata."
    if overall_score >= 75:
        return "Review title and keywords before upload; avoid repeated metadata across similar images."
    return "Metadata looks reasonably unique."


def build_metadata_similarity_pairs(images: List[UploadedImage], metadata_rows: Dict[str, Dict]) -> List[Dict]:
    """Compare metadata for every image pair without duplicate comparisons."""

    metadata_rows = safe_mapping(metadata_rows)
    pairs = []
    for first_index in range(len(images)):
        first = images[first_index]
        first_meta = metadata_rows.get(first.name, {})
        if not (first_meta.get("title") or first_meta.get("keywords")):
            continue
        for second_index in range(first_index + 1, len(images)):
            second = images[second_index]
            second_meta = metadata_rows.get(second.name, {})
            if not (second_meta.get("title") or second_meta.get("keywords")):
                continue
            scores = calculate_metadata_similarity(
                first_meta.get("title", ""),
                first_meta.get("keywords", ""),
                second_meta.get("title", ""),
                second_meta.get("keywords", ""),
            )
            overall = scores["metadata_similarity_percent"]
            if overall >= 75:
                pairs.append(
                    {
                        "image_1": first.name,
                        "image_2": second.name,
                        "title_1": first_meta.get("title", ""),
                        "title_2": second_meta.get("title", ""),
                        "keywords_1": first_meta.get("keywords", ""),
                        "keywords_2": second_meta.get("keywords", ""),
                        **scores,
                        "risk_level": metadata_risk_label(overall),
                        "recommendation": metadata_recommendation(
                            scores["title_similarity_percent"],
                            scores["keyword_similarity_percent"],
                            overall,
                        ),
                    }
                )
    return sorted(pairs, key=lambda row: row["metadata_similarity_percent"], reverse=True)


def build_metadata_summary(images: List[UploadedImage], metadata_rows: Dict[str, Dict], metadata_pairs: List[Dict]) -> Dict[str, Dict]:
    """Summarize strongest metadata match for each image."""

    metadata_rows = safe_mapping(metadata_rows)
    metadata_pairs = safe_sequence(metadata_pairs)

    summary = {
        image.name: {
            "title": metadata_rows.get(image.name, {}).get("title", ""),
            "keywords": metadata_rows.get(image.name, {}).get("keywords", ""),
            "metadata_similarity_percent": 0.0,
            "metadata_matched_with": "",
            "title_similarity_percent": 0.0,
            "keyword_similarity_percent": 0.0,
            "metadata_risk_level": "Low metadata risk",
            "metadata_recommendation": "Metadata was not checked or no risky repeated metadata was found.",
        }
        for image in images
    }
    for pair in metadata_pairs:
        for image_key, other_key in [("image_1", "image_2"), ("image_2", "image_1")]:
            filename = pair[image_key]
            current = summary[filename]
            if pair["metadata_similarity_percent"] > current["metadata_similarity_percent"]:
                current.update(
                    {
                        "metadata_similarity_percent": pair["metadata_similarity_percent"],
                        "metadata_matched_with": pair[other_key],
                        "title_similarity_percent": pair["title_similarity_percent"],
                        "keyword_similarity_percent": pair["keyword_similarity_percent"],
                        "metadata_risk_level": pair["risk_level"],
                        "metadata_recommendation": pair["recommendation"],
                    }
                )
    return summary


def csv_download_bytes(dataframe: pd.DataFrame) -> bytes:
    """Return a CSV export with safe cells and a subtle StockGuard footer row."""

    safe_df = dataframe.copy()
    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(sanitize_csv_cell)
    if not safe_df.empty and len(safe_df.columns) > 0:
        footer_row = {column: "" for column in safe_df.columns}
        footer_row[safe_df.columns[0]] = REPORT_FOOTER_TEXT
        safe_df = pd.concat([safe_df, pd.DataFrame([footer_row])], ignore_index=True)
    return safe_df.to_csv(index=False).encode("utf-8")


def clean_zip_summary(
    images: List[UploadedImage],
    auto_decision_df: pd.DataFrame,
    remove_indexes: List[int],
) -> Dict[str, int]:
    """Summarize exactly what the Clean ZIP will include and exclude."""

    included_indexes = set(range(len(images))) - set(remove_indexes)
    if auto_decision_df.empty:
        return {
            "ready_included": len(included_indexes),
            "best_shot_included": 0,
            "manual_keep_included": 0,
            "review_excluded": 0,
            "remove_excluded": len(remove_indexes),
            "total_in_zip": len(included_indexes),
        }

    included_df = auto_decision_df[auto_decision_df["image_index"].isin(included_indexes)]
    excluded_df = auto_decision_df[~auto_decision_df["image_index"].isin(included_indexes)]
    return {
        "ready_included": int((included_df["final_status"] == "Ready to Upload").sum()),
        "best_shot_included": int((included_df["best_shot_recommendation"] == "Recommended Keep").sum()),
        "manual_keep_included": int((included_df["user_decision"] == "Keep").sum()),
        "review_excluded": int((excluded_df["final_status"] == "Review Needed").sum()),
        "remove_excluded": int((excluded_df["final_status"] == "Remove Recommended").sum()),
        "total_in_zip": len(included_indexes),
    }


def render_recommended_next_step(
    remove_count: int,
    review_count: int,
    zip_enabled: bool,
) -> None:
    """Show the contributor what to do immediately after scan completion."""

    if remove_count > 0:
        message = "Review remove recommendations and export the Clean ZIP."
    elif review_count > 0:
        message = "Check Review Needed images, mark any final Keep images, then export the Clean ZIP."
    else:
        message = "Your batch looks clean. Download the Clean ZIP and upload to your stock platform."
    if not zip_enabled:
        message = f"{message} Upgrade to export the Clean ZIP batch."

    st.markdown(
        f"""
        <div class="action-box">
            <strong>Recommended next step:</strong> {escape_html(message)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_export_actions(
    user: Dict,
    images: List[UploadedImage],
    final_zip_remove_indexes: List[int],
    report_csv: pd.DataFrame,
    batch_name: str,
) -> None:
    """Render the primary result actions and retain transient exports for Downloads."""

    plan = get_current_plan(user)
    kept_count = max(0, len(images) - len(set(final_zip_remove_indexes)))
    clean_zip_bytes = None
    if plan["zip_export"] and kept_count > 0:
        clean_zip_bytes = build_cleaned_batch_zip(images, final_zip_remove_indexes)

    report_csv_bytes = None
    if plan["csv_export"] and not report_csv.empty:
        report_csv_bytes = csv_download_bytes(report_csv.drop(columns=["image_index"], errors="ignore"))

    st.session_state["latest_scan_exports"] = {
        "batch_name": batch_name,
        "clean_zip": clean_zip_bytes,
        "csv_report": report_csv_bytes,
    }

    with st.container(border=True, key="result_export_actions"):
        st.markdown(
            """
            <div class="sg-export-actions-heading">
                <h3>Export your cleaned batch</h3>
                <p>Download your kept images as a clean ZIP, or save the scan report as CSV.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        zip_column, csv_column, new_scan_column = st.columns([1.15, 1, 0.9], gap="medium")
        with zip_column:
            st.download_button(
                "Download Clean ZIP",
                data=clean_zip_bytes or b"",
                file_name="clean_stockguard_client_batch.zip",
                mime="application/zip",
                type="primary",
                icon=":material/folder_zip:",
                disabled=clean_zip_bytes is None,
                key="download_clean_zip_primary",
                use_container_width=True,
            )
            st.caption("Includes only images currently marked Keep.")
        with csv_column:
            st.download_button(
                "Download CSV Report",
                data=report_csv_bytes or b"",
                file_name="upload_readiness_report.csv",
                mime="text/csv",
                icon=":material/download:",
                disabled=report_csv_bytes is None,
                key="download_csv_report_primary",
                use_container_width=True,
            )
            st.caption("Save image-level decisions and review details.")
        with new_scan_column:
            if st.button(
                "Start New Scan",
                key="start_new_scan_primary_actions",
                icon=":material/add_photo_alternate:",
                use_container_width=True,
            ):
                reset_scan_session_state()
                st.rerun()
            st.caption("Clear this result and upload a new batch.")

        if kept_count == 0:
            st.warning("No kept images are available for the clean ZIP. Review your keep/remove selections first.")
        elif not plan["zip_export"]:
            st.info("Clean ZIP export is available on Starter and above.")
        if not plan["csv_export"]:
            st.info("CSV export is not enabled for your current plan.")

        st.markdown("---")
        st.caption(
            "Your uploaded images are processed in memory only and are cleared "
            "when you start a new scan, remove all, log out, or clear scan data."
        )
        if st.button(
            "Clear Scan Data",
            key="clear_scan_data",
            icon=":material/delete:",
            use_container_width=True,
        ):
            cleanup_temporary_scan_files()
            st.rerun()


def render_final_client_export(
    user: Dict,
    images: List[UploadedImage],
    auto_decision_df: pd.DataFrame,
    final_zip_remove_indexes: List[int],
    similarity_csv: pd.DataFrame,
) -> None:
    """Render the paid Clean ZIP conversion area and clear export cards."""

    plan = get_current_plan(user)
    summary = clean_zip_summary(images, auto_decision_df, final_zip_remove_indexes)
    review_count = int((auto_decision_df["final_status"] == "Review Needed").sum()) if not auto_decision_df.empty else 0
    remove_count = int((auto_decision_df["final_status"] == "Remove Recommended").sum()) if not auto_decision_df.empty else len(final_zip_remove_indexes)

    st.subheader("Final Client Export")
    st.caption("Download a Clean ZIP containing only images you have marked as ready to upload.")
    render_recommended_next_step(remove_count, review_count, bool(plan["zip_export"]))

    with st.container(border=True):
        render_export_card(
            "Clean Batch ZIP",
            (
                "Includes Ready to Upload images, Recommended Keep images, and manually marked Keep images. "
                "Excludes Remove Recommended images and manually removed files."
            ),
            "Client ZIP",
        )
        summary_columns = st.columns(3)
        summary_columns[0].metric("Ready images included", summary["ready_included"])
        summary_columns[1].metric("Best Shot / Keep included", summary["best_shot_included"])
        summary_columns[2].metric("Manual Keep included", summary["manual_keep_included"])
        summary_columns = st.columns(3)
        summary_columns[0].metric("Review excluded", summary["review_excluded"])
        summary_columns[1].metric("Remove excluded", summary["remove_excluded"])
        summary_columns[2].metric("Total images in ZIP", summary["total_in_zip"])

        if summary["review_excluded"] > 0:
            st.warning(
                "Some images still need review. They are not included in the Clean ZIP unless you mark them as Keep."
            )

        if plan["zip_export"]:
            st.caption("Use the primary Download Clean ZIP action above to export this reviewed batch.")
        else:
            st.info("Clean ZIP export is available on Starter and above.")
            if st.button("Upgrade to unlock Clean ZIP export", key="unlock_clean_zip_export", use_container_width=True):
                st.session_state["page"] = "Subscription"
                st.rerun()

    st.subheader("Additional Detail Export")
    with st.container(border=True):
        render_export_card("Pair Similarity CSV", "Download detailed image-to-image similarity scores.", "CSV")
        if plan["csv_export"]:
            st.download_button(
                "Download Similarity CSV",
                data=csv_download_bytes(similarity_csv),
                file_name="pair_similarity_report.csv",
                mime="text/csv",
                disabled=similarity_csv.empty,
                key="download_similarity_csv_export_results",
                use_container_width=True,
            )
        else:
            st.info("CSV export is not enabled for your current plan.")


def render_upload_readiness_report(
    user: Dict,
    project_name: str,
    batch_name: str,
    images: List[UploadedImage],
    quality_rows: List[Dict],
    all_pairs: List[Dict],
    groups: List[List[int]],
    near_duplicates: List[Dict],
    highest_similarity: float,
    remove_indexes: List[int],
    best_shot_decisions: Dict[int, str],
) -> pd.DataFrame:
    """Render paid upload-readiness report and return the CSV-ready dataframe."""

    plan = get_current_plan(user)
    if not plan["readiness_report"]:
        render_info_card(
            "Upload Readiness Report",
            (
                "Starter, Pro, and Agency users get per-image quality scores, best-shot "
                "recommendations, readiness status, and a professional upload CSV."
            ),
            "Paid feature preview",
        )
        starter_plan = get_subscription_plan("starter")
        upgrade_label = starter_plan["plan_name"] if starter_plan else "Starter"
        if st.button(f"Upgrade to {upgrade_label}", key="upgrade_readiness_starter"):
            update_user_plan(user["id"], "starter")
            st.success(f"Plan changed to {upgrade_label}.")
            st.rerun()
        return pd.DataFrame()

    rows = build_upload_readiness_rows(
        project_name=project_name,
        batch_name=batch_name,
        images=images,
        quality_rows=quality_rows,
        all_pairs=all_pairs,
        groups=groups,
        remove_indexes=remove_indexes,
        best_shot_decisions=best_shot_decisions,
    )
    report_df = pd.DataFrame(rows)

    ready_count = int((report_df["upload_readiness_status"] == "Ready to Upload").sum())
    review_count = int((report_df["upload_readiness_status"] == "Review Needed").sum())
    remove_count = int((report_df["upload_readiness_status"] == "Remove Recommended").sum())
    average_quality = float(report_df["quality_score"].mean()) if not report_df.empty else 0.0

    render_page_header(
        "Upload Readiness Report",
        "Know what to upload, what to review, and what to remove.",
        "Paid quality report",
    )
    st.markdown(
        """
        <div class="sg-card" style="padding: 0.9rem 1rem; margin-bottom: 0.9rem; border-radius: 18px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
            <div class="sg-card-title" style="margin-bottom: 0.15rem;">Readiness overview</div>
            <div class="sg-muted">Use this summary to confirm the quality and similarity status of each image before submission.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    report_action_columns = st.columns([4, 1])
    with report_action_columns[0]:
        st.caption("Use this report to decide what should be uploaded, reviewed, or removed before submission.")
    with report_action_columns[1]:
        if st.button("View Full Report", key=f"view_full_report_{batch_name}"):
            st.session_state["page"] = "Readiness Report"
            st.rerun()
    status_columns = st.columns(3)
    with status_columns[0]:
        render_status_card(
            "OK",
            "Ready to Upload",
            f"{ready_count} images",
            "High quality, unique, and safer to submit.",
            "green",
        )
    with status_columns[1]:
        render_status_card(
            "Warn",
            "Review Needed",
            f"{review_count} images",
            "Check similar groups and quality before deciding.",
            "orange",
        )
    with status_columns[2]:
        render_status_card(
            "Del",
            "Remove Recommended",
            f"{remove_count} images",
            "Low quality or duplicates. Consider removing.",
            "red",
        )
    render_metric_cards(
        [
            ("Img", "Total images", str(len(images)), "Images accepted for this scan."),
            ("Ready", "Ready to upload", str(ready_count), "Cleanest candidates."),
            ("Review", "Review needed", str(review_count), "Needs human decision."),
            ("Remove", "Remove recommended", str(remove_count), "High risk or low quality."),
            ("Score", "Average quality", f"{average_quality:.2f}", "Average score out of 100."),
        ]
    )
    render_metric_cards(
        [
            ("Near", "Near duplicates", str(len(near_duplicates)), "Pairs at 95% or higher."),
            ("Top", "Highest similarity", f"{highest_similarity:.2f}%", "Strongest pair similarity."),
        ]
    )

    status_classes = {
        "Ready to Upload": "ready",
        "Review Needed": "review",
        "Remove Recommended": "remove",
    }
    badge_classes = {
        "Ready to Upload": "green",
        "Review Needed": "yellow",
        "Remove Recommended": "red",
    }

    images_by_name = {image.name: image for image in images}
    for section_status in ["Ready to Upload", "Review Needed", "Remove Recommended"]:
        section_rows = [row for row in rows if row["upload_readiness_status"] == section_status]
        st.markdown(f"### {section_status}")
        if not section_rows:
            render_empty_state("None", f"No {section_status.lower()} images", "Nothing to show in this status.")
            continue

        for row in section_rows:
            warnings = next(
                (quality["warnings"] for quality in quality_rows if quality["filename"] == row["filename"]),
                [],
            )
            warning_badges = ""
            if row["highest_similarity_percent"] >= 95:
                warning_badges += "<span class='status-badge red'>Near Duplicate</span>"
            if row["quality_score"] < 55:
                warning_badges += "<span class='status-badge yellow'>Low Quality</span>"
            if "Blurry" in warnings:
                warning_badges += "<span class='status-badge red'>Blurry</span>"

            status = row["upload_readiness_status"]
            with st.container(border=True):
                columns = st.columns([1, 2])
                with columns[0]:
                    if row["filename"] in images_by_name:
                        st.image(images_by_name[row["filename"]].thumbnail, use_container_width=True)
                with columns[1]:
                    best_score_text = (
                        f"{row['best_shot_score']:.2f}/100"
                        if isinstance(row["best_shot_score"], (int, float))
                        else "N/A"
                    )
                    st.markdown(
                        f"""
                        <div class="readiness-card {status_classes[status]}">
                            <div class="sg-card-title">{row["filename"]}</div>
                            <span class="status-badge {badge_classes[status]}">{status}</span>
                            {warning_badges}
                            <p class="sg-muted">
                                Quality {row["quality_score"]:.2f}/100 |
                                Sharpness {row["sharpness_score"]:.2f} |
                                {row["brightness_status"]} |
                                Highest similarity {row["highest_similarity_percent"]:.2f}%
                            </p>
                            <p><strong>Reason:</strong> {row["reason"]}</p>
                            <p>{row["recommended_action"]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Best Shot: {best_score_text} - {row['best_shot_reason']}")

    if plan["csv_export"]:
        st.download_button(
            "Download Upload Readiness CSV",
            data=csv_download_bytes(report_df),
            file_name="upload_readiness_report.csv",
            mime="text/csv",
        )
    else:
        st.info("CSV export is not enabled for your current plan.")

    render_raw_table_expander("View raw upload readiness data", report_df, caption="Upload readiness raw table")

    return report_df


def render_auto_decision_report(
    user: Dict,
    project_name: str,
    batch_name: str,
    images: List[UploadedImage],
    quality_rows: List[Dict],
    all_pairs: List[Dict],
    groups: List[List[int]],
    near_duplicates: List[Dict],
    remove_indexes: List[int],
    best_shot_decisions: Dict[int, str],
) -> pd.DataFrame:
    """Render automatic large-batch classification and return report rows."""

    rows = build_auto_decision_rows(
        project_name=project_name,
        batch_name=batch_name,
        images=images,
        quality_rows=quality_rows,
        all_pairs=all_pairs,
        groups=groups,
        remove_indexes=remove_indexes,
        best_shot_decisions=best_shot_decisions,
        near_duplicates=near_duplicates,
    )
    report_df = pd.DataFrame(rows)
    ready_count = int((report_df["final_status"] == "Ready to Upload").sum()) if not report_df.empty else 0
    review_count = int((report_df["final_status"] == "Review Needed").sum()) if not report_df.empty else 0
    remove_count = int((report_df["final_status"] == "Remove Recommended").sum()) if not report_df.empty else 0
    zip_count = int(report_df["included_in_auto_clean_zip"].sum()) if not report_df.empty else 0

    render_page_header(
        "Auto Scan Summary",
        "StockGuard AI has automatically sorted your batch. Review only the images that need attention.",
        "Auto Scan Mode",
    )
    st.markdown(
        """
        <div class="sg-card" style="padding: 0.9rem 1rem; margin-bottom: 0.9rem; border-radius: 18px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
            <div class="sg-card-title" style="margin-bottom: 0.15rem;">Auto-sorted review view</div>
            <div class="sg-muted">This summary keeps the same scan logic while making the batch status easier to scan at a glance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_cards(
        [
            ("Img", "Total uploaded images", str(len(images)), "Valid images in this scan."),
            ("OK", "Ready to Upload", str(ready_count), "Good quality and low similarity risk."),
            ("Review", "Review Needed", str(review_count), "Check these before exporting your final batch."),
            ("Remove", "Remove Recommended", str(remove_count), "Near duplicates, weak variations, or low quality images."),
            ("ZIP", "Suggested Clean ZIP", str(zip_count), "Images included in Auto Clean ZIP."),
        ]
    )
    render_metric_cards(
        [
            ("Near", "Near Duplicate count", str(len(near_duplicates)), "Pairs at 95% or higher."),
        ]
    )

    images_by_name = {image.name: image for image in images}
    section_copy = {
        "Ready to Upload": "Good quality and low similarity risk.",
        "Review Needed": "Check these before exporting your final batch.",
        "Remove Recommended": "Near duplicates, weak variations, or low quality images.",
    }
    for status in ["Ready to Upload", "Review Needed", "Remove Recommended"]:
        section_rows = [row for row in rows if row["final_status"] == status]
        expanded = status != "Ready to Upload"
        with st.expander(f"{status} ({len(section_rows)}) - {section_copy[status]}", expanded=expanded):
            if not section_rows:
                st.info(f"No images in {status}.")
                continue
            display_rows = []
            for row in section_rows:
                display_rows.append(
                    {
                        "filename": row["filename"],
                        "auto_status": row["auto_status"],
                        "final_status": row["final_status"],
                        "reason": row["auto_reason"],
                        "similarity": f"{row['highest_similarity_percent']:.2f}%",
                        "quality": f"{row['quality_score']:.2f}",
                        "matched_with": row["matched_with"],
                        "zip": "Included" if row["included_in_auto_clean_zip"] else "Excluded",
                    }
                )
            render_light_html_table(pd.DataFrame(display_rows), caption="Auto decision detail rows")

            if status != "Ready to Upload" or len(section_rows) <= 24:
                preview_columns = st.columns(4)
                for preview_index, row in enumerate(section_rows[:24]):
                    with preview_columns[preview_index % 4]:
                        image = images_by_name.get(row["filename"])
                        if image:
                            st.image(image.thumbnail, caption=row["filename"], use_container_width=True)
                        st.caption(row["auto_reason"])

    if get_current_plan(user)["csv_export"]:
        st.download_button(
            "Download Auto Decision Report",
            data=csv_download_bytes(report_df.drop(columns=["image_index"], errors="ignore")),
            file_name="auto_decision_report.csv",
            mime="text/csv",
        )

    return report_df


def render_groups(
    groups: List[List[int]],
    risky_pairs: List[Dict],
    images: List[UploadedImage],
    quality_rows: List[Dict] | None = None,
    all_pairs: List[Dict] | None = None,
) -> None:
    st.subheader("Similar Groups")

    if not groups:
        render_empty_state(
            "Group",
            "No similar groups",
            "Connected similarity groups will appear here after risky pairs are detected.",
        )
        return

    sorted_groups = sorted(
        groups,
        key=lambda group: group_highest_similarity(group, risky_pairs),
        reverse=True,
    )
    quality_by_index = {row["index"]: row for row in (quality_rows or [])}
    similarity_summary = build_similarity_summary(images, all_pairs or [])
    best_details = build_best_shot_details(sorted_groups, quality_rows or [], similarity_summary) if quality_by_index else {}

    for group_number, group in enumerate(sorted_groups, start=1):
        highest_in_group = group_highest_similarity(group, risky_pairs)
        group_risk = risk_label(highest_in_group)
        best_index = None
        if quality_by_index:
            best_index = get_best_shot_for_group(group, quality_by_index, similarity_summary)

        with st.expander(
            f"Group {group_number}: {len(group)} images | "
            f"Highest similarity {highest_in_group:.2f}% | {group_risk}",
            expanded=True,
        ):
            render_similar_group_card(
                group_number,
                len(group),
                "Best sharpness and overall quality in this group.",
            )
            st.write(f"**Risk label:** {group_risk}")
            st.write(f"**Number of images:** {len(group)}")
            st.write(f"**Highest similarity in group:** {highest_in_group:.2f}%")
            if best_index is not None:
                st.write(f"**Best shot recommendation:** {images[best_index].name}")
                st.caption(best_details[best_index]["best_shot_reason"])
            st.caption(recommended_action(highest_in_group))

            columns_per_row = 4
            display_group = group
            if best_index is not None:
                display_group = [best_index] + [index for index in group if index != best_index]
            for start in range(0, len(display_group), columns_per_row):
                columns = st.columns(columns_per_row)
                for column, image_index in zip(columns, display_group[start : start + columns_per_row]):
                    with column:
                        item = images[image_index]
                        st.image(item.thumbnail, caption=item.name, use_container_width=True)
                        if image_index in quality_by_index:
                            quality = quality_by_index[image_index]
                            detail = best_details.get(image_index, {})
                            label = detail.get("label", "Review Needed")
                            render_image_review_card(
                                filename=item.name,
                                quality_score=quality["quality_score"],
                                similarity_percent=similarity_summary[image_index]["highest_similarity_percent"],
                                status=label,
                                recommended=label == "Recommended Keep",
                            )
                            st.caption(
                                f"Quality {quality['quality_score']:.2f}/100 | "
                                f"Best Shot {detail.get('best_shot_score', 0):.2f}/100 | "
                                f"Sharpness {quality['sharpness_score']:.2f} | "
                                f"{quality['brightness_status']} | "
                                f"{quality['width']}x{quality['height']}"
                            )
                            if detail:
                                st.caption(detail["best_shot_reason"])


def render_public_landing_page() -> None:
    """Public marketing/landing section shown before login."""

    render_page_header(
        PRODUCT_NAME,
        "Protect your stock contributor account before upload.",
        "Pre-upload Similarity Scanner",
    )
    cta_columns = st.columns([1, 1, 3])
    with cta_columns[0]:
        if st.button("Start Free Scan", key="landing_start_free_scan", use_container_width=True):
            st.info("Use the Sign Up tab below to create a Free account and start your first scan.")
    with cta_columns[1]:
        if st.button("View Pricing", key="landing_view_pricing", use_container_width=True):
            st.info("Pricing preview is shown below. Log in and open Subscription to change plans.")

    info_columns = st.columns(2)
    with info_columns[0]:
        render_info_card(
            "Problem",
            "Similar content rejection wastes time and can risk contributor accounts.",
            "Why it matters",
        )
    with info_columns[1]:
        render_info_card(
            "Solution",
            "Scan your batch, detect duplicates, pick best shots, and export a clean ZIP.",
            "How it helps",
        )

    st.subheader("Built for stock contributor workflows")
    feature_columns = st.columns(3)
    features = [
        ("Similarity Checker", "Find duplicate and near-duplicate images before upload."),
        ("Upload Readiness Report", "See what is ready, what needs review, and what should be removed."),
        ("Best Shot Selector", "Automatically suggest the strongest image in each similar group."),
        ("Clean ZIP Export", "Export a reviewed batch after keep/remove decisions."),
        ("Project Folders", "Organize scans by client, shoot, collection, or upload batch."),
        ("CSV Reports", "Download detailed reports for your own records."),
    ]
    for index, (title, text) in enumerate(features):
        with feature_columns[index % 3]:
            render_feature_card("Feature", title, text)

    st.subheader("Pricing preview")
    active_plans = get_active_subscription_plans()
    price_columns = st.columns(max(len(active_plans), 1))
    for column, plan in zip(price_columns, active_plans):
        with column:
            price_label = plan_price_label(plan)
            render_info_card(
                plan["plan_name"],
                f"{price_label} - {plan['monthly_scans']} scans/month - {plan['images_per_scan']} images/scan",
                "Plan",
            )

    st.markdown(
        """
        <div class="hero-note">
            <strong>Disclaimer:</strong> StockGuard AI is not affiliated with Adobe.
            This tool estimates similar-content risk and does not guarantee stock upload approval.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_info_card(
        "Beta feedback",
        f"Found an issue? Send feedback to {FEEDBACK_EMAIL}.",
        "Feedback",
    )
    st.caption(LANDING_FOOTER_TEXT)
    st.caption("StockGuard AI is an independent tool and is not affiliated with Adobe.")


def get_logged_in_user() -> Dict:
    return get_current_user()


def render_auth_page() -> None:
    outer_left, auth_column, outer_right = st.columns([1, 1.15, 1])
    with auth_column:
        st.markdown(
            """
            <style>
            .sg-auth-page-shell {
                min-height: auto !important;
                display: block !important;
                padding: 0 !important;
                overflow: visible !important;
            }
            .sg-auth-logo {
                display: block !important;
                margin: 0 auto 14px !important;
                width: 64px !important;
                height: 64px !important;
                object-fit: contain !important;
            }
            [data-testid="stMainBlockContainer"] {
                min-height: 100vh !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                box-sizing: border-box !important;
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
            }
            .st-key-sg_auth_card {
                width: 100% !important;
                max-width: 580px !important;
                margin: 0 auto !important;
                max-height: calc(100vh - 32px) !important;
                padding: 26px 34px 40px !important;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 255, 0.96)) !important;
                border: 1px solid rgba(59, 130, 246, 0.30) !important;
                border-radius: 30px !important;
                box-shadow: 0 26px 64px rgba(37, 99, 235, 0.14), 0 12px 30px rgba(124, 58, 237, 0.08) !important;
                backdrop-filter: blur(12px) !important;
                overflow-x: hidden !important;
                overflow-y: auto !important;
                animation: sgFadeSlideUp 320ms cubic-bezier(0.22, 1, 0.36, 1) both !important;
            }
            .sg-auth-card-inner {
                text-align: center;
                max-width: 560px;
                margin: 0 auto 10px auto;
                padding: 0 !important;
            }
            .sg-auth-card-inner .sg-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 5px 13px;
                border-radius: 999px;
                border: 1px solid #BFDBFE;
                background: #EFF6FF;
                color: #2563EB;
                font-size: 0.82rem;
                font-weight: 700;
                margin-bottom: 9px;
            }
            .sg-auth-card-inner h1 {
                margin: 0 0 6px 0;
                font-size: clamp(1.9rem, 3.6vw, 2.25rem);
                font-weight: 800;
                line-height: 1.06;
                letter-spacing: -0.03em;
                color: #0F172A;
            }
            .sg-auth-card-inner p {
                margin: 0 auto;
                max-width: 520px;
                color: #475569;
                font-size: 0.98rem;
                line-height: 1.4;
            }
            .sg-auth-card-inner .sg-muted {
                color: #64748B !important;
                font-size: 0.82rem;
                line-height: 1.45;
                margin-top: 0;
            }
            [data-testid="stTabs"] {
                width: 100%;
                max-width: 560px;
                margin: 0 auto 8px auto;
            }
            [data-testid="stTabs"] [role="tablist"] {
                gap: 24px !important;
                border-bottom: none !important;
                box-shadow: none !important;
            }
            button[data-baseweb="tab"] {
                color: #475569 !important;
                font-weight: 600 !important;
                padding: 0 0 7px 0 !important;
                border-bottom: none !important;
                box-shadow: none !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #2563EB !important;
                font-weight: 700 !important;
            }
            button[data-baseweb="tab"]::after {
                content: "" !important;
                position: absolute !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                height: 2px !important;
                border-radius: 999px !important;
                background: transparent !important;
            }
            button[data-baseweb="tab"][aria-selected="true"]::after {
                background: #2563EB !important;
            }
            [data-testid="stTabs"] [role="tab"] {
                position: relative !important;
                color: #475569 !important;
                font-weight: 600 !important;
                border-bottom: none !important;
                padding-bottom: 7px !important;
            }
            [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
                color: #2563EB !important;
                font-weight: 700 !important;
                border-bottom: none !important;
                border-bottom-color: transparent !important;
            }
            div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
                display: none !important;
                height: 0 !important;
                background-color: transparent !important;
            }
            div[data-testid="stForm"] {
                background: transparent !important;
                border: none !important;
                border-radius: 18px !important;
                padding: 0 !important;
                box-shadow: none !important;
            }
            div[data-testid="stForm"] > div[data-testid="stTextInput"],
            div[data-testid="stForm"] > div[data-testid="stPasswordInput"] {
                margin-bottom: 8px !important;
            }
            div[data-testid="stTextInput"] label,
            div[data-testid="stPasswordInput"] label,
            div[data-testid="stTextInput"] [data-testid="stWidgetLabel"],
            div[data-testid="stPasswordInput"] [data-testid="stWidgetLabel"] {
                color: #334155 !important;
                font-size: 0.92rem !important;
                font-weight: 600 !important;
                margin-bottom: 3px !important;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"],
            div[data-testid="stPasswordInput"] div[data-baseweb="input"] {
                background-color: #FFFFFF !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 14px !important;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04) !important;
                min-height: 46px !important;
                padding: 0 !important;
                transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease !important;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
            div[data-testid="stPasswordInput"] div[data-baseweb="input"] > div {
                background-color: #FFFFFF !important;
                border-radius: 12px !important;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
            div[data-testid="stPasswordInput"] div[data-baseweb="input"]:hover {
                background-color: #FFFFFF !important;
                border-color: #94A3B8 !important;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
            div[data-testid="stPasswordInput"] div[data-baseweb="input"]:focus-within,
            div[data-testid="stTextInput"] div[data-baseweb="input"][data-focused="true"],
            div[data-testid="stPasswordInput"] div[data-baseweb="input"][data-focused="true"] {
                background-color: #FFFFFF !important;
                border: 2px solid #2563EB !important;
                box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12), 0 14px 24px rgba(15, 23, 42, 0.08) !important;
                transform: translateY(-1px) !important;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"]:has(input:not(:placeholder-shown)),
            div[data-testid="stPasswordInput"] div[data-baseweb="input"]:has(input:not(:placeholder-shown)) {
                border-color: #60A5FA !important;
                box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.14), 0 10px 18px rgba(15, 23, 42, 0.06) !important;
            }
            div[data-testid="stTextInput"] div[data-baseweb="input"] input,
            div[data-testid="stPasswordInput"] div[data-baseweb="input"] input {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                -webkit-text-fill-color: #0F172A !important;
                caret-color: #2563EB !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
                min-height: 46px !important;
                font-size: 0.95rem !important;
                padding: 0 0.9rem !important;
            }
            div[data-testid="stTextInput"] input,
            div[data-testid="stPasswordInput"] input {
                background: #FFFFFF !important;
                color: #0F172A !important;
                border: none !important;
                border-radius: 12px !important;
                min-height: 46px !important;
                box-shadow: none !important;
                outline: none !important;
                font-size: 0.95rem !important;
                padding: 0 0.85rem !important;
            }
            div[data-testid="stTextInput"] input:-webkit-autofill,
            div[data-testid="stTextInput"] input:-webkit-autofill:hover,
            div[data-testid="stTextInput"] input:-webkit-autofill:focus,
            div[data-testid="stTextInput"] input:-webkit-autofill:active,
            div[data-testid="stPasswordInput"] input:-webkit-autofill,
            div[data-testid="stPasswordInput"] input:-webkit-autofill:hover,
            div[data-testid="stPasswordInput"] input:-webkit-autofill:focus,
            div[data-testid="stPasswordInput"] input:-webkit-autofill:active {
                -webkit-box-shadow: 0 0 0 1000px #FFFFFF inset !important;
                box-shadow: 0 0 0 1000px #FFFFFF inset !important;
                -webkit-text-fill-color: #0F172A !important;
                caret-color: #2563EB !important;
                transition: background-color 9999s ease-in-out 0s !important;
            }
            div[data-testid="stTextInput"] input::placeholder,
            div[data-testid="stPasswordInput"] input::placeholder {
                color: #94A3B8 !important;
                opacity: 1 !important;
            }
            div[data-testid="stTextInput"] input:focus,
            div[data-testid="stPasswordInput"] input:focus,
            div[data-testid="stTextInput"] input:focus-visible,
            div[data-testid="stPasswordInput"] input:focus-visible {
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
            .st-key-sg_auth_card div[data-testid="stTooltipContent"],
            .st-key-sg_auth_card [data-baseweb="tooltip"],
            .st-key-sg_auth_card div[role="tooltip"] {
                display: none !important;
                visibility: hidden !important;
            }
            .sg-auth-note {
                margin-top: 7px;
                text-align: center;
                color: #64748B !important;
                font-size: 0.82rem !important;
                line-height: 1.45;
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }
            .sg-auth-footer {
                max-width: 560px;
                margin: 12px auto 0 auto;
                padding-top: 10px;
                border-top: 1px solid rgba(148, 163, 184, 0.18);
                text-align: center;
                color: #64748B;
                font-size: 0.82rem;
                line-height: 1.35;
            }
            header[data-testid="stHeader"] {
                display: none !important;
            }
            [data-testid="stToolbar"] {
                display: none !important;
            }
            #MainMenu {
                visibility: hidden !important;
            }
            @media (prefers-reduced-motion: reduce) {
                .st-key-sg_auth_card,
                .st-key-sg_auth_card *,
                .st-key-sg_auth_card *::before,
                .st-key-sg_auth_card *::after {
                    animation: none !important;
                    transition: none !important;
                    scroll-behavior: auto !important;
                }
            }
            @media (max-width: 768px) {
                [data-testid="stMainBlockContainer"] {
                    min-height: 100dvh !important;
                    justify-content: flex-start !important;
                    padding: 0.75rem !important;
                }
                [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(.st-key-sg_auth_card) {
                    flex: 1 1 100% !important;
                    width: 100% !important;
                    min-width: 100% !important;
                }
                .st-key-sg_auth_card {
                    width: 100% !important;
                    max-width: 580px !important;
                    max-height: none !important;
                    margin: 0 auto !important;
                    padding: 22px 24px 36px !important;
                    border-radius: 24px !important;
                    overflow: visible !important;
                    box-shadow: 0 18px 42px rgba(37, 99, 235, 0.12), 0 8px 20px rgba(124, 58, 237, 0.06) !important;
                }
                .sg-auth-card-inner {
                    max-width: 100% !important;
                }
                .sg-auth-card-inner h1 {
                    font-size: clamp(1.75rem, 7vw, 2.05rem) !important;
                }
                .sg-auth-card-inner p {
                    font-size: 0.92rem !important;
                }
                .st-key-sg_auth_card [data-testid="stTabs"] [role="tablist"] {
                    overflow-x: visible !important;
                    justify-content: center !important;
                }
                .st-key-sg_auth_card input,
                .st-key-sg_auth_card button {
                    max-width: 100% !important;
                }
                header[data-testid="stHeader"] {
                    display: flex !important;
                    height: 3rem !important;
                    min-height: 3rem !important;
                    background: transparent !important;
                }
                [data-testid="collapsedControl"],
                button[aria-label="Open sidebar"],
                button[title="Open sidebar"] {
                    display: inline-flex !important;
                    visibility: visible !important;
                }
            }
            @media (max-width: 480px) {
                [data-testid="stMainBlockContainer"] {
                    padding: 0.55rem !important;
                }
                .st-key-sg_auth_card {
                    padding: 18px 16px 32px !important;
                    border-radius: 20px !important;
                }
                .sg-auth-logo {
                    width: 54px !important;
                    height: 54px !important;
                    margin-bottom: 10px !important;
                }
                .sg-auth-card-inner .sg-badge {
                    padding: 4px 10px !important;
                    font-size: 0.75rem !important;
                }
                .sg-auth-card-inner h1 {
                    font-size: 1.72rem !important;
                }
                .sg-auth-card-inner p,
                .sg-auth-footer {
                    font-size: 0.79rem !important;
                }
                .st-key-sg_auth_card [data-testid="stTabs"] [role="tablist"] {
                    gap: 18px !important;
                }
                .st-key-sg_auth_card [data-testid="stTabs"] [role="tab"] {
                    min-width: auto !important;
                    padding-left: 0.15rem !important;
                    padding-right: 0.15rem !important;
                }
            }
            @media (max-height: 820px) {
                [data-testid="stMainBlockContainer"] {
                    min-height: auto !important;
                    justify-content: flex-start !important;
                    padding-top: 0.75rem !important;
                    padding-bottom: 0.75rem !important;
                }
                .sg-auth-page-shell {
                    justify-content: flex-start !important;
                    padding-top: 0 !important;
                    padding-bottom: 0 !important;
                }
                .sg-auth-card-inner h1 {
                    font-size: 1.9rem !important;
                    margin-bottom: 4px !important;
                }
                .sg-auth-card-inner p {
                    font-size: 0.92rem !important;
                    line-height: 1.3 !important;
                }
                .st-key-sg_auth_card {
                    padding: 20px 28px 32px !important;
                }
                .sg-auth-footer {
                    margin-top: 8px !important;
                    padding-top: 8px !important;
                }
            }
            [data-testid="stFormSubmitButton"] button,
            [data-testid="stFormSubmitButton"] > button,
            button[kind="primary"],
            button[kind="primary"]:hover,
            button[kind="primary"]:focus,
            button[kind="primary"]:active,
            [data-testid="stFormSubmitButton"] button:hover,
            [data-testid="stFormSubmitButton"] button:focus,
            [data-testid="stFormSubmitButton"] button:active {
                background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
                color: #FFFFFF !important;
                -webkit-text-fill-color: #FFFFFF !important;
                border: 1px solid transparent !important;
                border-radius: 16px !important;
                font-weight: 800 !important;
                font-size: 1rem !important;
                height: 48px !important;
                width: 100% !important;
                box-shadow: 0 16px 32px rgba(37, 99, 235, 0.22) !important;
                transition: transform 120ms ease, filter 120ms ease !important;
            }
            [data-testid="stFormSubmitButton"] button p,
            [data-testid="stFormSubmitButton"] button span,
            [data-testid="stFormSubmitButton"] button div,
            button[kind="primary"] p,
            button[kind="primary"] span,
            button[kind="primary"] div {
                color: #FFFFFF !important;
                -webkit-text-fill-color: #FFFFFF !important;
                font-weight: 700 !important;
            }
            [data-testid="stFormSubmitButton"] button:hover {
                filter: brightness(1.03) !important;
                transform: translateY(-1px) !important;
            }
            [data-testid="stFormSubmitButton"] button p,
            [data-testid="stFormSubmitButton"] button span,
            [data-testid="stFormSubmitButton"] button div {
                color: #FFFFFF !important;
                -webkit-text-fill-color: #FFFFFF !important;
                font-weight: 700 !important;
            }
            div[data-testid="stPasswordInput"] button,
            div[data-testid="stPasswordInput"] button:hover,
            div[data-testid="stPasswordInput"] button:focus,
            div[data-testid="stPasswordInput"] button:focus-visible {
                background: transparent !important;
                color: #64748B !important;
                border: none !important;
                border-radius: 0 14px 14px 0 !important;
                width: 48px !important;
                min-height: 46px !important;
                height: 46px !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-shadow: none !important;
                outline: none !important;
                opacity: 1 !important;
                padding: 0 !important;
            }
            div[data-testid="stPasswordInput"] button svg,
            div[data-testid="stPasswordInput"] button path {
                fill: #64748B !important;
                color: #64748B !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sg-auth-page-shell">', unsafe_allow_html=True)
        with st.container(key="sg_auth_card", border=True):
            st.markdown(
                f"""
                <div class="sg-auth-card-inner">
                    <img class="sg-auth-logo" src="{_sg_logo_data_uri()}" alt="StockGuard AI">
                    <span class="sg-badge">Free plan • Privacy-first • AI review</span>
                    <h1 class="auth-title">StockGuard AI</h1>
                    <p class="auth-subtitle">Premium AI review workspace for safer stock uploads.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

            with login_tab:
                with st.form("login_form"):
                    email = st.text_input("Email", key="login_email")
                    password = st.text_input("Password", type="password", key="login_password")
                    submitted = st.form_submit_button("Login", use_container_width=True)
                    st.markdown("<div class=\"sg-auth-note\">Your images stay private and are processed temporarily.</div>", unsafe_allow_html=True)

                if submitted:
                    if login_is_temporarily_locked():
                        st.error("Too many failed login attempts. Please wait a few minutes.")
                    else:
                        user = authenticate_user(email, password)
                        if user:
                            clear_login_attempts()
                            st.session_state["user_id"] = user["id"]
                            st.session_state["page"] = "New Scan" if not is_admin_user(user) else "Dashboard"
                            st.success("Logged in successfully.")
                            st.rerun()
                        else:
                            record_failed_login_attempt()
                            st.error("Invalid email or password.")

            with signup_tab:
                with st.form("signup_form"):
                    name = st.text_input("Name", key="signup_name")
                    email = st.text_input("Email", key="signup_email")
                    password = st.text_input("Password", type="password", key="signup_password")
                    submitted = st.form_submit_button("Create Free Account", use_container_width=True)
                    st.markdown("<div class=\"sg-auth-note\">Start with the free plan. No credit card required.</div>", unsafe_allow_html=True)

                if submitted:
                    result = create_user(email, name, password)
                    if result["ok"]:
                        user = authenticate_user(email, password)
                        if user:
                            st.session_state["user_id"] = user["id"]
                            st.session_state["page"] = "New Scan" if not is_admin_user(user) else "Dashboard"
                            st.success("Account created. Opening the automatic batch cleaner...")
                            st.rerun()
                        st.success(result["message"])
                    else:
                        st.error(result["message"])


            st.markdown(
                '<div class="sg-auth-footer">© 2026 StockGuard AI. A product by BEE CLUSTER.<br>'
                'StockGuard AI is an independent tool and is not affiliated with Adobe.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


def get_sidebar_pages(user: Dict) -> Dict[str, List[str]]:
    """Return the page groups used by the sidebar for normal and admin users."""

    if is_admin_user(user):
        return {
            "main": [
                "New Scan",
                "Scan History",
                "Best Shot Selector",
                "Upload Readiness Report",
                "Auto Scan Summary",
                "CSV Reports",
                "My Exports",
                "Scan Profiles",
            ],
            "account": ["Subscription", "Billing History", "My Profile", "API Access", "Settings"],
            "admin": ["Admin Panel"],
        }

    return {
        "main": ["New Scan", "Scan History", "My Exports", "Subscription", "Settings"],
        "account": [],
        "admin": [],
    }


def install_mobile_sidebar_auto_close() -> None:
    """Close the native Streamlit sidebar after mobile navigation clicks."""

    components.html(
        """
        <script>
        (() => {
            const hostWindow = window.parent;
            const hostDocument = hostWindow.document;
            const handlerKey = "__stockguardMobileSidebarAutoClose";

            if (hostWindow[handlerKey]) return;

            const handleSidebarNavigation = (event) => {
                if (hostWindow.innerWidth > 768) return;

                const navButton = event.target.closest(
                    '[data-testid="stSidebar"] .stButton > button'
                );
                if (!navButton) return;
                if (navButton.closest('[data-testid="stSidebarCollapseButton"]')) return;

                hostWindow.setTimeout(() => {
                    const sidebar = hostDocument.querySelector('[data-testid="stSidebar"]');
                    if (!sidebar || sidebar.getAttribute('aria-expanded') !== 'true') return;

                    const closeButton = hostDocument.querySelector(
                        '[data-testid="stSidebarCollapseButton"] button'
                    );
                    if (closeButton) closeButton.click();
                }, 90);
            };

            hostDocument.addEventListener('click', handleSidebarNavigation, false);
            hostWindow[handlerKey] = handleSidebarNavigation;
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def sidebar_navigation(user: Dict) -> str:
    plan = PLANS[user["plan"]]
    plan_name = plan["plan_name"]
    used = current_month_scan_count(user["id"])
    usage_percent = min(used / plan["monthly_scans"], 1.0) if plan["monthly_scans"] else 0
    usage_width = max(0.0, min(usage_percent * 100, 100.0))
    reset_label = "Monthly reset"
    is_customer = not is_admin_user(user)
    default_page = "New Scan" if is_customer else "Dashboard"
    current_page_name = st.session_state.get("page") or default_page
    current_page_label = nav_display_label(current_page_name)
    avatar_html = _render_avatar_html(user, size=40)
    display_name = user.get("display_name") or user.get("name") or user.get("email", "")

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sg-sidebar-logo">
                <img class="sg-brand-mark" src="{_sg_logo_data_uri()}" alt="StockGuard AI">
                <div class="sg-brand-stack">
                    <div class="sg-brand-name">StockGuard AI</div>
                    <div class="sg-sidebar-product">AI Stock Image Protection</div>
                    <div class="sg-sidebar-company">A product by BEE CLUSTER</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="sidebar_plan_summary"):
            st.markdown(
                f"""
                <div class="sg-usage-card">
                    <div class="sg-usage-header">
                        <div class="sg-usage-plan">{plan_name} Plan</div>
                        <span class="sg-active-status">Active</span>
                    </div>
                    <div class="sg-usage-value">{used} / {plan['monthly_scans']} scans used</div>
                    <div class="sg-usage-meta">
                        <span>{plan['images_per_scan']} images per scan</span>
                        <span>{reset_label}</span>
                    </div>
                    <div class="sg-usage-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{usage_width:.1f}">
                        <div class="sg-usage-progress-fill" style="width: {usage_width:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="sg-sidebar-user">
                {avatar_html}
                <div class="sg-sidebar-user-info">
                    <div class="sg-sidebar-user-name">{escape_html(display_name)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sidebar_pages = get_sidebar_pages(user)
        main_pages = sidebar_pages["main"]
        account_pages = sidebar_pages["account"]
        admin_pages = sidebar_pages["admin"]


        def render_nav_section(title: str, pages: List[str]) -> None:
            st.markdown(
                f'<div class="sg-nav-section-title">{escape_html(title)}</div>',
                unsafe_allow_html=True,
            )
            for page_name in pages:
                label = nav_display_label(page_name)
                if page_name == current_page_name:
                    st.markdown(
                        f'<div class="sg-nav-active">{escape_html(label)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(label, key=f"nav_{page_name}", use_container_width=True):
                        st.session_state["page"] = page_name
                        st.rerun()

        st.markdown(
            '<div class="sg-nav-section-title">Main</div>',
            unsafe_allow_html=True,
        )
        if not is_customer:
            if current_page_name == "Dashboard":
                st.markdown(
                    f'<div class="sg-nav-active">{escape_html(nav_display_label("Dashboard"))}</div>',
                    unsafe_allow_html=True,
                )
            elif st.button(nav_display_label("Dashboard"), key="nav_Dashboard", use_container_width=True):
                st.session_state["page"] = "Dashboard"
                st.rerun()

        for page_name in main_pages:
            label = nav_display_label(page_name)
            if page_name == current_page_name:
                st.markdown(f'<div class="sg-nav-active">{escape_html(label)}</div>', unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state["page"] = page_name
                    st.rerun()

        if account_pages:
            render_nav_section("Account", account_pages)
        if admin_pages:
            render_nav_section("Admin", admin_pages)
        with st.container(key="sidebar_logout_action"):
            if st.button("Logout", key="nav_Logout", use_container_width=True):
                st.session_state["page"] = "Logout"
                st.rerun()

        st.markdown(
            '<div class="sg-sidebar-footer">A product by <span class="sg-sidebar-footer-brand">BEE CLUSTER</span>.</div>',
            unsafe_allow_html=True,
        )
        install_mobile_sidebar_auto_close()

        return current_page_name


def dashboard_page(user: Dict) -> None:
    plan = PLANS[user["plan"]]
    plan_name = plan["plan_name"]
    used = current_month_scan_count(user["id"])
    total_scans = total_scan_count(user["id"])

    render_page_header(
        "Dashboard",
        "Monitor your scan usage, recent activity, and upload readiness.",
        "Overview",
    )
    render_section_card(
        "Premium workflow, ready to launch",
        "Upload images, review similar-content risk, choose the strongest shots, and export a cleaner batch before your next stock submission.",
    )

    cta_columns = st.columns([1.2, 1.2, 3.6])
    with cta_columns[0]:
        if st.button("Start New Scan", type="primary", key="dashboard_start_scan"):
            st.session_state["page"] = "New Scan"
            st.rerun()
    with cta_columns[1]:
        if st.button("View Scan History", key="dashboard_view_history", type="secondary"):
            st.session_state["page"] = "Scan History"
            st.rerun()

    render_section_card("Usage snapshot", "A quick view of your current plan, monthly usage, and export readiness.")
    usage_columns = st.columns(4)
    with usage_columns[0]:
        render_metric_card("Current Plan", plan_name, "Managed from Subscription.", "●")
    with usage_columns[1]:
        render_metric_card("Scans Used", f"{used} / {plan['monthly_scans']}", "Monthly usage limit.", "↗")
    with usage_columns[2]:
        render_metric_card("Images Per Scan", str(plan["images_per_scan"]), "Upload limit for one batch.", "🖼")
    with usage_columns[3]:
        render_metric_card("Clean ZIP Access", "Included" if plan["zip_export"] else "Locked", "Export cleaner batches.", "📦")

    render_section_card("Workflow steps", "Each step stays in the same flow, but the visible guidance is now easier to scan.")
    workflow_columns = st.columns(3)
    with workflow_columns[0]:
        render_action_card("1", "Upload Images", "Prepare a batch of stock images for review.", "Start New Scan")
    with workflow_columns[1]:
        render_action_card("2", "Scan & Review", "Detect duplicates, weak variations, quality issues, and metadata risk.", "Review results")
    with workflow_columns[2]:
        render_action_card("3", "Export Clean Batch", "Download only the selected safe images and reports.", "Export")

    render_section_card("Recent Scan History", "This panel keeps your most recent batches easy to revisit from the dashboard.")
    scans = list_scans(user["id"])[:10]
    if scans:
        recent_rows = []
        for scan in scans[:6]:
            status = "Review" if int(scan["risky_pairs_count"] or 0) else "Clean"
            recent_rows.append(
                {
                    "Batch": scan["batch_name"],
                    "Project": scan.get("project_name") or "No Project",
                    "Date": scan["scan_datetime"],
                    "Total Images": scan["total_images"],
                    "Risky Pairs": scan["risky_pairs_count"],
                    "Near Duplicates": scan["near_duplicate_count"],
                    "Status": status,
                    "Action": "View Report",
                }
            )
        render_light_table(pd.DataFrame(recent_rows).to_dict('records'), list(pd.DataFrame(recent_rows).columns), status_columns={"Status": "success" if False else "warning"}, action_columns={"Action": "Action"})
        render_raw_table_expander("View raw table", pd.DataFrame(scans), caption="Recent scan raw rows")
    else:
        render_empty_state(
            "Scan",
            "No scans yet",
            "Start your first scan to clean your image batch.",
        )


def check_usage_limits(user: Dict, image_count: int) -> Tuple[bool, str]:
    plan = PLANS[user["plan"]]
    plan_name = plan["plan_name"]
    used = current_month_scan_count(user["id"])

    if used >= plan["monthly_scans"]:
        return (
            False,
            f"Monthly scan limit reached for the {plan_name} plan. Upgrade to continue scanning.",
        )

    if image_count > plan["images_per_scan"]:
        return (
            False,
            f"The {plan_name} plan allows {plan['images_per_scan']} images per scan. "
            f"You uploaded {image_count}. Remove images or upgrade your plan.",
        )

    return True, ""


def render_new_scan_header() -> None:
    """Header for the simplified automatic batch-cleaning flow."""

    st.markdown(
        """
        <div class="sg-page-title">
            <h1>Clean Your Stock Image Batch</h1>
            <p>Upload your photos. StockGuard AI finds similar images, weak variations, and risky duplicates, then prepares a cleaner ZIP for upload.</p>
            <p class="sg-muted" style="font-size:0.92rem;">Simple workflow: upload images, let AI review them, confirm suggestions, and download a clean ZIP.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sg-card" style="padding: 1rem; margin-bottom: 1rem; border-radius: 18px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
            <div class="sg-card-title" style="margin-bottom: 0.35rem;">How it works</div>
            <div class="sg-workflow-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;">
                <div class="sg-card" style="padding: 0.8rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px;">1. Upload photos</div>
                <div class="sg-card" style="padding: 0.8rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px;">2. AI reviews the batch</div>
                <div class="sg-card" style="padding: 0.8rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px;">3. Confirm keep / remove</div>
                <div class="sg-card" style="padding: 0.8rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px;">4. Download clean ZIP</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_saas_card_open(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{escape_html(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="sg-saas-card">
            <div class="sg-card-header-row">
                <div>
                    <h3>{escape_html(title)}</h3>
                    {subtitle_html}
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )


def render_saas_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_upload_dropzone(plan: Dict) -> None:
    """Visual upload zone that pairs with Streamlit's file uploader below."""

    st.markdown(
        f"""
        <div class="sg-upload-zone" style="background:#FFFFFF; border:1px dashed #CBD5E1; border-radius:18px; padding:1rem 1rem 1.05rem 1rem; box-shadow:0 18px 32px rgba(15,23,42,0.06);">
            <div class="sg-upload-icon" style="background:linear-gradient(135deg,#EFF6FF,#F5F3FF); border-radius:999px; width:2.7rem; height:2.7rem; display:grid; place-items:center; margin-bottom:0.45rem;">📸</div>
            <h3 style="margin:0.15rem 0 0.2rem 0; color:#0F172A; font-size:1.05rem;">Drag & drop your images here</h3>
            <p style="margin:0 0 0.35rem 0; color:#475569;">or click to browse</p>
            <div class="helper" style="color:#64748B; font-size:0.88rem; line-height:1.45;">
                Supports: JPG, JPEG, PNG, WebP | Max images depends on your plan | Max {MAX_IMAGE_SIZE_MB}MB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scan_mode_copy(mode_key: str) -> Tuple[str, str, int]:
    copies = {
        "Broad Review": (
            "Broad Review",
            "Finds more possible matches. Best for checking large AI batches.",
            SCAN_MODE_PRESETS["Broad Review"],
        ),
        "Balanced Recommended": (
            "Balanced / Recommended",
            "Recommended for most stock upload batches.",
            SCAN_MODE_PRESETS["Balanced Recommended"],
        ),
        "Strict": (
            "Strict / High Sensitivity",
            "Safer for stock uploads. Highlights strong similar-content risk.",
            SCAN_MODE_PRESETS["Strict"],
        ),
        "Near Duplicate Only": (
            "Near Duplicate Only",
            "Only catches almost identical or near-duplicate images.",
            SCAN_MODE_PRESETS["Near Duplicate Only"],
        ),
        "Custom": (
            "Custom",
            "Set your own similarity threshold from 70% to 98%.",
            70,
        ),
    }
    return copies[mode_key]


def render_scan_mode_cards(selected_mode: str, custom_threshold: int | None = None) -> None:
    mode_order = ["Broad Review", "Balanced Recommended", "Strict", "Near Duplicate Only", "Custom"]
    columns = st.columns(5)

    for index, mode_key in enumerate(mode_order):
        title, description, percent = scan_mode_copy(mode_key)
        is_active = mode_key == selected_mode
        if mode_key == "Custom":
            percent_label = "70–98%"
        else:
            percent_label = f"{percent}%"

        with columns[index]:
            card_class = "sg-scan-mode-card active" if is_active else "sg-scan-mode-card"
            st.markdown(f"<div class='{card_class}' style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:0.9rem;'>", unsafe_allow_html=True)
            st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem;'><strong>{escape_html(title)}</strong>{'<span class=\'sg-pill sg-pill-success\'>Selected</span>' if is_active else ''}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.8rem; font-weight:800; color:#111827; margin-top:0.35rem;'>{escape_html(percent_label)}</div>", unsafe_allow_html=True)
            st.caption(escape_html(description))
            if st.button(
                "Use this mode" if not is_active else "Selected",
                key=f"scan_mode_select_{mode_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["scan_mode_radio"] = mode_key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_compact_uploaded_preview(images: List[UploadedImage]) -> None:
    """Show a short preview row so the New Scan page does not become too long."""

    st.markdown(
        f"""
        <div class="sg-card-header-row" style="margin-top:1rem;">
            <div>
                <h3>Uploaded Files ({len(images)})</h3>
                <p><span class="sg-plan-badge">{len(images)} images ready to scan</span></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    preview_images = images[:12]
    columns = st.columns(6)
    for index, image in enumerate(preview_images):
        with columns[index % 6]:
            st.image(image.thumbnail, use_container_width=True)
            st.caption(image.name)
            st.markdown("<span class='sg-plan-badge'>Valid</span>", unsafe_allow_html=True)
    if len(images) > len(preview_images):
        st.info(f"+{len(images) - len(preview_images)} more image(s) will also be scanned.")


def render_scan_helper_panel(
    image_count: int,
    plan: Dict,
    result: Dict | None = None,
) -> None:
    """Right-side helper cards for the New Scan workflow."""

    if result:
        low_count = max(len(result["images"]) - len(result["risky_pairs"]), 0)
        review_count = len(result["risky_pairs"])
        high_count = len(result["near_duplicates"])
        similar_groups = len(result["groups"])
    else:
        low_count = image_count
        review_count = 0
        high_count = 0
        similar_groups = 0
    total = max(low_count + review_count + high_count, 1)
    low_percent = low_count / total * 100
    review_percent = review_count / total * 100
    high_percent = high_count / total * 100

    st.markdown("<div class='sg-right-panel'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sg-helper-card">
            <h3>Scan Summary (Preview)</h3>
            <p class="sg-muted">Images count: <strong>{image_count}</strong></p>
            <p class="sg-muted">Similar Groups: <strong>{similar_groups}</strong></p>
            <p class="sg-muted">High Risk: <strong>{high_count}</strong></p>
            <div class="sg-risk-bar"><div></div><div></div><div></div></div>
            <p class="sg-muted">Low Risk {low_percent:.0f}% | Review Needed {review_percent:.0f}% | High Risk {high_percent:.0f}%</p>
        </div>
        <div class="sg-helper-card">
            <h3>What we check</h3>
            <ul>
                <li>Visual similarity with ResNet50 AI</li>
                <li>Image quality and sharpness</li>
                <li>Metadata similarity for titles and tags</li>
                <li>Best shot selection</li>
                <li>Upload readiness analysis</li>
            </ul>
        </div>
        <div class="sg-helper-card">
            <h3>After Scan</h3>
            <ul>
                <li>Get Upload Readiness Report</li>
                <li>Review Similar Groups</li>
                <li>Download Clean ZIP</li>
                <li>Export CSV Report</li>
            </ul>
        </div>
        <div class="sg-helper-card">
            <h3>Privacy First</h3>
            <p class="sg-muted">Your images are processed temporarily and are not stored permanently. You can delete files anytime.</p>
            <p><a href="#">Learn more</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def suggest_next_batch_name(existing_names: List[str]) -> str:
    """Suggest Batch 001, Batch 002, etc. without reusing an old batch name."""

    highest_number = 0
    for name in existing_names:
        match = re.fullmatch(r"Batch\s+(\d+)", name.strip(), flags=re.IGNORECASE)
        if match:
            highest_number = max(highest_number, int(match.group(1)))

    return f"Batch {highest_number + 1:03d}"


def process_scan(
    user: Dict,
    project_name: str,
    batch_name: str,
    valid_images: List[UploadedImage],
    threshold_percent: int,
    metadata_rows: Dict[str, Dict] | None = None,
    metadata_pairs: List[Dict] | None = None,
) -> Dict:
    device = get_device()
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.info("Reading your images...")
    progress_bar.progress(8)
    with st.spinner("Loading the review model..."):
        model = load_model(str(device))

    with st.spinner("Processing images and calculating similarity scores..."):
        status_text.info("Checking image quality and previews...")
        progress_bar.progress(18)

        status_text.info("Checking image quality...")
        quality_rows = calculate_image_quality(valid_images)
        progress_bar.progress(38)

        status_text.info("Extracting visual features...")
        embeddings = create_embeddings(valid_images, model, device)
        progress_bar.progress(62)

        status_text.info("Comparing visual similarity...")
        risky_pairs, all_pairs, highest_similarity = compare_images(
            valid_images,
            embeddings,
            float(threshold_percent),
        )
        progress_bar.progress(82)

        status_text.info("Finding duplicate groups and best shots...")
        groups = build_groups(len(valid_images), risky_pairs)
        near_duplicates = [
            pair for pair in all_pairs if pair["similarity_percent"] >= NEAR_DUPLICATE_PERCENT
        ]
        export_rows = assign_group_numbers(groups, risky_pairs)
        export_rows = [
            {key: sanitize_csv_cell(value) for key, value in row.items()}
            for row in export_rows
        ]
        progress_bar.progress(94)

    status_text.info("Preparing your clean ZIP and report...")
    project_id = get_or_create_project(user["id"], project_name)
    scan_id = save_scan(
        user_id=user["id"],
        project_id=project_id,
        batch_name=batch_name,
        total_images=len(valid_images),
        risky_pairs_count=len(risky_pairs),
        near_duplicate_count=len(near_duplicates),
        highest_similarity_score=highest_similarity,
        csv_rows=export_rows,
    )
    progress_bar.progress(100)
    status_text.success("Your batch review is ready.")

    return {
        "scan_id": scan_id,
        "device": str(device),
        "project_name": project_name,
        "batch_name": batch_name,
        "threshold_percent": threshold_percent,
        "images": valid_images,
        "risky_pairs": risky_pairs,
        "all_pairs": all_pairs,
        "groups": groups,
        "near_duplicates": near_duplicates,
        "highest_similarity": highest_similarity,
        "quality_rows": quality_rows,
        "metadata_rows": metadata_rows or {},
        "metadata_pairs": metadata_pairs or [],
        "csv_data": pd.DataFrame(
            export_rows,
            columns=[
                "group_number",
                "image_1",
                "image_2",
                "similarity_percent",
                "risk_level",
                "recommended_action",
            ],
        ),
    }


def new_scan_page(user: Dict) -> None:
    plan = PLANS[user["plan"]]
    plan_name = plan["plan_name"]
    used = current_month_scan_count(user["id"])
    render_new_scan_header()
    render_workflow_steps(1)

    result = st.session_state.get("last_scan_result")
    if result:
        st.info("A previous scan is still loaded in this tab. Use the button below to clear it and upload a fresh batch.")
        if st.button("Start New Scan", key="start_new_scan_from_results", use_container_width=True):
            reset_scan_session_state()
            st.rerun()

    uploaded_files = []
    valid_images: List[UploadedImage] = []
    corrupted_files: List[str] = []
    metadata_rows: Dict[str, Dict] = {}
    metadata_pairs: List[Dict] = []

    main_column, helper_column = st.columns([0.7, 0.3], gap="large")

    with main_column:
        st.markdown(
            f"""
            <div class="sg-saas-card">
                <div class="sg-card-header-row">
                    <div>
                        <h3>Batch Details</h3>
                        <p>Choose where this scan should be saved.</p>
                    </div>
                    <span class="sg-plan-badge">{escape_html(plan_name)} Plan</span>
                </div>
                <p class="sg-muted">{used}/{plan['monthly_scans']} scans used this month | {plan['images_per_scan']} images per scan</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        project_name = st.text_input("Project / Client folder name", value="Default Project")
        existing_batch_names = list_batch_names(user["id"], project_name)
        suggested_batch_name = suggest_next_batch_name(existing_batch_names)
        batch_name = st.text_input(
            "Batch name",
            value=suggested_batch_name,
            key=f"batch_name_for_{project_name}",
            help="If this project already has Batch 001, the app will suggest Batch 002 automatically.",
        )
        if batch_name.strip() in existing_batch_names:
            st.warning("This batch name already exists in this project. Use the suggested next name to keep history cleaner.")

        with st.expander("Advanced options", expanded=False):
            st.caption("Balanced / Recommended is the default mode for most stock uploads.")
            if plan.get("advanced_scan_modes"):
                scan_mode_options = [
                    "Broad Review",
                    "Balanced Recommended",
                    "Strict",
                    "Near Duplicate Only",
                    "Custom",
                ]
            else:
                scan_mode_options = ["Balanced Recommended"]
                st.info("Advanced scan modes are available on Starter and above.")

            scan_mode = st.session_state.get("scan_mode_radio", DEFAULT_SCAN_MODE)
            if scan_mode not in scan_mode_options:
                scan_mode = scan_mode_options[0]
            st.session_state["scan_mode_radio"] = scan_mode

            if scan_mode == "Custom":
                threshold_percent = st.slider(
                    "Custom similarity threshold",
                    min_value=70,
                    max_value=98,
                    value=85,
                    step=1,
                    help="Pairs at or above this percentage will be marked as risky.",
                )
            else:
                threshold_percent = SCAN_MODE_PRESETS[scan_mode]
            render_scan_mode_cards(scan_mode, threshold_percent)
            st.caption("Higher thresholds catch only very similar images. Lower thresholds catch more weak variations but may need more review.")

        st.markdown(
            """
            <div class="sg-saas-card">
                <div class="sg-card-header-row">
                    <div>
                        <h3>Upload Images</h3>
                        <p>Upload a batch, then start the scan when the preview looks correct.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload Images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="scan_file_uploader",
            help="Drag & drop your images here, or click Upload to browse. Supports JPG, JPEG, PNG, and WebP. Max images depends on your plan. Max 25MB per image.",
        )
        st.caption("Drag & drop your images here, or click Upload to browse.")
        st.caption(f"Supports JPG, JPEG, PNG, and WebP. Max images depends on your plan. Max {MAX_IMAGE_SIZE_MB}MB per image.")
        st.markdown(
            """
            <div class="sg-saas-card" style="padding:0.9rem 1rem;">
                <strong>Privacy note:</strong>
                <span class="sg-muted"> Uploaded images are processed temporarily for similarity analysis and are not stored permanently.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Learn more about our privacy policy.", key="privacy_policy_link", use_container_width=False):
            st.session_state["page"] = "Privacy Policy"
            st.rerun()

    with helper_column:
        render_scan_helper_panel(0, plan, result=result)

    if not uploaded_files:
        render_app_footer()
        return

    selected_file_count = len(uploaded_files)
    if len(uploaded_files) > plan["images_per_scan"]:
        allowed_count = plan["images_per_scan"]
        extra_count = len(uploaded_files) - allowed_count
        st.error(
            f"Upload limit reached. Your {plan_name} plan allows only "
            f"{allowed_count} images per scan."
        )
        st.warning(
            f"The first {allowed_count} image(s) will be accepted for this scan. "
            f"The extra {extra_count} image(s) will be ignored. Upgrade your plan "
            "if you want to scan more images in one batch."
        )
        if st.button("View Upgrade Options", key="upgrade_options_image_limit"):
            st.session_state["page"] = "Subscription"
            st.rerun()
        uploaded_files = uploaded_files[:allowed_count]

    upload_signature_value = tuple(
        upload_signature(uploaded_file)
        for uploaded_file in uploaded_files
    )
    scan_context = (
        user["id"],
        project_name,
        batch_name,
        scan_mode,
        threshold_percent,
        upload_signature_value,
    )
    if st.session_state.get("active_scan_context") != scan_context:
        st.session_state["active_scan_context"] = scan_context
        st.session_state.pop("last_scan_result", None)

    if len(uploaded_files) > 100:
        st.warning("You uploaded more than 100 images. CPU processing may be slow.")
    if len(uploaded_files) > 300:
        st.error(
            "Large batch warning: more than 300 accepted images can take a long time "
            "and may use a lot of memory on CPU."
        )

    with st.spinner("Loading images safely and creating optimized previews..."):
        valid_images, corrupted_files = read_uploaded_images(uploaded_files)
    if corrupted_files:
        st.warning("Skipped unsafe or unsupported file(s): " + "; ".join(corrupted_files))

    if not valid_images:
        st.error("No valid images were found.")
        return

    with main_column:
        st.success(
            f"Selected {selected_file_count} file(s). "
            f"Accepted {len(valid_images)} file(s) for this {plan_name} plan scan."
        )
        render_compact_uploaded_preview(valid_images)
        action_cols = st.columns([1.1, 1.1, 2.2, 1.2])
        with action_cols[0]:
            with st.expander("Advanced Settings"):
                st.caption(f"Threshold: {threshold_percent}%")
                st.caption(f"Project: {project_name}")
                st.caption(f"Batch: {batch_name}")
        with action_cols[1]:
            if st.button("Remove All", key="remove_all_upload_notice"):
                reset_scan_session_state()
                st.info("The current upload batch has been cleared. Select a fresh set of images to continue.")
                st.rerun()
        with action_cols[3]:
            st.caption(f"This will use 1 scan credit for {len(valid_images)} images.")

        metadata_rows, metadata_pairs = render_metadata_input(valid_images, plan)

    if len(valid_images) < 2:
        st.info("Upload at least two valid images to compare similarity.")
        return

    can_scan, message = check_usage_limits(user, len(valid_images))
    if not can_scan:
        st.error(message)
        if st.button("View Upgrade Options", key="upgrade_options_scan_limit"):
            st.session_state["page"] = "Subscription"
            st.rerun()
        return

    with main_column:
        start_col, helper_text_col = st.columns([1.2, 2.5])
        with start_col:
            if st.button("Start AI Review", type="primary", key="process_scan_button", use_container_width=True):
                st.session_state["last_scan_result"] = process_scan(
                    user,
                    project_name,
                    batch_name,
                    valid_images,
                    threshold_percent,
                    metadata_rows,
                    metadata_pairs,
                )
                st.success("Scan completed and saved to your account.")
        with helper_text_col:
            st.caption(f"This will use 1 scan credit. Your plan allows {plan['monthly_scans']} scans per month.")

    result = st.session_state.get("last_scan_result")
    if not result:
        render_app_footer()
        return

    st.markdown("<div class='sg-page-title'><h1>Your batch is ready</h1><p>Review the suggestions, confirm what to keep, and download your clean ZIP when the batch looks right.</p></div>", unsafe_allow_html=True)
    render_workflow_steps(2)
    images = result["images"]
    risky_pairs = result["risky_pairs"]
    all_pairs = result["all_pairs"]
    groups = result["groups"]
    near_duplicates = result["near_duplicates"]
    highest_similarity = result["highest_similarity"]
    quality_rows = result["quality_rows"]
    metadata_rows = result.get("metadata_rows", {})
    metadata_pairs = result.get("metadata_pairs", [])
    metadata_summary = build_metadata_summary(images, metadata_rows, metadata_pairs)
    csv_data = result["csv_data"]

    st.caption(f"Using device: {result['device']}")
    st.caption(f"Saved scan ID: {result['scan_id']}")
    result_actions_placeholder = st.empty()

    best_shot_decisions = {}
    best_shot_remove_indexes = []

    suggestions_tab, readiness_tab, report_tab, raw_details_tab = st.tabs(
        ["Suggestions", "Readiness", "Report", "Raw Details"]
    )

    with suggestions_tab:
        overall_status = "High Risk" if near_duplicates else "Review Needed" if risky_pairs else "Ready to Upload"
        st.caption("Keep = safe to include in the clean ZIP. Review Needed = manual check. Remove Suggested = risky or weak variation to exclude.")
        st.subheader("Upload Readiness Score")
        render_scan_metric_cards(
            total_images=len(images),
            risky_pair_count=len(risky_pairs),
            group_count=len(groups),
            highest_similarity=highest_similarity,
            near_duplicate_count=len(near_duplicates),
        )
        st.info(
            f"Overall status: {overall_status}. "
            "This view keeps your main decisions simple: keep, review, or remove suggested."
        )
        if near_duplicates:
            st.warning("Similar images were found. Review the strongest candidates first and keep only the best image in each group.")
        elif risky_pairs:
            st.warning("Review the suggested groups and keep only the clearest variations.")
        else:
            st.success("Great! No risky similar groups were found. You can download your full batch or review the report.")
        render_metadata_similarity_report(images, metadata_rows, metadata_pairs, plan["csv_export"])

        if plan["best_shot"]:
            best_shot_decisions = render_best_shot_selector(groups, images, quality_rows, all_pairs)
            best_shot_remove_indexes = [
                index for index, decision in best_shot_decisions.items() if decision == "Remove"
            ]
        remove_indexes = render_keep_remove_workflow(
            risky_pairs,
            images,
            default_remove_indexes=best_shot_remove_indexes,
            csv_export_enabled=plan["csv_export"],
        )
        render_groups(groups, risky_pairs, images, quality_rows, all_pairs)

    auto_decision_rows = build_auto_decision_rows(
        project_name=project_name,
        batch_name=batch_name,
        images=images,
        quality_rows=quality_rows,
        all_pairs=all_pairs,
        groups=groups,
        near_duplicates=near_duplicates,
        remove_indexes=remove_indexes,
        best_shot_decisions=best_shot_decisions,
        metadata_summary=metadata_summary,
    )
    auto_decision_df = pd.DataFrame(auto_decision_rows)
    keep_count = int((auto_decision_df["final_status"] == "Keep").sum()) if not auto_decision_df.empty else 0
    review_count = int((auto_decision_df["final_status"] == "Review Needed").sum()) if not auto_decision_df.empty else 0
    remove_count = int((auto_decision_df["final_status"] == "Remove Recommended").sum()) if not auto_decision_df.empty else 0

    final_zip_remove_indexes = remove_indexes
    if not auto_decision_df.empty:
        final_zip_remove_indexes = auto_zip_remove_indexes(auto_decision_df.to_dict("records"), len(images))

    with result_actions_placeholder.container():
        st.markdown(
            """
            <div class="sg-card" style="padding: 1rem; margin-bottom: 1rem; border-radius: 18px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
                <div class="sg-card-title" style="margin-bottom: 0.25rem;">Your batch is ready</div>
                <p class="sg-muted" style="margin-bottom: 0.6rem;">Keep = safe to include in your clean ZIP. Review Needed = check manually before upload. Remove Suggested = risky, similar, or weak variation; exclude from the final batch.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_metric_cards(
            [
                ("Total", "Total images", str(len(images)), "Images included in this batch."),
                ("Keep", "Keep", str(keep_count), "Safe to include in the clean ZIP."),
                ("Review", "Review needed", str(review_count), "Needs a human check before upload."),
                ("Remove", "Remove suggested", str(remove_count), "Best to exclude from the final ZIP."),
            ]
        )
        render_result_export_actions(
            user=user,
            images=images,
            final_zip_remove_indexes=final_zip_remove_indexes,
            report_csv=auto_decision_df,
            batch_name=batch_name,
        )

    readiness_df = pd.DataFrame()
    if plan["readiness_report"]:
        readiness_df = pd.DataFrame(
            build_upload_readiness_rows(
                project_name=project_name,
                batch_name=batch_name,
                images=images,
                quality_rows=quality_rows,
                all_pairs=all_pairs,
                groups=groups,
                remove_indexes=remove_indexes,
                best_shot_decisions=best_shot_decisions,
                metadata_summary=metadata_summary,
            )
        )

    with readiness_tab:
        review_rows = [row for row in auto_decision_rows if row["final_status"] == "Review Needed"]
        remove_rows = [row for row in auto_decision_rows if row["final_status"] == "Remove Recommended"]
        if review_rows or remove_rows:
            st.caption("Keep, review, or remove suggested images are grouped here for quick decision-making.")
        render_filtered_image_list("Review Needed", review_rows, images, "No images need review right now.")
        render_filtered_image_list("Remove Suggested", remove_rows, images, "No images are currently recommended for removal.")

    with report_tab:
        render_workflow_steps(3)
        render_final_client_export(
            user=user,
            images=images,
            auto_decision_df=auto_decision_df,
            final_zip_remove_indexes=final_zip_remove_indexes,
            similarity_csv=csv_data,
        )

    with raw_details_tab:
        st.subheader("Advanced similarity details")
        render_near_duplicates(near_duplicates)
        render_top_risky_pairs(risky_pairs, images)
        with st.expander("All risky pair details"):
            if risky_pairs:
                render_light_html_table(csv_data, caption="All risky pair detail rows")
            else:
                st.info("No risky pairs found at the selected threshold.")
    render_app_footer()


def scan_history_page(user: Dict) -> None:
    render_page_header(
        "My Scans",
        "Review your saved batches, clean-up suggestions, and exported reports in one simple place.",
        "History",
    )
    plan = get_current_plan(user)
    if not plan["batch_history"]:
        render_locked_feature_card(
            "Full scan history is available on Pro",
            (
                "Full scan history is available on Pro. Your current scan downloads are available immediately after each scan. "
                "Use the latest scan results to review suggestions, download reports, and start another batch."
            ),
            "Pro",
        )
        if st.button("Upgrade to Pro", key="history_upgrade_to_pro", type="primary"):
            update_user_plan(user["id"], "Pro")
            st.success("Plan changed to Pro.")
            st.rerun()
        return

    scans = list_scans(user["id"])

    if not scans:
        render_empty_state(
            "History",
            "No scans yet",
            "Upload your first image batch and let StockGuard AI clean it automatically.",
            cta_text="Start New Scan",
            target_page="New Scan",
        )
        return

    render_section_card(
        "Saved batch reports",
        "Use the filter card below to quickly find the right saved batch and review its status.",
    )

    st.markdown("<div class='sg-filter-card'>", unsafe_allow_html=True)
    st.markdown("<div class='sg-card-title' style='margin-bottom: 0.35rem;'>Filter scans</div>", unsafe_allow_html=True)
    filter_cols = st.columns([1.6, 1.1, 1.1, 0.55])
    with filter_cols[0]:
        history_search = st.text_input("Search", placeholder="Search batch or project", key="history_search")
    with filter_cols[1]:
        status_filter = st.selectbox("Status", ["All", "Clean", "Review"], key="history_status")
    with filter_cols[2]:
        project_values = sorted({scan.get("project_name") or "No Project" for scan in scans})
        project_filter = st.selectbox("Project", ["All"] + project_values, key="history_project")
    with filter_cols[3]:
        st.caption(" ")
        st.button("Apply", key="history_filter_icon", help="Filter scans", type="secondary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    filtered_scans = []
    for scan in scans:
        scan_status = "Review" if int(scan["risky_pairs_count"] or 0) else "Clean"
        project_value = scan.get("project_name") or "No Project"
        search_blob = f"{scan['batch_name']} {project_value}".lower()
        if history_search and history_search.lower() not in search_blob:
            continue
        if status_filter != "All" and scan_status != status_filter:
            continue
        if project_filter != "All" and project_value != project_filter:
            continue
        filtered_scans.append(scan)

    history_rows = [
        {
            "Batch": scan["batch_name"],
            "Project": scan.get("project_name") or "No Project",
            "Date": scan["scan_datetime"],
            "Total Images": scan["total_images"],
            "Risky Pairs": scan["risky_pairs_count"],
            "Near Duplicates": scan["near_duplicate_count"],
            "Highest": f"{scan['highest_similarity_score']:.2f}%",
            "Status": "Review" if int(scan["risky_pairs_count"] or 0) else "Clean",
            "Action": "Choose below",
        }
        for scan in filtered_scans[:25]
    ]
    if history_rows:
        st.markdown("<div class='sg-card-title' style='margin-bottom: 0.35rem;'>Recent batch summary</div>", unsafe_allow_html=True)
        render_light_html_table(pd.DataFrame(history_rows), caption="Light table view for the primary scan history list.")
        render_raw_table_expander("View raw table", pd.DataFrame(history_rows), caption="Scan history raw rows")
    else:
        render_empty_state("History", "No matching scans", "Try changing your search or filters.")
        return

    scan_options = {
        f"#{scan['id']} - {scan['project_name'] or 'No Project'} - {scan['batch_name']} - {scan['scan_datetime']}": scan[
            "id"
        ]
        for scan in filtered_scans
    }
    selected_label = st.selectbox("Choose a scan", list(scan_options.keys()))
    selected_scan = get_scan(user["id"], scan_options[selected_label])

    if not selected_scan:
        st.error("This scan was not found.")
        return

    render_metric_cards(
        [
            ("Project", "Project", selected_scan.get("project_name") or "No Project", "Saved project/client folder."),
            ("Images", "Total images", str(selected_scan["total_images"]), "Images in this completed scan."),
            ("Risk", "Risky pairs", str(selected_scan["risky_pairs_count"]), "Pairs above threshold."),
            ("Near", "Near duplicates", str(selected_scan["near_duplicate_count"]), "Pairs at 95% or higher."),
            (
                "Top",
                "Highest score",
                f"{selected_scan['highest_similarity_score']:.2f}%",
                "Strongest pair similarity.",
            ),
        ]
    )

    rows = scan_report_rows(selected_scan)
    st.subheader("Saved CSV Report Data")
    st.info(
        "CSV/report data is available for saved scans. ZIP export is generated after scan completion "
        "and is not stored in this beta version."
    )
    if rows:
        report_df = pd.DataFrame(rows)
        if plan["csv_export"]:
            st.download_button(
                "Download saved CSV report",
                data=csv_download_bytes(report_df),
                file_name=f"scan_{selected_scan['id']}_report.csv",
                mime="text/csv",
            )
        else:
            st.info("CSV export is not enabled for your current plan.")
        with st.expander("View detailed report", expanded=False):
            st.markdown(
                """
                <div class="sg-card" style="padding: 0.8rem; margin-bottom: 0.65rem; border-radius: 16px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 16px 30px rgba(15, 23, 42, 0.06);">
                    <div class="sg-card-title" style="margin-bottom: 0.15rem;">Detailed report</div>
                    <div class="sg-muted">Readable, light table layout with horizontal scrolling inside the card only.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_light_html_table(report_df, caption="Detailed report rows")
            render_raw_table_expander("View raw report table", report_df, caption="Raw report rows")
    else:
        render_empty_state(
            "OK",
            "No risky pairs in this report",
            "This completed scan did not save any risky pair rows.",
        )


def projects_page(user: Dict) -> None:
    render_page_header(
        "Projects",
        "Organize scans by batch, client, or stock upload theme.",
        "Projects",
    )

    with st.container(border=True):
        st.subheader("Create Project")
        st.caption("Use projects to keep client batches and stock collections organized.")
        with st.form("create_project_form"):
            project_name = st.text_input("Project name", placeholder="Client A / Product Photos June")
            submitted = st.form_submit_button("Create Project", use_container_width=True)

    if submitted:
        if project_name.strip():
            get_or_create_project(user["id"], project_name)
            st.success("Project created.")
            st.rerun()
        else:
            st.error("Please enter a project name.")

    projects = list_projects(user["id"])
    if projects:
        st.subheader("Your Projects")
        columns = st.columns(2)
        for index, project in enumerate(projects):
            with columns[index % 2]:
                st.markdown(
                    f"""
                    <div class="project-card">
                        <div class="sg-card-title">{project["name"]}</div>
                        <div class="sg-muted">Created {project["created_at"]}</div>
                        <p>{project["scan_count"]} saved scan(s)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Start Scan",
                    key=f"start_project_{project['id']}",
                    use_container_width=True,
                ):
                    st.session_state["page"] = "New Scan"
                    st.rerun()
        render_raw_table_expander("View raw projects table", pd.DataFrame(projects), caption="Project raw rows")
    else:
        render_empty_state(
            "Project",
            "No projects yet",
            "No projects yet. Create a project to organize your scans.",
        )


def readiness_report_page(user: Dict) -> None:
    """Standalone navigation page for the latest upload-readiness result."""

    render_page_header(
        "Upload Readiness Report",
        "Know what to upload, what to review, and what to remove.",
        "Quality intelligence",
    )

    plan = get_current_plan(user)
    if not plan["readiness_report"]:
        render_locked_feature_card(
            "Upload Readiness Report is a paid feature",
            (
                "Starter, Pro, and Agency users get quality scores, Best Shot suggestions, "
                "readiness status, and professional CSV reports."
            ),
            "Starter",
        )
        starter_plan = get_subscription_plan("starter")
        upgrade_label = starter_plan["plan_name"] if starter_plan else "Starter"
        if st.button(f"Upgrade to {upgrade_label}", key="readiness_page_upgrade", type="primary"):
            update_user_plan(user["id"], "starter")
            st.success(f"Plan changed to {upgrade_label}.")
            st.rerun()
        return

    result = st.session_state.get("last_scan_result")
    if not result:
        render_empty_state(
            "Report",
            "No active readiness report yet",
            "Run a new scan first. After processing, this page will summarize ready, review, and remove recommendations.",
            "Start New Scan",
        )
        if st.button("Start New Scan", key="readiness_start_scan", type="primary"):
            st.session_state["page"] = "New Scan"
            st.rerun()
        feature_columns = st.columns(3)
        with feature_columns[0]:
            render_feature_card("Ready", "Ready to Upload", "Green cards show the cleanest upload candidates.")
        with feature_columns[1]:
            render_feature_card("Review", "Review Needed", "Yellow cards explain images that need a human decision.")
        with feature_columns[2]:
            render_feature_card("Remove", "Remove Recommended", "Red cards flag near duplicates or weak images.")
        return

    quality_rows = result["quality_rows"]
    readiness_rows = build_upload_readiness_rows(
        project_name=result.get("project_name", "Latest Project"),
        batch_name=result.get("batch_name", "Latest Batch"),
        images=result["images"],
        quality_rows=quality_rows,
        all_pairs=result["all_pairs"],
        groups=result["groups"],
        remove_indexes=[],
        best_shot_decisions={},
    )
    readiness_df = pd.DataFrame(readiness_rows)
    ready_count = int((readiness_df["upload_readiness_status"] == "Ready to Upload").sum()) if not readiness_df.empty else 0
    review_count = int((readiness_df["upload_readiness_status"] == "Review Needed").sum()) if not readiness_df.empty else 0
    remove_count = int((readiness_df["upload_readiness_status"] == "Remove Recommended").sum()) if not readiness_df.empty else 0
    average_quality = (
        sum(row["quality_score"] for row in quality_rows) / len(quality_rows)
        if quality_rows
        else 0.0
    )
    st.caption(
        f"Latest scan: {result.get('project_name', 'Project')} / "
        f"{result.get('batch_name', 'Batch')} at {result.get('threshold_percent', 80)}% threshold"
    )
    render_metric_cards(
        [
            ("Images", "Total images", str(len(result["images"])), "Latest processed scan."),
            ("Risk", "Risky pairs", str(len(result["risky_pairs"])), "Pairs above threshold."),
            ("Near", "Near duplicates", str(len(result["near_duplicates"])), "Pairs at 95% or higher."),
            ("Score", "Average quality", f"{average_quality:.2f}", "Average quality score out of 100."),
        ]
    )
    status_columns = st.columns(3)
    with status_columns[0]:
        render_status_card("OK", "Ready to Upload", str(ready_count), "Cleaner candidates in the latest scan.", "green")
    with status_columns[1]:
        render_status_card("Review", "Review Needed", str(review_count), "Needs a human decision before upload.", "orange")
    with status_columns[2]:
        render_status_card("Remove", "Remove Recommended", str(remove_count), "High risk or low quality candidates.", "red")

    if not readiness_df.empty:
        if plan["csv_export"]:
            st.download_button(
                "Download latest readiness CSV",
                data=csv_download_bytes(readiness_df),
                file_name="latest_upload_readiness_report.csv",
                mime="text/csv",
            )
        else:
            st.info("CSV export is not enabled for your current plan.")
        render_raw_table_expander("View detailed readiness data", readiness_df, caption="Readiness raw detail table")





def logout_page() -> None:
    st.session_state.clear()
    st.success("Logged out.")
    st.rerun()


def render_payment_request_card(payment: Dict) -> None:
    """Show beta manual payment instructions after a user requests an upgrade."""

    st.markdown(
        f"""
        <div class="sg-card">
            <div class="sg-badge">Payment request created</div>
            <h3>Manual payment required</h3>
            <p class="sg-muted">
                Please complete payment using the available payment method and send proof to admin.
                Your plan will be activated after approval.
            </p>
            <p><strong>Payment reference:</strong> {escape_html(payment['payment_ref'])}</p>
            <p><strong>Selected plan:</strong> {escape_html(payment['plan_name'])}</p>
            <p><strong>Amount:</strong> {float(payment['amount']):.2f} {escape_html(payment['currency'])}</p>
            <p><strong>User email:</strong> {escape_html(payment['user_email'])}</p>
            <p><strong>Payment method:</strong> {escape_html(payment.get('payment_method') or 'Manual')}</p>
            <p><strong>Status:</strong> {escape_html(str(payment.get('payment_status', 'pending')).title())}</p>
            <p class="sg-muted">
                Use this reference when sending proof. Admin approval activates the plan.
                Future PayHere, Paddle, Lemon Squeezy, or Stripe gateways can connect to this same payment record.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def payment_status_badge(status: str) -> str:
    """Return a small HTML status badge for manual payment status."""

    clean_status = (status or "pending").strip().lower()
    tone_map = {
        "pending": ("#F59E0B", "rgba(245, 158, 11, 0.14)"),
        "paid": ("#22C55E", "rgba(34, 197, 94, 0.14)"),
        "rejected": ("#EF4444", "rgba(239, 68, 68, 0.14)"),
        "failed": ("#EF4444", "rgba(239, 68, 68, 0.14)"),
        "refunded": ("#94A3B8", "rgba(148, 163, 184, 0.14)"),
    }
    color, background = tone_map.get(clean_status, tone_map["pending"])
    return (
        f'<span style="display:inline-flex;align-items:center;border-radius:999px;'
        f'padding:0.22rem 0.65rem;background:{background};color:{color};'
        f'border:1px solid {color};font-size:0.82rem;font-weight:700;">'
        f'{escape_html(clean_status.title())}</span>'
    )


def render_finance_summary_cards(summary: Dict) -> None:
    """Render finance metrics with the same premium card style used elsewhere in the app."""

    cards = [
        ("Total Revenue", f"${summary['total_revenue']:.2f}", "All approved payments", "💸"),
        ("This Month Revenue", f"${summary['this_month_revenue']:.2f}", "Revenue collected this month", "📅"),
        ("Pending Payments", str(summary["pending_payments"]), "Waiting for admin approval", "⏳"),
        ("Paid Payments", str(summary["paid_payments"]), "Approved payment records", "✅"),
        ("Refunded Payments", str(summary["refunded_payments"]), "Refunded payment records", "↩️"),
        ("Active Paid Users", str(summary["active_paid_users"]), "Enabled users on paid plans", "👥"),
    ]
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, (label, value, description, icon) in zip(columns, cards[start : start + 3]):
            with column:
                render_metric_card(label, value, description, icon)


def render_plan_revenue_cards(plan_rows: List[Dict]) -> None:
    """Render plan revenue details with the light StockGuard table style instead of raw default tables."""

    rows = [
        {
            "Plan": row["plan_name"],
            "Price": plan_price_label(row),
            "Monthly scan limit": int(row["monthly_scans"]),
            "Paid users": int(row.get("paid_users") or 0),
            "Revenue": f"${float(row.get('revenue') or 0):.2f}",
        }
        for row in plan_rows
    ]
    st.caption("Plan revenue summary")
    render_light_html_table(pd.DataFrame(rows), caption="Plan revenue summary")


def render_payments_table(payment_rows: List[Dict]) -> None:
    """Render the payment list with the same light StockGuard table style used elsewhere in the app."""

    rows = []
    for payment in payment_rows:
        manage_text = "Manage / Review" if payment["payment_status"] == "pending" else "View details"
        rows.append(
            {
                "Payment Ref": payment["payment_ref"],
                "User ID": payment.get("public_user_id") or payment.get("user_id") or "",
                "Email": payment["user_email"],
                "Plan": payment["plan_name"],
                "Amount": f"{float(payment['amount'] or 0):.2f}",
                "Currency": payment["currency"],
                "Status": str(payment["payment_status"]).title(),
                "Method": payment.get("payment_method") or "Manual",
                "Created": payment["created_at"],
                "Actions": manage_text,
            }
        )
    st.caption("Payment request details")
    render_light_html_table(pd.DataFrame(rows), caption="Payment request details")


def clear_admin_payment_filters() -> None:
    """Reset finance tab filters without touching payment data."""

    st.session_state["admin_payment_search"] = ""
    st.session_state["admin_payment_status"] = "All"
    st.session_state["admin_payment_plan"] = "All"
    st.session_state["admin_payment_sort"] = "Newest"


def render_user_payment_requests(user: Dict) -> None:
    """Show the current user's recent manual payment requests."""

    payments = get_payments(search_query=user["email"], limit=20)
    payments = [payment for payment in payments if int(payment["user_id"] or 0) == int(user["id"])]
    st.subheader("Recent Payment Requests")
    if not payments:
        st.info("No manual payment requests yet.")
        return

    status_messages = {
        "pending": "Waiting for admin approval.",
        "paid": "Payment approved. Your plan is active.",
        "rejected": "Payment rejected. Contact admin.",
        "failed": "Payment failed. Contact admin.",
        "refunded": "Payment refunded.",
    }
    for start in range(0, min(len(payments), 10), 2):
        columns = st.columns(2)
        for column, payment in zip(columns, payments[start : start + 2]):
            status = payment["payment_status"]
            with column:
                with st.container(border=True):
                    st.markdown(f"**{payment['payment_ref']}**")
                    st.caption(f"Status: {str(status).title()}")
                    st.write(f"Plan: **{payment['plan_name']}**")
                    st.write(f"Amount: **{float(payment['amount'] or 0):.2f} {payment['currency']}**")
                    st.caption(f"Created: {payment['created_at']}")
                    st.info(status_messages.get(status, ""))


def subscription_page_v2(user: Dict) -> None:
    render_page_header(
        "Subscription",
        "Choose the StockGuard AI plan that fits your stock contributor workflow. Free is great for testing small batches; paid plans add more scans, export options, and review history.",
        "Pricing",
    )
    render_info_card(
        "Choose the right workflow limit",
        "Paid upgrades create a pending manual payment request. Your plan changes only after admin approval.",
        "Manual beta billing",
    )
    render_info_card(
        "Plan guide for stock contributors",
        "Free: try small batches with basic similarity review. Starter: regular scans and CSV reports. Pro: larger batches, Clean ZIP export, and saved scan history. Agency: teams and higher-volume workflows.",
        "Simple guidance",
    )

    with st.container(key="subscription_payment_options"):
        st.markdown(
            """
            <div class="subscription-payment-heading">
                <div class="subscription-payment-icon">$</div>
                <div>
                    <div class="subscription-payment-title">Manual payment</div>
                    <p class="subscription-payment-helper">Choose a payment method and add a short proof note after payment.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        payment_cols = st.columns([1, 1.5])
        with payment_cols[0]:
            with st.container(key="manual_payment_select"):
                payment_method = st.selectbox(
                    "Manual payment method",
                    ["Bank Transfer", "PayHere Manual", "PayPal/Wise Manual"],
                    key="manual_payment_method",
                )
        with payment_cols[1]:
            proof_note = st.text_input(
                "Payment/proof note",
                placeholder="Example: bank slip number, sender name, payment date",
                key="manual_payment_note",
            )

    active_plans = get_active_subscription_plans()
    with st.container(key="subscription_pricing_grid"):
        columns = st.columns(max(len(active_plans), 1))

        for column, plan in zip(columns, active_plans):
            plan_name = plan["plan_name"]
            price = plan_price_label(plan)
            accent = plan_marketing_label(plan)
            is_popular = accent == "Most Popular"
            is_current = is_user_on_plan(user, plan)

            with column:
                with st.container(key=f"pricing_plan_{plan['plan_key']}"):
                    render_plan_card(
                        plan_name=plan_name,
                        plan=plan,
                        price=price,
                        accent=accent,
                        is_popular=is_popular,
                        is_current=is_current,
                    )
                    if is_current:
                        st.button(
                            "Current Plan",
                            disabled=True,
                            key=f"current_v2_{plan_name}",
                            use_container_width=True,
                        )
                    else:
                        is_paid_plan = float(plan["price_usd"] or 0) > 0
                        checkout_url = (plan.get("checkout_url") or "").strip()
                        if is_paid_plan and checkout_url:
                            st.link_button(
                                "Pay with Lemon Squeezy",
                                checkout_url,
                                use_container_width=True,
                            )
                            st.caption("Your plan will activate after successful payment confirmation.")
                        else:
                            if st.button(
                                "Request Payment" if is_paid_plan else f"Switch to {plan_name}",
                                key=f"upgrade_v2_{plan_name}",
                                use_container_width=True,
                            ):
                                if not is_paid_plan:
                                    update_user_plan(user["id"], plan["plan_key"])
                                    st.success(f"Plan changed to {plan_name}.")
                                    st.rerun()
                                payment = create_payment(
                                    user=user,
                                    plan=plan,
                                    payment_method=payment_method,
                                    proof_note=proof_note,
                                    gateway_name="manual",
                                )
                                st.session_state["latest_payment_ref"] = payment["payment_ref"]
                                st.success(f"Pending payment request created: {payment['payment_ref']}")
                                st.rerun()

    latest_payment_ref = st.session_state.get("latest_payment_ref")
    if latest_payment_ref:
        latest_payment = get_payment_by_ref(latest_payment_ref)
        if latest_payment and int(latest_payment["user_id"] or 0) == int(user["id"]):
            with st.container(key="subscription_payment_request"):
                render_payment_request_card(latest_payment)

    render_user_payment_requests(user)


def admin_panel_page(user: Dict) -> None:
    """Polished MVP admin panel controlled by ADMIN_EMAIL."""

    if not is_admin_user(user):
        render_locked_feature_card(
            "Admin access required",
            "This page is only available to the beta admin account configured with ADMIN_EMAIL.",
            "Admin",
        )
        return

    render_page_header(
        "Admin Panel",
        "View users, usage, plans, and manually manage access.",
        "Admin Operations",
    )
    st.markdown(
        """
        <div class="admin-hero">
            <h2>Beta Operations Dashboard</h2>
            <div class="admin-muted">
                Manage user access, simulated subscriptions, and the plan limits that power the product UI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not ADMIN_EMAIL:
        st.warning("ADMIN_EMAIL is not set. Set it before beta launch.")

    users = list_all_users_with_usage()
    active_plans = get_active_subscription_plans()
    all_plans = get_all_subscription_plans(include_inactive=True)
    plan_names = [plan["plan_name"] for plan in active_plans]
    plan_keys = [plan["plan_key"] for plan in active_plans]
    total_scans = sum(int(row["total_scans"] or 0) for row in users)
    disabled_users = sum(1 for row in users if int(row["is_disabled"] or 0))
    paid_users = sum(
        1
        for row in users
        if (get_subscription_plan(row["plan"]) or get_subscription_plan("free"))["plan_key"] != "free"
    )

    metric_cards = [
        ("Total Users", str(len(users)), "All registered accounts.", "👥"),
        ("Total Scans", str(total_scans), "All saved scans.", "📊"),
        ("Active Paid Users", str(paid_users), "Users on paid plans.", "💳"),
        ("Disabled Users", str(disabled_users), "Blocked accounts.", "⛔"),
    ]
    metrics_columns = st.columns(4)
    for column, (label, value, description, icon) in zip(metrics_columns, metric_cards):
        with column:
            render_metric_card(label, value, description, icon)

    st.markdown('<div class="admin-section-title">User Management</div>', unsafe_allow_html=True)
    if not users:
        render_empty_state("Users", "No users yet", "Beta users will appear here after sign up.")
    for row in users:
        current_plan = get_subscription_plan(row["plan"]) or get_subscription_plan("free")
        current_plan_key = current_plan["plan_key"] if current_plan else "free"
        current_plan_name = current_plan["plan_name"] if current_plan else "Free"
        current_index = plan_keys.index(current_plan_key) if current_plan_key in plan_keys else 0
        disabled = bool(int(row["is_disabled"] or 0))
        status_text = "Disabled" if disabled else "Active"
        status_class = "red" if disabled else "green"
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="admin-user-card">
                    <div class="admin-user-top">
                        <div style="min-width:0;">
                            <div class="admin-email">{escape_html(row['email'])}</div>
                            <div class="admin-name">{escape_html(row.get('name') or 'No name')}</div>
                            <div class="admin-muted">Created {escape_html(row['created_at'])}</div>
                        </div>
                        <div class="admin-badges">
                            <span class="admin-pill blue">{escape_html(current_plan_name)}</span>
                            <span class="admin-pill {status_class}">{status_text}</span>
                        </div>
                    </div>
                    <div class="admin-stat-row">
                        <div class="admin-stat">
                            <div class="admin-stat-label">Plan</div>
                            <div class="admin-stat-value">{escape_html(current_plan_name)}</div>
                        </div>
                        <div class="admin-stat">
                            <div class="admin-stat-label">Monthly scans</div>
                            <div class="admin-stat-value">{int(row['monthly_scans'] or 0)}</div>
                        </div>
                        <div class="admin-stat">
                            <div class="admin-stat-label">Total scans</div>
                            <div class="admin-stat-value">{int(row['total_scans'] or 0)}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            control_columns = st.columns([2, 1, 1])
            with control_columns[0]:
                selected_plan = st.selectbox(
                    "Change plan",
                    plan_names,
                    index=current_index,
                    key=f"admin_clean_plan_{row['id']}",
                )
            with control_columns[1]:
                st.write("")
                if st.button("Save Plan", key=f"admin_clean_save_plan_{row['id']}", use_container_width=True):
                    selected_plan_key = plan_keys[plan_names.index(selected_plan)]
                    update_user_plan(row["id"], selected_plan_key)
                    st.success(f"Updated {row['email']} to {selected_plan}.")
                    st.rerun()
            with control_columns[2]:
                st.write("")
                toggle_label = "Enable User" if disabled else "Disable User"
                if st.button(toggle_label, key=f"admin_clean_toggle_user_{row['id']}", use_container_width=True):
                    set_user_disabled(row["id"], not disabled)
                    st.success(f"{'Enabled' if disabled else 'Disabled'} {row['email']}.")
                    st.rerun()

    st.markdown('<div class="admin-section-title">Plan Management</div>', unsafe_allow_html=True)
    st.caption("Plan prices, limits, and feature access are loaded from the SQLite subscription_plans table.")
    plan_columns = st.columns(min(len(all_plans), 4) or 1)
    for index, plan in enumerate(all_plans):
        with plan_columns[index % len(plan_columns)]:
            render_admin_plan_summary_card(plan)
            with st.expander("Edit Plan"):
                with st.form(f"clean_edit_plan_{plan['plan_key']}"):
                    name_col, price_col = st.columns(2)
                    with name_col:
                        plan_name_value = st.text_input("Plan display name", value=plan["plan_name"])
                        billing_value = st.text_input("Billing label", value=plan["billing_label"] or "month")
                    with price_col:
                        price_value = st.number_input(
                            "Price USD",
                            min_value=0.0,
                            value=float(plan["price_usd"]),
                            step=1.0,
                            key=f"clean_price_{plan['plan_key']}",
                        )
                        active_enabled = st.checkbox("Active status", value=bool(plan["is_active"]), key=f"clean_active_{plan['plan_key']}")

                    limit_col, image_col = st.columns(2)
                    with limit_col:
                        monthly_limit_value = st.number_input(
                            "Monthly scan limit",
                            min_value=0,
                            value=int(plan["monthly_scans"]),
                            step=1,
                            key=f"clean_monthly_limit_{plan['plan_key']}",
                        )
                    with image_col:
                        image_limit_value = st.number_input(
                            "Images per scan limit",
                            min_value=1,
                            value=int(plan["images_per_scan"]),
                            step=1,
                            key=f"clean_image_limit_{plan['plan_key']}",
                        )

                    feature_col_1, feature_col_2 = st.columns(2)
                    with feature_col_1:
                        csv_enabled = st.checkbox("CSV export enabled", value=bool(plan["csv_export"]), key=f"clean_csv_{plan['plan_key']}")
                        zip_enabled = st.checkbox("ZIP export enabled", value=bool(plan["zip_export"]), key=f"clean_zip_{plan['plan_key']}")
                        readiness_enabled = st.checkbox("Readiness Report enabled", value=bool(plan["readiness_report"]), key=f"clean_readiness_{plan['plan_key']}")
                        best_shot_enabled = st.checkbox("Best Shot enabled", value=bool(plan["best_shot"]), key=f"clean_best_{plan['plan_key']}")
                    with feature_col_2:
                        history_enabled = st.checkbox("Scan History enabled", value=bool(plan["batch_history"]), key=f"clean_history_{plan['plan_key']}")
                        project_enabled = st.checkbox("Project folders enabled", value=bool(plan["project_folders"]), key=f"clean_project_{plan['plan_key']}")
                        client_enabled = st.checkbox("Client folders enabled", value=bool(plan["client_folders"]), key=f"clean_client_{plan['plan_key']}")

                    summary_value = st.text_area(
                        "Feature summary",
                        value=plan.get("feature_summary") or "",
                        key=f"clean_summary_{plan['plan_key']}",
                    )
                    if st.form_submit_button("Save Plan", use_container_width=True):
                        update_subscription_plan(
                            plan["plan_key"],
                            {
                                "plan_name": plan_name_value.strip() or plan["plan_name"],
                                "price_usd": price_value,
                                "billing_label": billing_value.strip() or "month",
                                "monthly_scan_limit": int(monthly_limit_value),
                                "images_per_scan_limit": int(image_limit_value),
                                "csv_export_enabled": int(csv_enabled),
                                "zip_export_enabled": int(zip_enabled),
                                "readiness_report_enabled": int(readiness_enabled),
                                "best_shot_enabled": int(best_shot_enabled),
                                "scan_history_enabled": int(history_enabled),
                                "project_folders_enabled": int(project_enabled),
                                "client_folders_enabled": int(client_enabled),
                                "feature_summary": summary_value,
                                "is_active": int(active_enabled),
                            },
                        )
                        st.success(f"Saved {plan_name_value} plan.")
                        st.rerun()

    st.markdown('<div class="admin-section-title">Maintenance</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="admin-maintenance-card">
            <div class="admin-plan-name">Reset Default Plans</div>
            <div class="admin-muted">This will reset plan prices, limits, and feature flags.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Reset subscription plans to default seed values", key="clean_reset_subscription_plans"):
        reset_default_subscription_plans()
        st.success("Subscription plans reset to database seed defaults.")
        st.rerun()


def render_admin_user_manage_panel(row: Dict, active_plans: List[Dict]) -> None:
    """Compact manage panel for one admin user row."""

    plan_names = [plan["plan_name"] for plan in active_plans]
    plan_keys = [plan["plan_key"] for plan in active_plans]
    current_plan = get_subscription_plan(row["plan"]) or get_subscription_plan("free")
    current_plan_key = current_plan["plan_key"] if current_plan else "free"
    current_plan_name = current_plan["plan_name"] if current_plan else row["plan"]
    current_index = plan_keys.index(current_plan_key) if current_plan_key in plan_keys else 0
    disabled = bool(int(row["is_disabled"] or 0))

    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.markdown(f"**User ID:** `{row['public_user_id']}`")
        st.caption(f"Name: {row.get('name') or 'No name'}")
    with detail_cols[1]:
        st.markdown(f"**Email:** {row['email']}")
        st.caption(f"Joined: {row['created_at']}")
    with detail_cols[2]:
        st.markdown(f"**Current plan:** {current_plan_name}")
        st.caption("Status: Disabled" if disabled else "Status: Active")

    usage_cols = st.columns(4)
    usage_cols[0].metric("Monthly scans", f"{int(row['monthly_scans'] or 0)} / {current_plan['monthly_scans']}")
    usage_cols[1].metric("Image limit", current_plan["images_per_scan"])
    usage_cols[2].metric("Total scans", int(row["total_scans"] or 0))
    usage_cols[3].metric("Account", "Disabled" if disabled else "Active")

    action_cols = st.columns([2, 1, 1, 1])
    with action_cols[0]:
        selected_plan = st.selectbox(
            "Change plan",
            plan_names,
            index=current_index,
            key=f"admin_tab_plan_{row['id']}",
        )
    with action_cols[1]:
        st.write("")
        if st.button("Save Plan", key=f"admin_tab_save_{row['id']}", use_container_width=True):
            update_user_plan(row["id"], plan_keys[plan_names.index(selected_plan)])
            st.success(f"Updated {row['public_user_id']} to {selected_plan}.")
            st.rerun()
    with action_cols[2]:
        st.write("")
        if st.button("Reset Usage", key=f"admin_tab_reset_usage_{row['id']}", use_container_width=True):
            reset_user_monthly_usage(row["id"])
            st.success(f"Monthly usage reset for {row['public_user_id']}.")
            st.rerun()
    with action_cols[3]:
        st.write("")
        toggle_label = "Enable" if disabled else "Disable"
        if st.button(toggle_label, key=f"admin_tab_toggle_{row['id']}", use_container_width=True):
            set_user_enabled(row["id"], disabled)
            st.success(f"{'Enabled' if disabled else 'Disabled'} {row['public_user_id']}.")
            st.rerun()


def render_admin_plans_tab(all_plans: List[Dict]) -> None:
    st.markdown('<div class="admin-section-title">Plan Management</div>', unsafe_allow_html=True)
    st.caption("Plans are loaded from SQLite `subscription_plans`. Changes update pricing, limits, and feature locks across the app.")

    plan_columns = st.columns(min(len(all_plans), 4) or 1)
    for index, plan in enumerate(all_plans):
        with plan_columns[index % len(plan_columns)]:
            render_admin_plan_summary_card(plan)
            with st.expander("Edit Plan"):
                with st.form(f"tabs_edit_plan_{plan['plan_key']}"):
                    name_col, price_col = st.columns(2)
                    with name_col:
                        plan_name_value = st.text_input("Plan display name", value=plan["plan_name"])
                        billing_value = st.text_input("Billing label", value=plan["billing_label"] or "month")
                    with price_col:
                        price_value = st.number_input(
                            "Price USD",
                            min_value=0.0,
                            value=float(plan["price_usd"]),
                            step=1.0,
                            key=f"tabs_price_{plan['plan_key']}",
                        )
                        active_enabled = st.checkbox("Active status", value=bool(plan["is_active"]), key=f"tabs_active_{plan['plan_key']}")

                    limit_col, image_col = st.columns(2)
                    with limit_col:
                        monthly_limit_value = st.number_input(
                            "Monthly scan limit",
                            min_value=0,
                            value=int(plan["monthly_scans"]),
                            step=1,
                            key=f"tabs_monthly_limit_{plan['plan_key']}",
                        )
                    with image_col:
                        image_limit_value = st.number_input(
                            "Images per scan limit",
                            min_value=1,
                            value=int(plan["images_per_scan"]),
                            step=1,
                            key=f"tabs_image_limit_{plan['plan_key']}",
                        )

                    feature_col_1, feature_col_2 = st.columns(2)
                    with feature_col_1:
                        csv_enabled = st.checkbox("CSV export enabled", value=bool(plan["csv_export"]), key=f"tabs_csv_{plan['plan_key']}")
                        zip_enabled = st.checkbox("ZIP export enabled", value=bool(plan["zip_export"]), key=f"tabs_zip_{plan['plan_key']}")
                        readiness_enabled = st.checkbox("Readiness Report enabled", value=bool(plan["readiness_report"]), key=f"tabs_readiness_{plan['plan_key']}")
                        best_shot_enabled = st.checkbox("Best Shot enabled", value=bool(plan["best_shot"]), key=f"tabs_best_{plan['plan_key']}")
                        metadata_enabled = st.checkbox(
                            "Metadata Similarity Checker enabled",
                            value=bool(plan.get("metadata_checker", plan.get("metadata_checker_enabled", 0))),
                            key=f"tabs_metadata_{plan['plan_key']}",
                        )
                    with feature_col_2:
                        history_enabled = st.checkbox("Scan History enabled", value=bool(plan["batch_history"]), key=f"tabs_history_{plan['plan_key']}")
                        project_enabled = st.checkbox("Project folders enabled", value=bool(plan["project_folders"]), key=f"tabs_project_{plan['plan_key']}")
                        client_enabled = st.checkbox("Client folders enabled", value=bool(plan["client_folders"]), key=f"tabs_client_{plan['plan_key']}")
                        advanced_modes_enabled = st.checkbox(
                            "Advanced Scan Modes enabled",
                            value=bool(plan.get("advanced_scan_modes", plan.get("advanced_scan_modes_enabled", 0))),
                            key=f"tabs_advanced_modes_{plan['plan_key']}",
                        )

                    summary_value = st.text_area(
                        "Feature summary",
                        value=plan.get("feature_summary") or "",
                        key=f"tabs_summary_{plan['plan_key']}",
                    )
                    st.caption("Lemon Squeezy setup is optional for beta. If checkout URL is empty, the app keeps using manual payment requests.")
                    lemon_col, checkout_col = st.columns(2)
                    with lemon_col:
                        lemon_variant_value = st.text_input(
                            "Lemon Squeezy variant ID",
                            value=plan.get("lemon_variant_id") or "",
                            placeholder="Example: 123456",
                            key=f"tabs_lemon_variant_{plan['plan_key']}",
                        )
                    with checkout_col:
                        checkout_url_value = st.text_input(
                            "Fallback checkout URL",
                            value=plan.get("checkout_url") or "",
                            placeholder="https://your-store.lemonsqueezy.com/checkout/...",
                            key=f"tabs_checkout_url_{plan['plan_key']}",
                        )
                    if st.form_submit_button("Save Plan", use_container_width=True):
                        update_subscription_plan(
                            plan["plan_key"],
                            {
                                "plan_name": plan_name_value.strip() or plan["plan_name"],
                                "price_usd": price_value,
                                "billing_label": billing_value.strip() or "month",
                                "monthly_scan_limit": int(monthly_limit_value),
                                "images_per_scan_limit": int(image_limit_value),
                                "csv_export_enabled": int(csv_enabled),
                                "zip_export_enabled": int(zip_enabled),
                                "readiness_report_enabled": int(readiness_enabled),
                                "best_shot_enabled": int(best_shot_enabled),
                                "metadata_checker_enabled": int(metadata_enabled),
                                "advanced_scan_modes_enabled": int(advanced_modes_enabled),
                                "scan_history_enabled": int(history_enabled),
                                "project_folders_enabled": int(project_enabled),
                                "client_folders_enabled": int(client_enabled),
                                "feature_summary": summary_value,
                                "lemon_variant_id": lemon_variant_value.strip(),
                                "checkout_url": checkout_url_value.strip(),
                                "is_active": int(active_enabled),
                            },
                        )
                        st.success(f"Saved {plan_name_value} plan.")
                        st.rerun()

    st.markdown('<div class="admin-section-title">Maintenance</div>', unsafe_allow_html=True)
    st.warning("This will reset plan prices, limits, and feature flags.")
    if st.button("Reset default plans", key="tabs_reset_subscription_plans"):
        reset_default_subscription_plans()
        st.success("Subscription plans reset to database seed defaults.")
        st.rerun()


def render_admin_finance_tab(all_plans: List[Dict], admin_user: Dict) -> None:
    """Finance dashboard for beta manual payments."""

    if not is_admin_user(admin_user):
        st.error("Access denied. Admin permissions required.")
        st.stop()

    st.caption("Admin Finance")
    st.subheader("Financial Dashboard")
    st.caption("Track revenue, pending payments, plan upgrades, and manual payment approvals.")

    summary = get_financial_summary()
    st.markdown("### Revenue Summary")
    render_finance_summary_cards(summary)

    st.markdown("### Plan Revenue Breakdown")
    revenue_lookup = {row["plan_key"]: row for row in get_plan_revenue_breakdown()}
    plan_rows = []
    for plan in all_plans:
        revenue_row = revenue_lookup.get(plan["plan_key"], {})
        merged_plan = dict(plan)
        merged_plan["paid_users"] = int(revenue_row.get("paid_users") or 0)
        merged_plan["revenue"] = float(revenue_row.get("revenue") or 0)
        plan_rows.append(merged_plan)
    render_plan_revenue_cards(plan_rows)

    with st.expander("Monthly revenue details", expanded=False):
        monthly_rows = [
            {
                "Month": row["month"],
                "Revenue": f"${float(row['revenue'] or 0):.2f}",
                "Payments": int(row["payments"] or 0),
            }
            for row in get_monthly_revenue_summary()
        ]
        if monthly_rows:
            render_light_html_table(pd.DataFrame(monthly_rows), caption="Monthly revenue details")
        else:
            render_empty_state("Revenue", "No revenue yet", "Approved payment revenue will appear here.")

    st.markdown("### Payment Filters")
    st.markdown("<div class='sg-filter-card'>", unsafe_allow_html=True)
    plan_filter_options = {"All": "All"}
    plan_filter_options.update(
        {
            plan["plan_name"]: plan["plan_key"]
            for plan in all_plans
            if bool(plan.get("is_active", 1))
        }
    )
    filter_cols = st.columns([1.6, 1, 1, 1, 0.75])
    with filter_cols[0]:
        payment_search = st.text_input(
            "Search payments",
            placeholder="Search by payment ref, user ID, or email",
            key="admin_payment_search",
        )
    with filter_cols[1]:
        payment_status = st.selectbox(
            "Status",
            ["All", "pending", "paid", "rejected", "failed", "refunded"],
            key="admin_payment_status",
        )
    with filter_cols[2]:
        payment_plan_label = st.selectbox(
            "Plan",
            list(plan_filter_options.keys()),
            key="admin_payment_plan",
        )
    with filter_cols[3]:
        payment_sort = st.selectbox(
            "Sort",
            ["Newest", "Oldest", "Amount high to low", "Amount low to high"],
            key="admin_payment_sort",
        )
    with filter_cols[4]:
        st.write("")
        st.button(
            "Clear filters",
            key="admin_payment_clear_filters",
            use_container_width=True,
            on_click=clear_admin_payment_filters,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    payments = get_payments(
        status_filter=payment_status,
        search_query=payment_search,
        plan_filter=plan_filter_options[payment_plan_label],
        sort_by=payment_sort,
        limit=300,
    )
    all_payment_count = len(get_payments(limit=1))
    pending_payments = [payment for payment in payments if payment["payment_status"] == "pending"]

    st.markdown("### Payments")
    st.caption(f"Showing {len(payments)} payment request(s). Results are capped for performance.")
    if not payments:
        if all_payment_count == 0:
            render_empty_state("Payments", "No payments yet", "Payment requests will appear here after users choose a paid plan.")
        else:
            render_empty_state("Search", "No matching payments", "Try changing the search or filters.")
    else:
        render_payments_table(payments)

    st.markdown("### Pending Payment Actions")
    if not pending_payments and payments:
        render_empty_state("Pending", "No pending payments", "All payment requests in this view have been reviewed.")
    elif not payments:
        st.caption("No payment actions to review.")

    for payment in payments:
        ref = payment["payment_ref"]
        expander_label = (
            f"{ref} - {payment.get('public_user_id') or payment['user_id']} - "
            f"{payment['user_email']} - {payment['plan_name']} - {payment['payment_status']}"
        )
        with st.expander(expander_label, expanded=payment["payment_status"] == "pending"):
            detail_cols = st.columns([1.2, 0.8])
            with detail_cols[0]:
                with st.container(border=True):
                    st.markdown("**Payment details**")
                    st.write(f"Payment reference: **{ref}**")
                    st.write(f"Public user ID: **{payment.get('public_user_id') or payment['user_id']}**")
                    st.write(f"Email: **{payment['user_email']}**")
                    st.write(f"Selected plan: **{payment['plan_name']}**")
                    st.write(f"Amount: **{float(payment['amount'] or 0):.2f} {payment['currency']}**")
                    st.write(f"Method: **{payment.get('payment_method') or 'Manual'}**")
                    st.write(f"Gateway: **{payment.get('gateway_name') or 'manual'}**")
                    st.write(f"Status: **{str(payment['payment_status']).title()}**")
                    st.caption(f"Created date: {payment['created_at']}")
                    st.caption(f"Paid at: {payment.get('paid_at') or 'Not paid yet'}")
            with detail_cols[1]:
                with st.container(border=True):
                    st.markdown("**Notes**")
                    st.caption("Proof note")
                    st.write(payment.get("proof_note") or "No proof note submitted.")
                    st.caption("Admin note")
                    st.write(payment.get("admin_note") or "No admin note yet.")

            if payment["payment_status"] == "pending":
                st.markdown("**Admin actions**")
                action_left, action_right = st.columns([1.3, 0.9])
                with action_left:
                    admin_note = st.text_area(
                        "Admin note",
                        placeholder="Optional approval/rejection note",
                        key=f"admin_payment_note_{ref}",
                        height=90,
                    )
                with action_right:
                    st.write("")
                    if st.button(
                        "Approve Payment",
                        key=f"approve_payment_{ref}",
                        use_container_width=True,
                        type="primary",
                    ):
                        if not is_admin_user(admin_user):
                            st.error("Access denied. Admin permissions required.")
                            st.stop()
                        approve_manual_payment(ref, admin_user.get("email", "admin"))
                        st.success("Payment approved and user upgraded.")
                        st.rerun()
                    st.write("")
                    if st.button(
                        "Reject Payment",
                        key=f"reject_payment_{ref}",
                        use_container_width=True,
                    ):
                        if not is_admin_user(admin_user):
                            st.error("Access denied. Admin permissions required.")
                            st.stop()
                        reject_manual_payment(ref, admin_note or "Rejected by admin")
                        st.warning(f"Rejected {ref}.")
                        st.rerun()
            else:
                st.caption("Only pending payment requests can be approved or rejected.")

    st.info(
        "Manual payment and gateway-ready notes: manual payments stay pending until an admin approves "
        "or rejects them. Lemon Squeezy-ready records can use gateway_name and gateway_payment_id later "
        "from a separate webhook service. Do not store card data or secrets in Streamlit."
    )


def admin_panel_page(user: Dict) -> None:
    """Scalable admin panel for many users."""

    if not is_admin_user(user):
        render_locked_feature_card(
            "Admin access required",
            "This page is only available to the beta admin account configured with ADMIN_EMAIL.",
            "Admin",
        )
        return

    render_page_header(
        "Admin Panel",
        "Search users, manage plans, review scan logs, and control beta settings.",
        "Admin Operations",
    )
    if not ADMIN_EMAIL:
        st.warning("ADMIN_EMAIL is not set. Set it before beta launch.")

    active_plans = get_active_subscription_plans()
    all_plans = get_all_subscription_plans(include_inactive=True)
    plan_filter_options = {"All": "All"}
    plan_filter_options.update({plan["plan_name"]: plan["plan_key"] for plan in all_plans})

    overview_tab, users_tab, plans_tab, finance_tab, scan_logs_tab, settings_tab = st.tabs(
        ["Overview", "Users", "Plans", "Finance", "Scan Logs", "Settings"],
        key="admin_panel_tabs",
    )

    with overview_tab:
        stats = get_admin_overview_stats()
        metric_items = [
            ("Total Users", str(stats["total_users"]), "All registered accounts.", "👥"),
            ("Active Users", str(stats["active_users"]), "Accounts not disabled.", "✓"),
            ("Paid Users", str(stats["paid_users"]), "Users on paid plans.", "💳"),
            ("Disabled Users", str(stats["disabled_users"]), "Blocked accounts.", "⛔"),
            ("Total Scans", str(stats["total_scans"]), "All saved scans.", "📊"),
            ("Scans This Month", str(stats["scans_this_month"]), "Current calendar month.", "📅"),
            ("Most Used Plan", str(stats["most_used_plan"]), "Largest user segment.", "🧭"),
            ("Images Processed", str(stats["total_images_processed"]), "Total images from saved scans.", "🖼"),
        ]
        render_metric_cards(metric_items)

        overview_cols = st.columns(3)
        with overview_cols[0]:
            render_section_card("Recent Users", "Latest registered accounts in the current beta workspace.")
            render_light_table(pd.DataFrame(stats["recent_users"]).to_dict("records"), list(pd.DataFrame(stats["recent_users"]).columns))
        with overview_cols[1]:
            render_section_card("Recent Scans", "A readable summary of recent saved batch activity.")
            render_light_table(pd.DataFrame(stats["recent_scans"]).to_dict("records"), list(pd.DataFrame(stats["recent_scans"]).columns))
        with overview_cols[2]:
            render_section_card("Plan Distribution", "Current plan mix across active users.")
            render_light_table(pd.DataFrame(stats["plan_distribution"]).to_dict("records"), list(pd.DataFrame(stats["plan_distribution"]).columns))

    with users_tab:
        st.markdown('<div class="admin-section-title">User Management</div>', unsafe_allow_html=True)
        st.markdown("<div class='sg-filter-card'>", unsafe_allow_html=True)
        filter_cols = st.columns([2.2, 1, 1, 1])
        with filter_cols[0]:
            search_query = st.text_input(
                "Search users",
                placeholder="Search by User ID, email, or name",
                key="admin_user_search",
            )
        with filter_cols[1]:
            selected_plan_label = st.selectbox("Plan", list(plan_filter_options.keys()), key="admin_user_plan_filter")
        with filter_cols[2]:
            status_filter = st.selectbox("Status", ["All", "Active", "Disabled"], key="admin_user_status_filter")
        with filter_cols[3]:
            sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Most scans", "Plan"], key="admin_user_sort")

        st.markdown("</div>", unsafe_allow_html=True)

        users = get_users_for_admin(
            search_query=search_query,
            plan_filter=plan_filter_options[selected_plan_label],
            status_filter=status_filter,
            sort_by=sort_by,
        )
        user_table_rows = []
        for row in users:
            plan = get_subscription_plan(row["plan"]) or get_subscription_plan("free")
            user_table_rows.append(
                {
                    "User ID": row["public_user_id"],
                    "Email": row["email"],
                    "Name": row["name"],
                    "Plan": plan["plan_name"],
                    "Usage": f"{int(row['monthly_scans'] or 0)} / {plan['monthly_scans']}",
                    "Total Scans": int(row["total_scans"] or 0),
                    "Status": "Disabled" if int(row["is_disabled"] or 0) else "Active",
                    "Joined": row["created_at"],
                    "Action": "Open expander below",
                }
            )
        st.caption(f"Showing {len(users)} user(s). Results are capped for performance.")
        render_light_html_table(pd.DataFrame(user_table_rows), caption="User management table")

        for row in users:
            with st.expander(f"Manage {row['public_user_id']} - {row['email']}"):
                render_admin_user_manage_panel(row, active_plans)

    with plans_tab:
        render_admin_plans_tab(all_plans)

    with scan_logs_tab:
        st.markdown('<div class="admin-section-title">Scan Logs</div>', unsafe_allow_html=True)
        st.markdown("<div class='sg-filter-card'>", unsafe_allow_html=True)
        scan_filter_cols = st.columns([2.2, 1, 1])
        with scan_filter_cols[0]:
            scan_search = st.text_input(
                "Search scans",
                placeholder="Search by User ID, email, batch name, or project",
                key="admin_scan_search",
            )
        with scan_filter_cols[1]:
            scan_plan_label = st.selectbox("Plan filter", list(plan_filter_options.keys()), key="admin_scan_plan_filter")
        with scan_filter_cols[2]:
            scan_sort = st.selectbox("Sort scans", ["Newest", "Highest risky pairs", "Highest similarity"], key="admin_scan_sort")

        st.markdown("</div>", unsafe_allow_html=True)

        logs = get_scan_logs_for_admin(
            search_query=scan_search,
            plan_filter=plan_filter_options[scan_plan_label],
            sort_by=scan_sort,
        )
        log_rows = [
            {
                "Scan ID": row["scan_id"],
                "User ID": row["public_user_id"],
                "Email": row["email"],
                "Project": row.get("project_name") or "No Project",
                "Batch": row["batch_name"],
                "Date": row["scan_datetime"],
                "Images": row["total_images"],
                "Risky": row["risky_pairs_count"],
                "Near Duplicates": row["near_duplicate_count"],
                "Highest": f"{row['highest_similarity_score']:.2f}%",
                "Plan": row["plan_name"],
            }
            for row in logs
        ]
        st.caption(f"Showing {len(log_rows)} scan log(s).")
        render_light_html_table(pd.DataFrame(log_rows), caption="Scan log table")

    with finance_tab:
        render_admin_finance_tab(all_plans, user)

    with settings_tab:
        st.markdown(
            """
            <div class="sg-card" style="padding: 0.9rem 1rem; margin-bottom: 0.9rem;">
                <div class="sg-card-title" style="margin-bottom: 0.15rem;">Admin Settings</div>
                <div class="sg-muted">Read-only environment and app settings for the current beta deployment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        settings_items = [
            ("Admin email", ADMIN_EMAIL or "Not set"),
            ("App mode", "Beta"),
            ("Feedback contact", FEEDBACK_EMAIL),
            ("File cleanup", "Uploaded image files are processed temporarily in-memory and are not stored permanently."),
            ("Privacy notice", "Images are processed only for similarity analysis and are not used for AI training."),
        ]

        settings_cols = st.columns(2, gap="medium")
        for index, (label, value) in enumerate(settings_items):
            with settings_cols[index % 2]:
                st.markdown(
                    f"""
                    <div class="sg-card" style="padding: 1rem; height: 100%; border-radius: 18px; background: #FFFFFF; border: 1px solid #E2E8F0; box-shadow: 0 18px 30px rgba(15, 23, 42, 0.06);">
                        <div class="sg-card-title" style="font-size: 1rem; margin-bottom: 0.25rem;">{escape_html(label)}</div>
                        <div class="sg-muted" style="white-space: pre-wrap; line-height: 1.45;">{escape_html(value)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("These values are stored in the app environment and the SQLite-backed plan metadata.")


def main() -> None:
    st.set_page_config(
        page_title=PRODUCT_NAME,
        page_icon="logo.png",
        layout="wide",
        initial_sidebar_state="auto",
    )
    init_db()
    apply_custom_styles()

    query_params = st.query_params
    if "nav" in query_params and query_params["nav"] == "profile":
        st.session_state["page"] = "My Profile"
        del query_params["nav"]
        st.rerun()

    user = get_logged_in_user()
    if not user:
        render_auth_page()
        return

    page = render_sidebar(user)
    st.session_state["page"] = page
    render_topbar(user, page)

    if page == "Dashboard":
        dashboard_page(user)
    elif page == "New Scan":
        new_scan_page(user)
    elif page == "Best Shot Selector":
        feature_placeholder_page(
            user,
            "Best Shot Selector",
            "Choose the strongest image in each similar group after a scan.",
            [("Start New Scan", "New Scan"), ("Open Scan History", "Scan History")],
        )
    elif page == "Readiness Report":
        readiness_report_page(user)
    elif page == "Upload Readiness Report":
        readiness_report_page(user)
    elif page == "Auto Scan Summary":
        feature_placeholder_page(
            user,
            "Auto Scan Summary",
            "See automatic Ready, Review, and Remove recommendations after scanning.",
            [("Start New Scan", "New Scan"), ("Open Latest Report", "Upload Readiness Report")],
        )
    elif page == "CSV Reports":
        feature_placeholder_page(
            user,
            "CSV Reports",
            "Export similarity, readiness, and metadata CSV reports from completed scans.",
            [("Start New Scan", "New Scan"), ("Open Scan History", "Scan History")],
        )
    elif page == "My Exports":
        feature_placeholder_page(
            user,
            "My Exports",
            "Download Clean ZIP and CSV exports after reviewing a scan.",
            [("Start New Scan", "New Scan"), ("Open Scan History", "Scan History")],
        )
    elif page == "Scan Profiles":
        scan_profiles_page(user)
    elif page == "Scan History":
        scan_history_page(user)
    elif page == "Projects":
        projects_page(user)
    elif page == "Privacy Policy":
        privacy_policy_page(user)
    elif page == "Subscription":
        subscription_page_v2(user)
    elif page == "Billing History":
        billing_history_page(user)
    elif page == "My Profile":
        my_profile_page(user)
    elif page == "API Access":
        api_access_page(user)
    elif page == "Settings":
        settings_page(user)
    elif page == "Admin Panel":
        if not is_admin_user(user):
            st.error("Access denied. Admin permissions required.")
            st.session_state["page"] = "Dashboard"
            st.stop()
        admin_panel_page(user)
    elif page == "Logout":
        logout_page()

    if page not in {"New Scan", "Logout"}:
        render_app_footer()


if __name__ == "__main__":
    main()


