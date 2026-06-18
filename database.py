import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import bcrypt
try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional until installed.
    load_dotenv = None


if load_dotenv:
    load_dotenv()


database_path_value = os.getenv("DATABASE_PATH", "adobe_saas.db").strip() or "adobe_saas.db"
DATABASE_PATH = Path(database_path_value)
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = Path(__file__).resolve().parent / DATABASE_PATH
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()

DEFAULT_SUBSCRIPTION_PLANS = [
    {
        "plan_key": "free",
        "plan_name": "Free",
        "price_usd": 0,
        "billing_label": "month",
        "monthly_scan_limit": 3,
        "images_per_scan_limit": 20,
        "csv_export_enabled": 1,
        "zip_export_enabled": 0,
        "readiness_report_enabled": 0,
        "best_shot_enabled": 0,
        "scan_history_enabled": 0,
        "project_folders_enabled": 1,
        "client_folders_enabled": 0,
        "metadata_checker_enabled": 0,
        "advanced_scan_modes_enabled": 0,
        "feature_summary": "Basic similarity scan with limited usage.",
        "is_active": 1,
        "sort_order": 1,
    },
    {
        "plan_key": "starter",
        "plan_name": "Starter",
        "price_usd": 5,
        "billing_label": "month",
        "monthly_scan_limit": 30,
        "images_per_scan_limit": 100,
        "csv_export_enabled": 1,
        "zip_export_enabled": 1,
        "readiness_report_enabled": 1,
        "best_shot_enabled": 1,
        "scan_history_enabled": 0,
        "project_folders_enabled": 1,
        "client_folders_enabled": 0,
        "metadata_checker_enabled": 0,
        "advanced_scan_modes_enabled": 1,
        "feature_summary": "Great for regular contributors who need readiness reports and clean ZIP export.",
        "is_active": 1,
        "sort_order": 2,
    },
    {
        "plan_key": "pro",
        "plan_name": "Pro",
        "price_usd": 12,
        "billing_label": "month",
        "monthly_scan_limit": 150,
        "images_per_scan_limit": 300,
        "csv_export_enabled": 1,
        "zip_export_enabled": 1,
        "readiness_report_enabled": 1,
        "best_shot_enabled": 1,
        "scan_history_enabled": 1,
        "project_folders_enabled": 1,
        "client_folders_enabled": 0,
        "metadata_checker_enabled": 1,
        "advanced_scan_modes_enabled": 1,
        "feature_summary": "Best for serious stock contributors who need full workflow and history.",
        "is_active": 1,
        "sort_order": 3,
    },
    {
        "plan_key": "agency",
        "plan_name": "Agency",
        "price_usd": 29,
        "billing_label": "month",
        "monthly_scan_limit": 500,
        "images_per_scan_limit": 500,
        "csv_export_enabled": 1,
        "zip_export_enabled": 1,
        "readiness_report_enabled": 1,
        "best_shot_enabled": 1,
        "scan_history_enabled": 1,
        "project_folders_enabled": 1,
        "client_folders_enabled": 1,
        "metadata_checker_enabled": 1,
        "advanced_scan_modes_enabled": 1,
        "feature_summary": "Built for agencies and contributors managing multiple clients.",
        "is_active": 1,
        "sort_order": 4,
    },
]


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    """Return True when a table column already exists."""

    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
    return column_name in columns


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'Free',
                created_at TEXT NOT NULL
            )
            """
        )
        if not column_exists(connection, "users", "is_disabled"):
            connection.execute("ALTER TABLE users ADD COLUMN is_disabled INTEGER NOT NULL DEFAULT 0")
        if not column_exists(connection, "users", "public_user_id"):
            connection.execute("ALTER TABLE users ADD COLUMN public_user_id TEXT")
        if not column_exists(connection, "users", "role"):
            connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if not column_exists(connection, "users", "display_name"):
            connection.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
        if not column_exists(connection, "users", "profile_photo_path"):
            connection.execute("ALTER TABLE users ADD COLUMN profile_photo_path TEXT")
        if ADMIN_EMAIL:
            connection.execute(
                "UPDATE users SET role = 'admin' WHERE lower(email) = ?",
                (ADMIN_EMAIL,),
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, name),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                project_id INTEGER,
                batch_name TEXT NOT NULL,
                scan_datetime TEXT NOT NULL,
                total_images INTEGER NOT NULL,
                risky_pairs_count INTEGER NOT NULL,
                near_duplicate_count INTEGER NOT NULL,
                highest_similarity_score REAL NOT NULL,
                csv_report_json TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(project_id) REFERENCES projects(id)
            )
            """
        )
        init_subscription_plans(connection)
        init_usage_resets(connection)
        ensure_user_public_ids(connection)
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_public_user_id ON users(public_user_id)"
        )
        init_payments(connection)


def plan_key_from_value(plan_value: str) -> str:
    value = (plan_value or "free").strip().lower()
    name_to_key = {plan["plan_name"].lower(): plan["plan_key"] for plan in DEFAULT_SUBSCRIPTION_PLANS}
    key_values = {plan["plan_key"] for plan in DEFAULT_SUBSCRIPTION_PLANS}
    if value in key_values:
        return value
    return name_to_key.get(value, "free")


def add_column_if_missing(connection: sqlite3.Connection, table_name: str, column_name: str, ddl: str) -> bool:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")
        return True
    return False


def init_subscription_plans(connection: sqlite3.Connection | None = None) -> None:
    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    assert connection is not None
    now = datetime.now().isoformat(timespec="seconds")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS subscription_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_key TEXT UNIQUE NOT NULL,
            plan_name TEXT NOT NULL,
            price_usd REAL NOT NULL DEFAULT 0,
            billing_label TEXT DEFAULT 'month',
            monthly_scan_limit INTEGER NOT NULL,
            images_per_scan_limit INTEGER NOT NULL,
            csv_export_enabled INTEGER DEFAULT 1,
            zip_export_enabled INTEGER DEFAULT 0,
            readiness_report_enabled INTEGER DEFAULT 0,
            best_shot_enabled INTEGER DEFAULT 0,
            scan_history_enabled INTEGER DEFAULT 0,
            project_folders_enabled INTEGER DEFAULT 1,
            client_folders_enabled INTEGER DEFAULT 0,
            metadata_checker_enabled INTEGER DEFAULT 0,
            advanced_scan_modes_enabled INTEGER DEFAULT 0,
            feature_summary TEXT,
            lemon_variant_id TEXT,
            checkout_url TEXT,
            is_active INTEGER DEFAULT 1,
            sort_order INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    safe_columns = {
        "price_usd": "price_usd REAL NOT NULL DEFAULT 0",
        "billing_label": "billing_label TEXT DEFAULT 'month'",
        "monthly_scan_limit": "monthly_scan_limit INTEGER NOT NULL DEFAULT 3",
        "images_per_scan_limit": "images_per_scan_limit INTEGER NOT NULL DEFAULT 20",
        "csv_export_enabled": "csv_export_enabled INTEGER DEFAULT 1",
        "zip_export_enabled": "zip_export_enabled INTEGER DEFAULT 0",
        "readiness_report_enabled": "readiness_report_enabled INTEGER DEFAULT 0",
        "best_shot_enabled": "best_shot_enabled INTEGER DEFAULT 0",
        "scan_history_enabled": "scan_history_enabled INTEGER DEFAULT 0",
        "project_folders_enabled": "project_folders_enabled INTEGER DEFAULT 1",
        "client_folders_enabled": "client_folders_enabled INTEGER DEFAULT 0",
        "metadata_checker_enabled": "metadata_checker_enabled INTEGER DEFAULT 0",
        "advanced_scan_modes_enabled": "advanced_scan_modes_enabled INTEGER DEFAULT 0",
        "feature_summary": "feature_summary TEXT",
        "lemon_variant_id": "lemon_variant_id TEXT",
        "checkout_url": "checkout_url TEXT",
        "is_active": "is_active INTEGER DEFAULT 1",
        "sort_order": "sort_order INTEGER DEFAULT 0",
        "created_at": "created_at TEXT",
        "updated_at": "updated_at TEXT",
    }
    added_columns = set()
    for column_name, ddl in safe_columns.items():
        if add_column_if_missing(connection, "subscription_plans", column_name, ddl):
            added_columns.add(column_name)

    if "metadata_checker_enabled" in added_columns or "advanced_scan_modes_enabled" in added_columns:
        for plan in DEFAULT_SUBSCRIPTION_PLANS:
            connection.execute(
                """
                UPDATE subscription_plans
                SET metadata_checker_enabled = ?,
                    advanced_scan_modes_enabled = ?
                WHERE plan_key = ?
                """,
                (
                    int(plan["metadata_checker_enabled"]),
                    int(plan["advanced_scan_modes_enabled"]),
                    plan["plan_key"],
                ),
            )

    for plan in DEFAULT_SUBSCRIPTION_PLANS:
        existing = connection.execute(
            "SELECT id FROM subscription_plans WHERE plan_key = ?",
            (plan["plan_key"],),
        ).fetchone()
        if existing:
            continue
        fields = dict(plan)
        fields["created_at"] = now
        fields["updated_at"] = now
        connection.execute(
            """
            INSERT INTO subscription_plans (
                plan_key, plan_name, price_usd, billing_label,
                monthly_scan_limit, images_per_scan_limit,
                csv_export_enabled, zip_export_enabled, readiness_report_enabled,
                best_shot_enabled, scan_history_enabled, project_folders_enabled,
                client_folders_enabled, metadata_checker_enabled, advanced_scan_modes_enabled,
                feature_summary, is_active, sort_order,
                created_at, updated_at
            )
            VALUES (
                :plan_key, :plan_name, :price_usd, :billing_label,
                :monthly_scan_limit, :images_per_scan_limit,
                :csv_export_enabled, :zip_export_enabled, :readiness_report_enabled,
                :best_shot_enabled, :scan_history_enabled, :project_folders_enabled,
                :client_folders_enabled, :metadata_checker_enabled, :advanced_scan_modes_enabled,
                :feature_summary, :is_active, :sort_order,
                :created_at, :updated_at
            )
            """,
            fields,
        )

    if owns_connection:
        connection.commit()
        connection.close()


def init_usage_resets(connection: sqlite3.Connection | None = None) -> None:
    """Track admin monthly usage resets without deleting scan history."""

    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    assert connection is not None
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_monthly_usage_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            month_key TEXT NOT NULL,
            reset_at TEXT NOT NULL,
            UNIQUE(user_id, month_key),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    if owns_connection:
        connection.commit()
        connection.close()


def init_payments(connection: sqlite3.Connection | None = None) -> None:
    """Create the manual payment table for beta billing."""

    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    assert connection is not None
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payment_ref TEXT UNIQUE,
            user_id INTEGER,
            public_user_id TEXT,
            user_email TEXT,
            plan_key TEXT,
            plan_name TEXT,
            amount REAL,
            currency TEXT DEFAULT 'USD',
            billing_period TEXT DEFAULT 'month',
            payment_method TEXT,
            payment_status TEXT DEFAULT 'pending',
            proof_note TEXT,
            admin_note TEXT,
            gateway_name TEXT DEFAULT 'manual',
            gateway_payment_id TEXT,
            created_at TEXT,
            paid_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    safe_columns = {
        "payment_ref": "payment_ref TEXT UNIQUE",
        "user_id": "user_id INTEGER",
        "public_user_id": "public_user_id TEXT",
        "user_email": "user_email TEXT",
        "plan_key": "plan_key TEXT",
        "plan_name": "plan_name TEXT",
        "amount": "amount REAL",
        "currency": "currency TEXT DEFAULT 'USD'",
        "billing_period": "billing_period TEXT DEFAULT 'month'",
        "payment_method": "payment_method TEXT",
        "payment_status": "payment_status TEXT DEFAULT 'pending'",
        "proof_note": "proof_note TEXT",
        "admin_note": "admin_note TEXT",
        "gateway_name": "gateway_name TEXT DEFAULT 'manual'",
        "gateway_payment_id": "gateway_payment_id TEXT",
        "created_at": "created_at TEXT",
        "paid_at": "paid_at TEXT",
        "updated_at": "updated_at TEXT",
    }
    for column_name, ddl in safe_columns.items():
        add_column_if_missing(connection, "payments", column_name, ddl)

    if owns_connection:
        connection.commit()
        connection.close()


def generate_next_payment_ref(connection: sqlite3.Connection) -> str:
    row = connection.execute("SELECT MAX(id) AS max_id FROM payments").fetchone()
    next_id = int(row["max_id"] or 0) + 1
    return f"PAY-{next_id:06d}"


def generate_next_public_user_id(connection: sqlite3.Connection | None = None) -> str:
    """Generate the next display-safe user ID like SG-000001."""

    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    assert connection is not None
    rows = connection.execute(
        "SELECT public_user_id FROM users WHERE public_user_id LIKE 'SG-%'"
    ).fetchall()
    highest = 0
    for row in rows:
        value = row["public_user_id"] or ""
        try:
            highest = max(highest, int(value.replace("SG-", "")))
        except ValueError:
            continue
    next_id = f"SG-{highest + 1:06d}"

    if owns_connection:
        connection.close()
    return next_id


def ensure_user_public_ids(connection: sqlite3.Connection | None = None) -> None:
    """Backfill public user IDs for existing accounts safely."""

    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    assert connection is not None
    if not column_exists(connection, "users", "public_user_id"):
        connection.execute("ALTER TABLE users ADD COLUMN public_user_id TEXT")

    rows = connection.execute(
        "SELECT id FROM users WHERE public_user_id IS NULL OR public_user_id = '' ORDER BY id"
    ).fetchall()
    for row in rows:
        connection.execute(
            "UPDATE users SET public_user_id = ? WHERE id = ?",
            (generate_next_public_user_id(connection), row["id"]),
        )

    if owns_connection:
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_public_user_id ON users(public_user_id)"
        )
        connection.commit()
        connection.close()


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_user(email: str, name: str, password: str) -> Dict:
    clean_email = email.strip().lower()
    clean_name = name.strip()

    if not clean_email or not clean_name or not password:
        return {"ok": False, "message": "Please fill all sign up fields."}

    if len(password) < 6:
        return {"ok": False, "message": "Password must be at least 6 characters."}

    try:
        with get_connection() as connection:
            ensure_user_public_ids(connection)
            public_user_id = generate_next_public_user_id(connection)
            role = "admin" if ADMIN_EMAIL and clean_email == ADMIN_EMAIL else "user"
            connection.execute(
                """
                INSERT INTO users (email, name, password_hash, plan, created_at, public_user_id, role)
                VALUES (?, ?, ?, 'free', ?, ?, ?)
                """,
                (clean_email, clean_name, hash_password(password), datetime.now().isoformat(), public_user_id, role),
            )
        return {"ok": True, "message": "Account created. You can log in now."}
    except sqlite3.IntegrityError:
        return {"ok": False, "message": "An account with this email already exists."}


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    clean_email = email.strip().lower()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()

    if row and int(row["is_disabled"] or 0):
        return None

    if row and verify_password(password, row["password_hash"]):
        return dict(row)
    return None


def get_user(user_id: int) -> Optional[Dict]:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def plan_row_to_dict(row: sqlite3.Row) -> Dict:
    plan = dict(row)
    # Backward-friendly aliases used by older app code.
    plan["monthly_scans"] = plan["monthly_scan_limit"]
    plan["images_per_scan"] = plan["images_per_scan_limit"]
    plan["csv_export"] = bool(plan["csv_export_enabled"])
    plan["zip_export"] = bool(plan["zip_export_enabled"])
    plan["readiness_report"] = bool(plan["readiness_report_enabled"])
    plan["best_shot"] = bool(plan["best_shot_enabled"])
    plan["batch_history"] = bool(plan["scan_history_enabled"])
    plan["project_folders"] = bool(plan["project_folders_enabled"])
    plan["client_folders"] = bool(plan["client_folders_enabled"])
    plan["metadata_checker"] = bool(plan["metadata_checker_enabled"])
    plan["advanced_scan_modes"] = bool(plan["advanced_scan_modes_enabled"])
    plan["lemon_variant_id"] = plan.get("lemon_variant_id") or ""
    plan["checkout_url"] = plan.get("checkout_url") or ""
    return plan


def get_all_subscription_plans(include_inactive: bool = False) -> List[Dict]:
    init_subscription_plans()
    query = "SELECT * FROM subscription_plans"
    params: tuple = ()
    if not include_inactive:
        query += " WHERE is_active = 1"
    query += " ORDER BY sort_order, id"
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [plan_row_to_dict(row) for row in rows]


def get_active_subscription_plans() -> List[Dict]:
    return get_all_subscription_plans(include_inactive=False)


def get_subscription_plan(plan_key: str) -> Optional[Dict]:
    init_subscription_plans()
    clean_key = plan_key_from_value(plan_key)
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM subscription_plans WHERE plan_key = ?",
            (clean_key,),
        ).fetchone()
    return plan_row_to_dict(row) if row else None


def update_subscription_plan(plan_key: str, fields: Dict) -> None:
    allowed_fields = {
        "plan_name",
        "price_usd",
        "billing_label",
        "monthly_scan_limit",
        "images_per_scan_limit",
        "csv_export_enabled",
        "zip_export_enabled",
        "readiness_report_enabled",
        "best_shot_enabled",
        "scan_history_enabled",
        "project_folders_enabled",
        "client_folders_enabled",
        "metadata_checker_enabled",
        "advanced_scan_modes_enabled",
        "feature_summary",
        "lemon_variant_id",
        "checkout_url",
        "is_active",
        "sort_order",
    }
    clean_key = plan_key_from_value(plan_key)
    updates = {key: value for key, value in fields.items() if key in allowed_fields}
    if not updates:
        return
    updates["updated_at"] = datetime.now().isoformat(timespec="seconds")
    set_clause = ", ".join(f"{key} = :{key}" for key in updates)
    updates["plan_key"] = clean_key
    with get_connection() as connection:
        connection.execute(
            f"UPDATE subscription_plans SET {set_clause} WHERE plan_key = :plan_key",
            updates,
        )


def reset_default_subscription_plans() -> None:
    now = datetime.now().isoformat(timespec="seconds")
    init_subscription_plans()
    with get_connection() as connection:
        for plan in DEFAULT_SUBSCRIPTION_PLANS:
            fields = dict(plan)
            fields["created_at"] = now
            fields["updated_at"] = now
            connection.execute(
                """
                INSERT INTO subscription_plans (
                    plan_key, plan_name, price_usd, billing_label,
                    monthly_scan_limit, images_per_scan_limit,
                    csv_export_enabled, zip_export_enabled, readiness_report_enabled,
                    best_shot_enabled, scan_history_enabled, project_folders_enabled,
                    client_folders_enabled, metadata_checker_enabled, advanced_scan_modes_enabled,
                    feature_summary, is_active, sort_order,
                    created_at, updated_at
                )
                VALUES (
                    :plan_key, :plan_name, :price_usd, :billing_label,
                    :monthly_scan_limit, :images_per_scan_limit,
                    :csv_export_enabled, :zip_export_enabled, :readiness_report_enabled,
                    :best_shot_enabled, :scan_history_enabled, :project_folders_enabled,
                    :client_folders_enabled, :metadata_checker_enabled, :advanced_scan_modes_enabled,
                    :feature_summary, :is_active, :sort_order,
                    :created_at, :updated_at
                )
                ON CONFLICT(plan_key) DO UPDATE SET
                    plan_name = excluded.plan_name,
                    price_usd = excluded.price_usd,
                    billing_label = excluded.billing_label,
                    monthly_scan_limit = excluded.monthly_scan_limit,
                    images_per_scan_limit = excluded.images_per_scan_limit,
                    csv_export_enabled = excluded.csv_export_enabled,
                    zip_export_enabled = excluded.zip_export_enabled,
                    readiness_report_enabled = excluded.readiness_report_enabled,
                    best_shot_enabled = excluded.best_shot_enabled,
                    scan_history_enabled = excluded.scan_history_enabled,
                    project_folders_enabled = excluded.project_folders_enabled,
                    client_folders_enabled = excluded.client_folders_enabled,
                    metadata_checker_enabled = excluded.metadata_checker_enabled,
                    advanced_scan_modes_enabled = excluded.advanced_scan_modes_enabled,
                    feature_summary = excluded.feature_summary,
                    is_active = excluded.is_active,
                    sort_order = excluded.sort_order,
                    updated_at = excluded.updated_at
                """,
                fields,
            )


def get_user_plan_details(user_id: Optional[int] = None, user_email: Optional[str] = None) -> Optional[Dict]:
    with get_connection() as connection:
        if user_id is not None:
            user = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        elif user_email is not None:
            user = connection.execute("SELECT * FROM users WHERE email = ?", (user_email.strip().lower(),)).fetchone()
        else:
            user = None
    if not user:
        return None
    return get_subscription_plan(user["plan"])


def get_user_by_public_user_id(public_user_id: str) -> Optional[Dict]:
    ensure_user_public_ids()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE public_user_id = ?",
            (public_user_id.strip().upper(),),
        ).fetchone()
    return dict(row) if row else None


def get_users_for_admin(
    search_query: str = "",
    plan_filter: str = "All",
    status_filter: str = "All",
    sort_by: str = "Newest",
    limit: int = 250,
) -> List[Dict]:
    """Search and filter users for the scalable admin table."""

    month_prefix = datetime.now().strftime("%Y-%m")
    where_clauses = []
    params: List = [month_prefix, month_prefix]

    clean_search = search_query.strip()
    if clean_search:
        where_clauses.append(
            "(u.public_user_id LIKE ? OR u.email LIKE ? OR u.name LIKE ?)"
        )
        search_value = f"%{clean_search}%"
        params.extend([search_value, search_value, search_value])

    if plan_filter and plan_filter != "All":
        where_clauses.append("lower(u.plan) = lower(?)")
        params.append(plan_key_from_value(plan_filter))

    if status_filter == "Active":
        where_clauses.append("u.is_disabled = 0")
    elif status_filter == "Disabled":
        where_clauses.append("u.is_disabled = 1")

    order_by = {
        "Newest": "u.created_at DESC",
        "Oldest": "u.created_at ASC",
        "Most scans": "total_scans DESC",
        "Plan": "p.sort_order ASC, u.created_at DESC",
    }.get(sort_by, "u.created_at DESC")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)
    with get_connection() as connection:
        ensure_user_public_ids(connection)
        init_usage_resets(connection)
        rows = connection.execute(
            f"""
            SELECT
                u.id,
                u.public_user_id,
                u.email,
                u.name,
                u.plan,
                u.created_at,
                u.is_disabled,
                p.plan_name,
                p.monthly_scan_limit,
                p.images_per_scan_limit,
                COUNT(s.id) AS total_scans,
                SUM(
                    CASE
                        WHEN substr(s.scan_datetime, 1, 7) = ?
                        AND (r.reset_at IS NULL OR s.scan_datetime > r.reset_at)
                        THEN 1 ELSE 0
                    END
                ) AS monthly_scans
            FROM users u
            LEFT JOIN subscription_plans p ON p.plan_key = lower(u.plan)
            LEFT JOIN user_monthly_usage_resets r
                ON r.user_id = u.id AND r.month_key = ?
            LEFT JOIN scans s ON s.user_id = u.id
            {where_sql}
            GROUP BY u.id
            ORDER BY {order_by}
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [dict(row) for row in rows]


def get_admin_overview_stats() -> Dict:
    month_prefix = datetime.now().strftime("%Y-%m")
    with get_connection() as connection:
        ensure_user_public_ids(connection)
        users = connection.execute(
            """
            SELECT u.*, p.plan_name, p.price_usd
            FROM users u
            LEFT JOIN subscription_plans p ON p.plan_key = lower(u.plan)
            """
        ).fetchall()
        scan_row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_scans,
                SUM(CASE WHEN substr(scan_datetime, 1, 7) = ? THEN 1 ELSE 0 END) AS scans_this_month,
                SUM(total_images) AS total_images_processed
            FROM scans
            """,
            (month_prefix,),
        ).fetchone()
        plan_rows = connection.execute(
            """
            SELECT COALESCE(p.plan_name, u.plan) AS plan_name, COUNT(*) AS count
            FROM users u
            LEFT JOIN subscription_plans p ON p.plan_key = lower(u.plan)
            GROUP BY COALESCE(p.plan_name, u.plan)
            ORDER BY count DESC
            """
        ).fetchall()
        recent_users = connection.execute(
            """
            SELECT public_user_id, email, name, plan, created_at, is_disabled
            FROM users
            ORDER BY created_at DESC
            LIMIT 8
            """
        ).fetchall()
        recent_scans = connection.execute(
            """
            SELECT s.id, u.public_user_id, u.email, p.name AS project_name,
                   s.batch_name, s.scan_datetime, s.total_images,
                   s.risky_pairs_count, s.near_duplicate_count,
                   s.highest_similarity_score
            FROM scans s
            JOIN users u ON u.id = s.user_id
            LEFT JOIN projects p ON p.id = s.project_id
            ORDER BY s.scan_datetime DESC
            LIMIT 8
            """
        ).fetchall()

    total_users = len(users)
    disabled_users = sum(1 for row in users if int(row["is_disabled"] or 0))
    paid_users = sum(1 for row in users if float(row["price_usd"] or 0) > 0)
    most_used_plan = dict(plan_rows[0])["plan_name"] if plan_rows else "None"
    return {
        "total_users": total_users,
        "active_users": total_users - disabled_users,
        "paid_users": paid_users,
        "disabled_users": disabled_users,
        "total_scans": int(scan_row["total_scans"] or 0),
        "scans_this_month": int(scan_row["scans_this_month"] or 0),
        "most_used_plan": most_used_plan,
        "total_images_processed": int(scan_row["total_images_processed"] or 0),
        "plan_distribution": [dict(row) for row in plan_rows],
        "recent_users": [dict(row) for row in recent_users],
        "recent_scans": [dict(row) for row in recent_scans],
    }


def get_scan_logs_for_admin(
    search_query: str = "",
    plan_filter: str = "All",
    sort_by: str = "Newest",
    limit: int = 250,
) -> List[Dict]:
    """Return compact scan logs for the admin Scan Logs tab."""

    where_clauses = []
    params: List = []
    clean_search = search_query.strip()
    if clean_search:
        where_clauses.append(
            """
            (u.public_user_id LIKE ? OR u.email LIKE ? OR s.batch_name LIKE ?
             OR COALESCE(p.name, '') LIKE ?)
            """
        )
        search_value = f"%{clean_search}%"
        params.extend([search_value, search_value, search_value, search_value])
    if plan_filter and plan_filter != "All":
        where_clauses.append("lower(u.plan) = lower(?)")
        params.append(plan_key_from_value(plan_filter))

    order_by = {
        "Newest": "s.scan_datetime DESC",
        "Highest risky pairs": "s.risky_pairs_count DESC",
        "Highest similarity": "s.highest_similarity_score DESC",
    }.get(sort_by, "s.scan_datetime DESC")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)
    with get_connection() as connection:
        ensure_user_public_ids(connection)
        rows = connection.execute(
            f"""
            SELECT
                s.id AS scan_id,
                u.public_user_id,
                u.email,
                COALESCE(sp.plan_name, u.plan) AS plan_name,
                p.name AS project_name,
                s.batch_name,
                s.scan_datetime,
                s.total_images,
                s.risky_pairs_count,
                s.near_duplicate_count,
                s.highest_similarity_score
            FROM scans s
            JOIN users u ON u.id = s.user_id
            LEFT JOIN projects p ON p.id = s.project_id
            LEFT JOIN subscription_plans sp ON sp.plan_key = lower(u.plan)
            {where_sql}
            ORDER BY {order_by}
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [dict(row) for row in rows]


def update_user_plan(user_id: int, plan: str) -> None:
    plan_key = plan_key_from_value(plan)
    if not get_subscription_plan(plan_key):
        raise ValueError("Unknown plan")

    with get_connection() as connection:
        connection.execute(
            "UPDATE users SET plan = ? WHERE id = ?",
            (plan_key, user_id),
        )


def set_user_disabled(user_id: int, is_disabled: bool) -> None:
    with get_connection() as connection:
        connection.execute(
            "UPDATE users SET is_disabled = ? WHERE id = ?",
            (1 if is_disabled else 0, user_id),
        )


def set_user_enabled(user_id: int, enabled: bool) -> None:
    set_user_disabled(user_id, not enabled)


def update_user_profile(user_id: int, display_name: str | None = None, profile_photo_path: str | None = None) -> None:
    """Update a user's display name and/or profile photo path."""
    with get_connection() as connection:
        if display_name is not None:
            connection.execute(
                "UPDATE users SET display_name = ? WHERE id = ?",
                (display_name, user_id),
            )
        if profile_photo_path is not None:
            connection.execute(
                "UPDATE users SET profile_photo_path = ? WHERE id = ?",
                (profile_photo_path, user_id),
            )


def get_current_month_reset_at(connection: sqlite3.Connection, user_id: int) -> Optional[str]:
    month_prefix = datetime.now().strftime("%Y-%m")
    row = connection.execute(
        """
        SELECT reset_at
        FROM user_monthly_usage_resets
        WHERE user_id = ? AND month_key = ?
        """,
        (user_id, month_prefix),
    ).fetchone()
    return row["reset_at"] if row else None


def reset_user_monthly_usage(user_id: int) -> None:
    """Reset usage for this month without deleting scan history."""

    month_prefix = datetime.now().strftime("%Y-%m")
    reset_at = datetime.now().isoformat(timespec="seconds")
    with get_connection() as connection:
        init_usage_resets(connection)
        connection.execute(
            """
            INSERT INTO user_monthly_usage_resets (user_id, month_key, reset_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, month_key) DO UPDATE SET reset_at = excluded.reset_at
            """,
            (user_id, month_prefix, reset_at),
        )


def list_all_users_with_usage() -> List[Dict]:
    month_prefix = datetime.now().strftime("%Y-%m")
    with get_connection() as connection:
        ensure_user_public_ids(connection)
        init_usage_resets(connection)
        rows = connection.execute(
            """
            SELECT
                u.id,
                u.public_user_id,
                u.email,
                u.name,
                u.plan,
                u.created_at,
                u.is_disabled,
                COUNT(s.id) AS total_scans,
                SUM(
                    CASE
                        WHEN substr(s.scan_datetime, 1, 7) = ?
                        AND (r.reset_at IS NULL OR s.scan_datetime > r.reset_at)
                        THEN 1 ELSE 0
                    END
                ) AS monthly_scans
            FROM users u
            LEFT JOIN user_monthly_usage_resets r
                ON r.user_id = u.id AND r.month_key = ?
            LEFT JOIN scans s ON s.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
            """,
            (month_prefix, month_prefix),
        ).fetchall()
    return [dict(row) for row in rows]


def current_month_scan_count(user_id: int) -> int:
    month_prefix = datetime.now().strftime("%Y-%m")
    with get_connection() as connection:
        init_usage_resets(connection)
        reset_at = get_current_month_reset_at(connection, user_id)
        reset_filter = "AND scan_datetime > ?" if reset_at else ""
        params = [user_id, month_prefix]
        if reset_at:
            params.append(reset_at)
        row = connection.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM scans
            WHERE user_id = ? AND substr(scan_datetime, 1, 7) = ?
            {reset_filter}
            """,
            tuple(params),
        ).fetchone()
    return int(row["count"])


def total_scan_count(user_id: int) -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM scans WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row["count"])


def get_or_create_project(user_id: int, project_name: str) -> int:
    clean_name = project_name.strip() or "Default Project"

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM projects WHERE user_id = ? AND name = ?",
            (user_id, clean_name),
        ).fetchone()
        if existing:
            return int(existing["id"])

        cursor = connection.execute(
            """
            INSERT INTO projects (user_id, name, created_at)
            VALUES (?, ?, ?)
            """,
            (user_id, clean_name, datetime.now().isoformat()),
        )
        return int(cursor.lastrowid)


def list_projects(user_id: int) -> List[Dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.id, p.name, p.created_at, COUNT(s.id) AS scan_count
            FROM projects p
            LEFT JOIN scans s ON s.project_id = p.id AND s.user_id = p.user_id
            WHERE p.user_id = ?
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_batch_names(user_id: int, project_name: str) -> List[str]:
    clean_name = project_name.strip() or "Default Project"
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT s.batch_name
            FROM scans s
            JOIN projects p ON p.id = s.project_id
            WHERE s.user_id = ? AND p.user_id = ? AND p.name = ?
            ORDER BY s.scan_datetime DESC
            """,
            (user_id, user_id, clean_name),
        ).fetchall()
    return [row["batch_name"] for row in rows]


def save_scan(
    user_id: int,
    project_id: int,
    batch_name: str,
    total_images: int,
    risky_pairs_count: int,
    near_duplicate_count: int,
    highest_similarity_score: float,
    csv_rows: List[Dict],
) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO scans (
                user_id,
                project_id,
                batch_name,
                scan_datetime,
                total_images,
                risky_pairs_count,
                near_duplicate_count,
                highest_similarity_score,
                csv_report_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                project_id,
                batch_name.strip() or "Untitled Batch",
                datetime.now().isoformat(timespec="seconds"),
                total_images,
                risky_pairs_count,
                near_duplicate_count,
                highest_similarity_score,
                json.dumps(csv_rows),
            ),
        )
        return int(cursor.lastrowid)


def list_scans(user_id: int) -> List[Dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                s.id,
                s.batch_name,
                s.scan_datetime,
                s.total_images,
                s.risky_pairs_count,
                s.near_duplicate_count,
                s.highest_similarity_score,
                p.name AS project_name
            FROM scans s
            LEFT JOIN projects p ON p.id = s.project_id
            WHERE s.user_id = ?
            ORDER BY s.scan_datetime DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_scan(user_id: int, scan_id: int) -> Optional[Dict]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                s.*,
                p.name AS project_name
            FROM scans s
            LEFT JOIN projects p ON p.id = s.project_id
            WHERE s.user_id = ? AND s.id = ?
            """,
            (user_id, scan_id),
        ).fetchone()
    return dict(row) if row else None


def scan_report_rows(scan: Dict) -> List[Dict]:
    return json.loads(scan["csv_report_json"] or "[]")


def create_payment(
    user: Dict,
    plan: Dict,
    payment_method: str = "Bank Transfer",
    proof_note: str = "",
    gateway_name: str = "manual",
) -> Dict:
    """Create a pending manual payment request.

    Future PayHere/Paddle/Lemon Squeezy webhooks can create or update rows in
    this same table using gateway_name and gateway_payment_id.
    """

    init_payments()
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection() as connection:
        init_payments(connection)
        ensure_user_public_ids(connection)
        fresh_user = connection.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        payment_ref = generate_next_payment_ref(connection)
        connection.execute(
            """
            INSERT INTO payments (
                payment_ref, user_id, public_user_id, user_email,
                plan_key, plan_name, amount, currency, billing_period,
                payment_method, payment_status, proof_note, gateway_name,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            """,
            (
                payment_ref,
                user["id"],
                fresh_user["public_user_id"] if fresh_user else user.get("public_user_id"),
                user["email"],
                plan["plan_key"],
                plan["plan_name"],
                float(plan["price_usd"]),
                "USD",
                plan.get("billing_label") or "month",
                payment_method,
                proof_note,
                gateway_name,
                now,
                now,
            ),
        )
    return get_payment_by_ref(payment_ref) or {"payment_ref": payment_ref}


def get_payments(
    status_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    plan_filter: Optional[str] = None,
    sort_by: str = "Newest",
    limit: int = 300,
) -> List[Dict]:
    init_payments()
    allowed_statuses = {"pending", "paid", "failed", "rejected", "refunded"}
    where_clauses = []
    params: List = []
    if status_filter and status_filter != "All":
        clean_status = status_filter.strip().lower()
        if clean_status in allowed_statuses:
            where_clauses.append("payment_status = ?")
            params.append(clean_status)
    if search_query:
        search_value = f"%{search_query.strip()}%"
        where_clauses.append(
            "(payment_ref LIKE ? OR CAST(user_id AS TEXT) LIKE ? OR public_user_id LIKE ? OR user_email LIKE ?)"
        )
        params.extend([search_value, search_value, search_value, search_value])
    if plan_filter and plan_filter != "All":
        where_clauses.append("plan_key = ?")
        params.append(plan_key_from_value(plan_filter))

    order_by = {
        "Newest": "created_at DESC",
        "Oldest": "created_at ASC",
        "Amount high to low": "amount DESC, created_at DESC",
        "Amount low to high": "amount ASC, created_at DESC",
    }.get(sort_by, "created_at DESC")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM payments
            {where_sql}
            ORDER BY {order_by}
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [dict(row) for row in rows]


def get_payment_by_ref(payment_ref: str) -> Optional[Dict]:
    init_payments()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM payments WHERE payment_ref = ?",
            (payment_ref.strip().upper(),),
        ).fetchone()
    return dict(row) if row else None


def update_payment_status(payment_ref: str, status: str, admin_note: Optional[str] = None) -> None:
    allowed_statuses = {"pending", "paid", "failed", "rejected", "refunded"}
    clean_status = status.strip().lower()
    if clean_status not in allowed_statuses:
        raise ValueError("Unsupported payment status")

    now = datetime.now().isoformat(timespec="seconds")
    paid_at = now if clean_status == "paid" else None
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE payments
            SET payment_status = ?,
                admin_note = COALESCE(?, admin_note),
                paid_at = COALESCE(?, paid_at),
                updated_at = ?
            WHERE payment_ref = ?
            """,
            (clean_status, admin_note, paid_at, now, payment_ref.strip().upper()),
        )


def approve_manual_payment(payment_ref: str, admin_user_email: str) -> None:
    """Mark payment paid and upgrade the user to the paid plan."""

    payment = get_payment_by_ref(payment_ref)
    if not payment:
        raise ValueError("Payment not found")
    note = f"Approved by {admin_user_email}"
    update_payment_status(payment_ref, "paid", note)
    update_user_plan(int(payment["user_id"]), payment["plan_key"])


def reject_manual_payment(payment_ref: str, admin_note: Optional[str] = None) -> None:
    update_payment_status(payment_ref, "rejected", admin_note or "Rejected by admin")


def get_financial_summary() -> Dict:
    init_payments()
    month_prefix = datetime.now().strftime("%Y-%m")
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN payment_status = 'paid' THEN amount ELSE 0 END) AS total_revenue,
                SUM(CASE WHEN payment_status = 'paid' AND substr(paid_at, 1, 7) = ? THEN amount ELSE 0 END) AS this_month_revenue,
                SUM(CASE WHEN payment_status = 'pending' THEN 1 ELSE 0 END) AS pending_payments,
                SUM(CASE WHEN payment_status = 'paid' THEN 1 ELSE 0 END) AS paid_payments,
                SUM(CASE WHEN payment_status = 'refunded' THEN 1 ELSE 0 END) AS refunded_payments
            FROM payments
            """,
            (month_prefix,),
        ).fetchone()
        paid_users = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM users u
            LEFT JOIN subscription_plans p ON p.plan_key = lower(u.plan)
            WHERE COALESCE(p.price_usd, 0) > 0 AND u.is_disabled = 0
            """
        ).fetchone()
    return {
        "total_revenue": float(row["total_revenue"] or 0),
        "this_month_revenue": float(row["this_month_revenue"] or 0),
        "pending_payments": int(row["pending_payments"] or 0),
        "paid_payments": int(row["paid_payments"] or 0),
        "refunded_payments": int(row["refunded_payments"] or 0),
        "active_paid_users": int(paid_users["count"] or 0),
    }


def get_monthly_revenue_summary() -> List[Dict]:
    init_payments()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT substr(paid_at, 1, 7) AS month, SUM(amount) AS revenue, COUNT(*) AS payments
            FROM payments
            WHERE payment_status = 'paid' AND paid_at IS NOT NULL
            GROUP BY substr(paid_at, 1, 7)
            ORDER BY month DESC
            LIMIT 12
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_plan_revenue_breakdown() -> List[Dict]:
    init_payments()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.plan_key,
                p.plan_name,
                (
                    SELECT COUNT(*)
                    FROM users u
                    WHERE lower(u.plan) = p.plan_key
                ) AS users_count,
                (
                    SELECT COUNT(DISTINCT pay.user_id)
                    FROM payments pay
                    WHERE pay.plan_key = p.plan_key
                    AND pay.payment_status = 'paid'
                ) AS paid_users,
                (
                    SELECT COALESCE(SUM(pay.amount), 0)
                    FROM payments pay
                    WHERE pay.plan_key = p.plan_key
                    AND pay.payment_status = 'paid'
                ) AS revenue
            FROM subscription_plans p
            ORDER BY p.sort_order
            """
        ).fetchall()
    return [dict(row) for row in rows]
