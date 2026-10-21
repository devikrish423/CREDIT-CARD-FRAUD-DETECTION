"""SQLite persistence for fraud-prediction history."""

from datetime import datetime
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "prediction_history.db"


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    """Create the prediction history table when it does not already exist."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
                probability REAL CHECK (
                    probability IS NULL OR (probability >= 0 AND probability <= 1)
                ),
                amount REAL CHECK (amount IS NULL OR amount >= 0)
            )
            """
        )


def add_prediction(
    predicted_label: int,
    probability: float | None,
    amount: float | None = None,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Store a prediction and return its database ID."""
    if predicted_label not in (0, 1):
        raise ValueError("predicted_label must be 0 (legitimate) or 1 (fraud).")
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1.")
    if amount is not None and amount < 0:
        raise ValueError("amount cannot be negative.")

    initialize_database(database_path)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (timestamp, predicted_label, probability, amount)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp, predicted_label, probability, amount),
        )
        prediction_id = cursor.lastrowid
    if prediction_id is None:
        raise RuntimeError("SQLite did not return an ID for the saved prediction.")
    return prediction_id


def get_prediction_history(database_path: Path = DATABASE_PATH) -> list[dict]:
    """Return prediction records, newest first."""
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, timestamp, predicted_label, probability, amount
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def clear_prediction_history(database_path: Path = DATABASE_PATH) -> None:
    """Remove all stored prediction history."""
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM predictions")
