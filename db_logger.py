"""
db_logger.py
------------
Logs portfolio risk snapshots to MySQL for historical tracking.
Auto-creates schema, handles upserts and chunked inserts.
"""

import json
import time
import logging
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import Error

log = logging.getLogger(__name__)

DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 3306,
    "user"    : "root",
    "password": "your_password",
    "database": "financial_risk_db",
}

SCHEMAS = {
    "risk_snapshots": """
        CREATE TABLE IF NOT EXISTS risk_snapshots (
            id                    INT AUTO_INCREMENT PRIMARY KEY,
            snapshot_date         DATE NOT NULL,
            ticker                VARCHAR(20) NOT NULL,
            portfolio_name        VARCHAR(50) DEFAULT 'default',
            annualized_return_pct DECIMAL(8,4),
            annualized_vol_pct    DECIMAL(8,4),
            sharpe_ratio          DECIMAL(8,4),
            sortino_ratio         DECIMAL(8,4),
            calmar_ratio          DECIMAL(8,4),
            max_drawdown_pct      DECIMAL(8,4),
            var_historical        DECIMAL(10,6),
            var_parametric        DECIMAL(10,6),
            var_monte_carlo       DECIMAL(10,6),
            cvar_historical       DECIMAL(10,6),
            var_hist_dollar       DECIMAL(14,2),
            beta                  DECIMAL(8,4),
            alpha                 DECIMAL(8,4),
            skewness              DECIMAL(8,4),
            excess_kurtosis       DECIMAL(8,4),
            created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_ticker_date (ticker, snapshot_date, portfolio_name)
        )
    """,
    "portfolio_snapshots": """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id                     INT AUTO_INCREMENT PRIMARY KEY,
            snapshot_date          DATE NOT NULL,
            portfolio_name         VARCHAR(50) NOT NULL,
            tickers                TEXT,
            weights                TEXT,
            portfolio_value        DECIMAL(16,2),
            annualized_return_pct  DECIMAL(8,4),
            annualized_vol_pct     DECIMAL(8,4),
            sharpe_ratio           DECIMAL(8,4),
            max_drawdown_pct       DECIMAL(8,4),
            var_dollar             DECIMAL(14,2),
            cvar_dollar            DECIMAL(14,2),
            created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_port_date (portfolio_name, snapshot_date)
        )
    """,
}

MAX_RETRIES = 3


def get_connection():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            return conn
        except Error as e:
            log.warning(f"Connection attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError("Could not connect to MySQL")


def ensure_tables(conn):
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.execute(f"USE {DB_CONFIG['database']}")
    for table, ddl in SCHEMAS.items():
        cursor.execute(ddl)
    conn.commit()
    cursor.close()
    log.info("Tables ensured")


def log_asset_risk(conn, report: dict, portfolio_name: str = "default"):
    """Insert or update a single asset risk snapshot."""
    cursor = conn.cursor()
    today  = datetime.today().date()

    sql = """
        INSERT INTO risk_snapshots
            (snapshot_date, ticker, portfolio_name,
             annualized_return_pct, annualized_vol_pct, sharpe_ratio,
             sortino_ratio, calmar_ratio, max_drawdown_pct,
             var_historical, var_parametric, var_monte_carlo,
             cvar_historical, var_hist_dollar, beta, alpha,
             skewness, excess_kurtosis)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            annualized_return_pct = VALUES(annualized_return_pct),
            annualized_vol_pct    = VALUES(annualized_vol_pct),
            sharpe_ratio          = VALUES(sharpe_ratio),
            var_historical        = VALUES(var_historical),
            cvar_historical       = VALUES(cvar_historical),
            beta                  = VALUES(beta)
    """
    vals = (
        today,
        report.get("ticker", "UNKNOWN"),
        portfolio_name,
        report.get("annualized_return", report.get("mean", 0)) * 100,
        report.get("annualized_vol", report.get("std_annual", 0)) * 100,
        report.get("sharpe_ratio", 0),
        report.get("sortino_ratio", 0),
        report.get("calmar_ratio", 0),
        report.get("max_drawdown_pct", 0),
        report.get("var_historical", 0),
        report.get("var_parametric", 0),
        report.get("var_monte_carlo", 0),
        report.get("cvar_historical", 0),
        report.get("var_hist_dollar", 0),
        report.get("beta", None),
        report.get("alpha", None),
        report.get("skewness", None),
        report.get("excess_kurtosis", None),
    )
    cursor.execute(sql, vals)
    conn.commit()
    cursor.close()
    log.info(f"Logged risk snapshot for {report.get('ticker')}")


def log_portfolio_risk(conn, report: dict, portfolio_name: str = "default"):
    """Log portfolio-level risk snapshot."""
    cursor = conn.cursor()
    today  = datetime.today().date()

    sql = """
        INSERT INTO portfolio_snapshots
            (snapshot_date, portfolio_name, tickers, weights,
             portfolio_value, annualized_return_pct, annualized_vol_pct,
             sharpe_ratio, max_drawdown_pct, var_dollar, cvar_dollar)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            annualized_return_pct = VALUES(annualized_return_pct),
            annualized_vol_pct    = VALUES(annualized_vol_pct),
            sharpe_ratio          = VALUES(sharpe_ratio),
            var_dollar            = VALUES(var_dollar)
    """
    vals = (
        today,
        portfolio_name,
        json.dumps(report.get("tickers", [])),
        json.dumps(report.get("weights", {})),
        report.get("portfolio_value", 0),
        report.get("annualized_return_pct", 0),
        report.get("annualized_vol_pct", 0),
        report.get("sharpe_ratio", 0),
        report.get("max_drawdown_pct", 0),
        report.get("var_dollar", 0),
        report.get("cvar_dollar", 0),
    )
    cursor.execute(sql, vals)
    conn.commit()
    cursor.close()
    log.info(f"Logged portfolio snapshot: {portfolio_name}")


def get_risk_history(conn, ticker: str, days: int = 90) -> pd.DataFrame:
    """Retrieve historical risk snapshots for a ticker."""
    cursor = conn.cursor(dictionary=True)
    sql = """
        SELECT * FROM risk_snapshots
        WHERE ticker = %s
          AND snapshot_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY snapshot_date
    """
    cursor.execute(sql, (ticker, days))
    rows = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(rows) if rows else pd.DataFrame()
