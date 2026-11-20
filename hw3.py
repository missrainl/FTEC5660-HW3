#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any
import os


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.

CHOSEN_POINTS: list[int] = [1, 5, 6]  ### YOUR CODE HERE


def build_agent() -> Any:
    """Create and return your model or agent once.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env. Each company has its own tools (see ``analyze_companies``), so
    you will probably create one tool-calling agent per company there.
    """
    ### YOUR CODE HERE
    import os
    from langchain_deepseek import ChatDeepSeek

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Please check your .env file."
        )

    model = ChatDeepSeek(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0
    )

    return model


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Analyze each company with its SQL tool and a DeepSeek LangChain agent."""
    import asyncio
    from langchain.agents import create_agent

    # All calculations are performed by SQLite, not by the language model.
    sql = """
    WITH revenue AS (
        SELECT
            SUM(CASE
                WHEN order_date >= '2023-01-01'
                 AND order_date < '2024-01-01'
                THEN amount ELSE 0 END) AS revenue_2023,
            SUM(CASE
                WHEN order_date >= '2025-01-01'
                 AND order_date < '2026-01-01'
                THEN amount ELSE 0 END) AS revenue_2025
        FROM orders
    ),
    costs AS (
        SELECT
            SUM(CASE
                WHEN month >= '2023-01' AND month < '2024-01'
                THEN cost_of_sales + selling_admin + research_development
                ELSE 0 END) AS costs_2023,
            SUM(CASE
                WHEN month >= '2025-01' AND month < '2026-01'
                THEN cost_of_sales + selling_admin + research_development
                ELSE 0 END) AS costs_2025
        FROM monthly_costs
    ),
    base AS (
        SELECT
            c.name AS company,
            c.industry,
            r.revenue_2023,
            r.revenue_2025,
            co.costs_2023,
            co.costs_2025,
            im.revenue_growth_2y_pct AS industry_growth_pct,
            im.operating_margin_pct AS industry_margin_pct
        FROM company AS c
        CROSS JOIN revenue AS r
        CROSS JOIN costs AS co
        JOIN industry_medians AS im
          ON im.industry = c.industry
    ),
    metrics AS (
        SELECT
            *,
            100.0 * (revenue_2025 - revenue_2023)
                / NULLIF(revenue_2023, 0) AS company_growth_pct,
            100.0 * (revenue_2025 - costs_2025)
                / NULLIF(revenue_2025, 0) AS margin_2025_pct,
            100.0 * (revenue_2023 - costs_2023)
                / NULLIF(revenue_2023, 0) AS margin_2023_pct
        FROM base
    )
    SELECT
        company,
        industry,
        ROUND(company_growth_pct, 2) AS company_growth_pct,
        ROUND(industry_growth_pct, 2) AS industry_growth_pct,
        ROUND(margin_2025_pct, 2) AS margin_2025_pct,
        ROUND(industry_margin_pct, 2) AS industry_margin_pct,
        ROUND(margin_2023_pct, 2) AS margin_2023_pct,
        CASE
            WHEN company_growth_pct >= industry_growth_pct
            THEN 'PASS' ELSE 'FAIL'
        END AS point_1,
        CASE
            WHEN margin_2025_pct >= industry_margin_pct
            THEN 'PASS' ELSE 'FAIL'
        END AS point_5,
        CASE
            WHEN margin_2025_pct >= margin_2023_pct
            THEN 'PASS' ELSE 'FAIL'
        END AS point_6
    FROM metrics;
    """

    system_prompt = """
You are a financial analyst applying Philip Fisher's investment framework.

The user will ask you to analyze one fictional company and provide an SQL query.
You MUST call the company's sql_query tool with that exact SQL query.
Do not change the query, do not use other data sources, and do not invent data.

The SQL result already contains the verdict columns point_1, point_5, and point_6.
Use those verdicts exactly. Do not recalculate or override them.

Point 1: Market potential for sales growth.
PASS rule: company two-year revenue growth >= its industry median.

Point 5: Worthwhile profit margin.
PASS rule: 2025 operating margin >= its industry median operating margin.

Point 6: Maintaining or improving margins.
PASS rule: 2025 full-year operating margin >= 2023 full-year operating margin.

After calling the SQL tool, provide a concise 2-3 sentence summary of the evidence.
Do not return JSON. Do not claim a verdict if the SQL tool returned an error or no data.
"""

    semaphore = asyncio.Semaphore(3)

    def extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        return str(content) if content is not None else ""

    def parse_sql_result(text: str) -> dict[str, str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or lines[0].startswith("SQL error:"):
            raise ValueError(text or "SQL tool returned an empty result")

        header = [part.strip() for part in lines[0].split("|")]
        data_line = next(
            (line for line in lines[1:] if "|" in line and not line.startswith("...")),
            None,
        )
        if data_line is None:
            raise ValueError(f"SQL tool returned no data row: {text}")

        values = [part.strip() for part in data_line.split("|")]
        if len(header) != len(values):
            raise ValueError(f"Could not parse SQL result row: {text}")

        row = dict(zip(header, values))
        required = ("point_1", "point_5", "point_6")
        for key in required:
            if row.get(key, "").upper() not in ("PASS", "FAIL"):
                raise ValueError(f"Missing/invalid {key} in SQL result: {text}")
        return row

    async def analyze_one(name: str, tools: list[Any]):
        async with semaphore:
            sql_tool = next((tool for tool in tools if tool.name == "sql_query"), None)
            if sql_tool is None:
                return name, {
                    "points": {p: None for p in CHOSEN_POINTS},
                    "summary": "Analysis error: sql_query tool not found.",
                }

            messages = []
            try:
                company_agent = create_agent(
                    model=agent,
                    tools=tools,
                    system_prompt=system_prompt,
                )
                result = await asyncio.wait_for(
                    company_agent.ainvoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        f"Analyze company: {name}. "
                                        "Call sql_query with this exact query:\n"
                                        f"{sql}"
                                    ),
                                }
                            ]
                        },
                        config={"recursion_limit": 8},
                    ),
                    timeout=120,
                )
                messages = result.get("messages", [])
            except Exception as exc:
                # The provided LangChain SQL tool is still used below as a
                # safe fallback if the model/tool-calling step fails.
                print(f"Agent call failed for {name}: {type(exc).__name__}: {exc}")

            # Prefer the SQL tool result produced during the agent run.
            sql_output = None
            for message in messages:
                if getattr(message, "type", "") == "tool":
                    content = extract_text(getattr(message, "content", ""))
                    if "point_1" in content and "point_5" in content and "point_6" in content:
                        sql_output = content

            # If the agent did not successfully call the SQL tool, invoke
            # the same provided LangChain tool directly; never inspect the DB.
            if sql_output is None:
                try:
                    sql_output = await sql_tool.ainvoke({"query": sql})
                except Exception as exc:
                    print(f"SQL tool failed for {name}: {type(exc).__name__}: {exc}")
                    return name, {
                        "points": {p: None for p in CHOSEN_POINTS},
                        "summary": f"Analysis error: {type(exc).__name__}",
                    }

            try:
                row = parse_sql_result(extract_text(sql_output))
            except Exception as exc:
                print(f"Could not parse SQL result for {name}: {exc}")
                return name, {
                    "points": {p: None for p in CHOSEN_POINTS},
                    "summary": f"SQL result error: {type(exc).__name__}",
                }

            # Verdicts come directly from SQLite CASE expressions.
            points = {
                1: row["point_1"].upper(),
                5: row["point_5"].upper(),
                6: row["point_6"].upper(),
            }

            # Use the model's final response as the explanation, but never
            # let its wording change the SQL-derived verdicts.
            summary = ""
            for message in reversed(messages):
                if getattr(message, "type", "") == "ai":
                    summary = extract_text(getattr(message, "content", "")).strip()
                    if summary:
                        break

            if not summary:
                summary = (
                    f"Revenue growth was {row['company_growth_pct']}% versus "
                    f"the industry median of {row['industry_growth_pct']}%. "
                    f"The 2025 operating margin was {row['margin_2025_pct']}% "
                    f"versus the industry median of {row['industry_margin_pct']}%, "
                    f"and compared with {row['margin_2023_pct']}% in 2023."
                )

            return name, {"points": points, "summary": summary}

    results = await asyncio.gather(
        *(analyze_one(name, tools) for name, tools in companies.items())
    )
    return dict(results)



# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())
