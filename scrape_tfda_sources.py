from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


TFDA_PAGES = [
    "https://www.fda.gov.tw/TC/siteListContent.aspx?id=30804&sid=10992",
    "https://www.fda.gov.tw/TC/siteListContent.aspx?id=46365&sid=1894",
    "https://www.fda.gov.tw/tc/newsContent.aspx?cid=3&id=31285",
]

OUTPUT_DIR = Path("tfda_downloads")
INDEX_PATH = OUTPUT_DIR / "sources_index.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; student-research-bot/1.0; "
        "+https://example.com/course-report)"
    )
}

REQUEST_TIMEOUT = 20
SLEEP_SECONDS = 1.5


def is_tfda_url(url: str) -> bool:
    """Only allow downloads from fda.gov.tw."""
    parsed = urlparse(url)
    return parsed.netloc.endswith("fda.gov.tw")


def safe_filename(name: str, fallback: str) -> str:
    """Create a filesystem-safe filename."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name)
    cleaned = cleaned.strip(" ._")
    return cleaned or fallback


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return response.text


def extract_attachment_links(page_url: str, html: str) -> list[dict[str, str]]:
    """Extract likely attachment links from a TFDA announcement page."""
    soup = BeautifulSoup(html, "lxml")
    attachments: list[dict[str, str]] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        text = anchor.get_text(" ", strip=True)

        absolute_url = urljoin(page_url, href)

        # TFDA attachments often use GetFile.ashx.
        is_attachment = "GetFile.ashx" in absolute_url or "download" in href.lower()

        if is_attachment and is_tfda_url(absolute_url):
            attachments.append(
                {
                    "title": text or "tfda_attachment",
                    "url": absolute_url,
                    "source_page": page_url,
                }
            )

    # Remove duplicates while preserving order.
    seen = set()
    unique_attachments = []
    for item in attachments:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique_attachments.append(item)

    return unique_attachments


def download_file(session: requests.Session, url: str, output_path: Path) -> None:
    with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    file.write(chunk)


def write_index(rows: Iterable[dict[str, str]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames = ["title", "url", "source_page", "saved_as"]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    downloaded_rows: list[dict[str, str]] = []

    for page_url in TFDA_PAGES:
        print(f"Fetching page: {page_url}")

        html = fetch_html(session, page_url)
        attachments = extract_attachment_links(page_url, html)

        if not attachments:
            print("  No attachments found.")
            continue

        for index, attachment in enumerate(attachments, start=1):
            title = attachment["title"]
            file_name = safe_filename(title, fallback=f"attachment_{index}") + ".pdf"
            output_path = OUTPUT_DIR / file_name

            print(f"  Downloading: {title}")
            download_file(session, attachment["url"], output_path)

            downloaded_rows.append(
                {
                    **attachment,
                    "saved_as": str(output_path),
                }
            )

            time.sleep(SLEEP_SECONDS)

        time.sleep(SLEEP_SECONDS)

    write_index(downloaded_rows, INDEX_PATH)
    print(f"Done. Index saved to: {INDEX_PATH}")


if __name__ == "__main__":
    main()