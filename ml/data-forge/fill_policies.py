#!/usr/bin/env python3
"""
fill_policies.py – Complete Reset Harvester

1. Purges (blanks) all existing .txt files in the target directories.
2. Reconstructs the domain from the filename.
3. Fetches the policy using dual-tier stealth.
4. Cleans HTML, filters non-English text, and drops UI junk.
5. Populates the blank files with ML-ready text.
"""

import os
import asyncio
import re
from pathlib import Path
from urllib.parse import urljoin

from curl_cffi.requests import AsyncSession
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from readability import Document
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

INPUT_DIRS = ["./raw-policies", "./indian-seeds"]

PATH_PATTERNS = [
    "/privacy",
    "/privacy-policy",
    "/legal/privacy",
    "/legal/privacy-policy",
    "/en/privacy",
    "/en/privacy-policy",
    "/us/en/privacy",
    "/us/en/privacy-policy",
    "/about/privacy",
    "/policies/privacy"
]

# Regex for non-Latin scripts (Devanagari, CJK, Cyrillic, Arabic)
NON_ENGLISH_SCRIPTS_RE = re.compile(r'[\u0900-\u097F\u4e00-\u9fff\u0600-\u06FF\u0400-\u04FF]')

def filter_english(text: str, non_latin_threshold: float = 0.05) -> str:
    """Filters out lines containing significant non-English/non-Latin text."""
    clean_lines = []
    for line in text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        non_latin_chars = len(NON_ENGLISH_SCRIPTS_RE.findall(line_str))
        if (non_latin_chars / max(1, len(line_str))) < non_latin_threshold:
            clean_lines.append(line_str)
    return '\n'.join(clean_lines)

def clean_html_for_ml(html_content: str) -> str:
    """Extracts readable legal text, filters non-English content, and normalizes formatting."""
    if not html_content:
        return None
        
    try:
        # 1. Readability extraction (drops navbars, sidebars, footers)
        doc = Document(html_content)
        summary_html = doc.summary()
        
        # 2. Strip HTML tags
        soup = BeautifulSoup(summary_html, 'html.parser')
        text = soup.get_text(separator='\n')
        
        # 3. Unicode & Space Normalization
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 4. Enforce English-only filter
        text = filter_english(text)
        
        # 5. Smart line filtering (Preserves short section titles, drops menu junk)
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_lines = []
        
        junk_patterns = re.compile(r'^(?:cookie settings|accept all|sign in|log in|privacy policy|home|menu|search|back to top|copyright|all rights reserved)$', re.IGNORECASE)

        for line in raw_lines:
            if junk_patterns.match(line):
                continue
            # Keep if line has standard punctuation/sentences (>20 chars) OR looks like a section header/numbered clause
            if len(line) > 20 or re.match(r'^(?:\d+\.|\#|section|clause|article|\b[A-Z])', line, re.IGNORECASE):
                cleaned_lines.append(line)
                
        cleaned_text = '\n\n'.join(cleaned_lines).strip()
        
        # 6. Quality Gate: Ensure policy length exceeds minimum valid threshold (>2000 chars)
        if len(cleaned_text) > 2000:
            return cleaned_text
    except Exception:
        pass
        
    return None

async def try_stealth_http(domain):
    base = f"https://{domain}" if not domain.startswith('http') else domain
    async with AsyncSession(impersonate="chrome120") as session:
        for path in PATH_PATTERNS:
            url = urljoin(base, path)
            try:
                resp = await session.get(url, timeout=8, allow_redirects=True)
                if resp.status_code == 200:
                    cleaned = clean_html_for_ml(resp.text)
                    if cleaned:
                        return url, cleaned
            except Exception:
                continue
    return None, None

async def try_stealth_playwright(domain, browser):
    base = f"https://{domain}" if not domain.startswith('http') else domain
    context = await browser.new_context()
    page = await context.new_page()
    
    # Apply Playwright 2.x stealth API
    await Stealth().apply_stealth_async(page)

    # Block heavy media resources
    await page.route("**/*", lambda route:
        route.abort() if route.request.resource_type in ["image", "media", "font"] else route.continue_()
    )

    for path in PATH_PATTERNS:
        url = urljoin(base, path)
        try:
            resp = await page.goto(url, timeout=12000, wait_until='domcontentloaded')
            if resp and resp.status == 200:
                await page.wait_for_timeout(1500)
                raw_html = await page.content()
                cleaned = clean_html_for_ml(raw_html)
                if cleaned:
                    await context.close()
                    return url, cleaned
        except Exception:
            continue
            
    await context.close()
    return None, None

async def harvest_and_fill(domain, filepath, browser, semaphore, pbar):
    async with semaphore:
        try:
            url, text = await try_stealth_http(domain)
            if not text:
                url, text = await try_stealth_playwright(domain, browser)

            # Fill the blanked file with cleaned ML text
            if text:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
        finally:
            pbar.update(1)

async def main():
    pending_tasks = []
    
    print("🧹 Phase 1: Blanking existing policy files...")
    for directory in INPUT_DIRS:
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found: {directory}")
            continue
            
        for filepath in Path(directory).glob("*.txt"):
            # 1. DELETE ALL CONTENTS (Truncate to 0 bytes)
            open(filepath, 'w').close()
            
            # 2. Reconstruct domain and add to fetch queue
            domain = filepath.stem.replace('_', '.')
            pending_tasks.append((domain, str(filepath)))

    print(f"🚀 Phase 2: Fetching & Cleaning {len(pending_tasks)} domains...")
    if not pending_tasks:
        print("✅ No target files found. Exiting.")
        return

    semaphore = asyncio.Semaphore(15)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        with tqdm(total=len(pending_tasks), desc="Fetching & Processing") as pbar:
            tasks = [
                harvest_and_fill(domain, filepath, browser, semaphore, pbar)
                for domain, filepath in pending_tasks
            ]
            await asyncio.gather(*tasks)

    print("✅ Finished. All target .txt files have been purged and refilled with clean, English-filtered policy text.")

if __name__ == "__main__":
    asyncio.run(main())