#!/usr/bin/env python3
"""
fetch_indian_seeds.py – Indian Privacy Policy Harvester for GAN Forge Stylistic Anchors

- Targets 200+ Indian companies across all sectors
- Indian-specific URL patterns (.in domains, /privacy-policy.html, etc.)
- TLS fingerprint spoofing (curl_cffi mimics Chrome 120)
- Playwright fallback with stealth.js injection
- Saves to ./indian-seeds/ (required by GAN forge)
- Resume logic (success.log)
- Expect ~60-70% success rate (Indian sites have aggressive bot protection)
"""

import os
import asyncio
import time
from urllib.parse import urljoin

from curl_cffi.requests import AsyncSession
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from readability import Document
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

OUTPUT_DIR = "./indian-seeds"
LOG_FILE = "./indian_seeds_success.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 200+ INDIAN COMPANIES (Comprehensive, All Sectors)
# ═══════════════════════════════════════════════════════════════════════════
INDIAN_COMPANIES = [
    # ── Tech & IT Services ──
    "tcs.com", "infosys.com", "wipro.com", "hcltech.com", "techmahindra.com",
    "mindtree.com", "ltimindtree.com", "mphasis.com", "hexaware.com", "persistent.com",
    "coforge.com", "cyient.com", "tatatechnologies.com", "lntinfotech.com",
    "zoho.com", "freshworks.com", "postman.com", "razorpay.com",
    
    # ── E-commerce & Retail ──
    "flipkart.com", "amazon.in", "meesho.com", "myntra.com", "ajio.com",
    "nykaa.com", "tatacliq.com", "shoppersstop.com", "reliancedigital.in",
    "croma.com", "vijaysales.com", "bigbasket.com", "blinkit.com", "swiggy.com",
    "zomato.com", "dunzo.com", "jio.com", "airtel.in",
    "vodafoneidea.com", "bsnl.co.in", "paytm.com", "phonepe.com", "gpay.in",
    "cred.club", "upstox.com", "groww.in", "zerodha.com", "angelone.in",
    
    # ── Banking & Finance ──
    "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com", "sbi.co.in",
    "pnbindia.in", "bankofbaroda.in", "canarabank.com", "unionbankofindia.co.in",
    "yesbank.in", "idfcfirstbank.com", "federalbank.co.in", "indusind.com",
    "rblbank.com", "bandhanbank.com", "aubank.in", "equitasbank.com",
    "bajajfinserv.in", "bajajfinance.com", "hdfclife.com", "icicilombard.com",
    "sbilife.co.in", "maxlifeinsurance.com", "tataaig.com", "reliancelife.com",
    
    # ── Conglomerates & Manufacturing ──
    "ril.com", "tata.com", "mahindra.com", "adityabirla.com", "godrej.com",
    "larsentoubro.com", "birlacorp.com", "ultratechcement.com", "jspl.com",
    "tatasteel.com", "jswsteel.com", "sail.co.in", "vedantalimited.com",
    "hindalco.com", "adani.com", "adaniports.com", "adanienergy.com",
    
    # ── Automotive ──
    "tatamotors.com", "marutisuzuki.com", "mahindraandmahindra.com",
    "hyundai.co.in", "kia.com/in", "mgmotor.in", "toyotabharat.com",
    "honda.co.in", "volkswagen.co.in", "skoda-auto.co.in",
    "renault.co.in", "nissan.in", "fordindia.com", "bajajauto.com",
    "heromotocorp.com", "eichermotors.com", "ashokleyland.com",
    
    # ── Telecom & Media ──
    "hotstar.com", "sonyliv.com", "zee5.com", "voot.com", "mxplayer.in",
    "jiocinema.com", "timesofindia.com", "indiatoday.in", "hindustantimes.com",
    "ndtv.com", "thehindu.com", "indianexpress.com", "firstpost.com",
    "moneycontrol.com", "livemint.com", "businesstoday.in",
    "economictimes.indiatimes.com",
    
    # ── Travel & Hospitality ──
    "makemytrip.com", "goibibo.com", "yatra.com", "cleartrip.com", "ixigo.com",
    "oyo.com", "airbnb.in", "goindigo.in", "airindia.in",
    "spicejet.com", "vistara.com", "airasia.in",
    "irctc.co.in", "redbus.in", "abhibus.com",
    
    # ── Healthcare & Pharma ──
    "tatahealth.com", "apollohospitals.com", "fortishealthcare.com",
    "maxhealthcare.com", "medanta.org", "manipalhospitals.com",
    "practo.com", "1mg.com", "pharmeasy.in", "netmeds.com", "medplushealth.com",
    "sunpharma.com", "cipla.com", "drreddys.com", "lupin.com",
    "aurobindo.com", "divislabs.com", "glindia.com", "zydus.com",
    
    # ── Education & EdTech ──
    "byjus.com", "vedantu.com", "unacademy.com", "upgrad.com", "simplilearn.com",
    "coursera.org", "udemy.com", "greatlearning.in", "physicswallah.com",
    "iitb.ac.in", "iitd.ac.in", "iima.ac.in", "iisc.ac.in", "isb.edu",
    "bits-pilani.ac.in", "vit.ac.in", "manipal.edu", "amity.edu",
    
    # ── Real Estate & Construction ──
    "lodha.com", "godrejproperties.com", "oberoirealty.com", "dlf.in",
    "macrotechdevelopers.com", "puravankara.com", "sobha.com", "prestigeconstructions.com",
    "brigadegroup.com", "totalenvironment.com", "assethomes.com",
    
    # ── FMCG & Consumer Goods ──
    "hul.co.in", "itcportal.com", "dabur.com", "marico.com", "godrejconsumer.com",
    "patanjaliayurved.net", "emami.net", "colgatepalmolive.co.in",
    "unilever.com", "nestle.in", "pepsicoindia.co.in",
    
    # ── Energy & Utilities ──
    "ongcindia.com", "iocl.com", "bpcl.in", "hpcl.in", "gailonline.com",
    "tatapower.com", "nhpc.in", "powergrid.in", "nptc.in",
    
    # ── Logistics & Supply Chain ──
    "delhivery.com", "bluedart.com", "dtdc.com", "ecomexpress.in",
    "xpressbees.com", "shadowfax.in", "rivigo.com", "blackbuck.com",
    "allcargo.com", "gati.com",
    
    # ── Gaming & Entertainment ──
    "dream11.com", "my11circle.com", "games24x7.com", "mpl.live",
    "winzo.com", "headout.com", "bookmyshow.com",
    "ticketgenie.in", "eventshigh.com",
    
    # ── Government & Public Sector ──
    "india.gov.in", "mygov.in", "uidai.gov.in", "digilocker.gov.in",
    "npci.org.in", "bharatpay.in", "sebi.gov.in", "rbi.org.in",
    "bseindia.com", "nseindia.com", "indianrailways.gov.in",
    
    # ── Startups & Unicorns (duplicates removed) ──
    "browserstack.com", "druva.com", "freshdesk.com",
    
    # ── Insurance (additional) ──
    "licindia.in", "bajajallianz.com", "starhealth.in", "nivabupa.com",
    "careinsurance.com", "universalsompo.com", "rahejaqbe.com",
]
# ═══════════════════════════════════════════════════════════════════════════
# INDIAN-SPECIFIC URL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
PATH_PATTERNS = [
    "/privacy-policy",
    "/privacy-policy.html",
    "/privacy",
    "/privacy.html",
    "/legal/privacy",
    "/legal/privacy-policy",
    "/legal/privacy-policy.html",
    "/en/privacy-policy",
    "/en/privacy-policy.html",
    "/in/privacy-policy",
    "/in/privacy-policy.html",
    "/about/privacy-policy",
    "/about-us/privacy-policy",
    "/policies/privacy",
    "/site/privacy-policy",
    "/content/privacy-policy",
    "/legal/privacypolicy",
    "/privacy_policy",
    "/privacy_policy.html",
    "/data-protection",
    "/data-privacy",
    "/personal-data-protection",
]

# ═══════════════════════════════════════════════════════════════════════════
# RESUME LOGIC
# ═══════════════════════════════════════════════════════════════════════════
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        COMPLETED_DOMAINS = set(f.read().splitlines())
else:
    COMPLETED_DOMAINS = set()

REMAINING = [d for d in INDIAN_COMPANIES if d not in COMPLETED_DOMAINS]
print(f"📋 {len(COMPLETED_DOMAINS)} already harvested. {len(REMAINING)} Indian domains remaining.\n")

# ═══════════════════════════════════════════════════════════════════════════
# CLEANING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def clean_html(html_content):
    """Extract readable text, return None if too short."""
    doc = Document(html_content)
    summary = doc.summary()
    soup = BeautifulSoup(summary, 'html.parser')
    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean = '\n'.join(lines)
    if len(clean) > 2000:
        return clean
    return None

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: STEALTH HTTP (TLS spoofed)
# ═══════════════════════════════════════════════════════════════════════════
async def try_stealth_http(domain):
    base = f"https://{domain}" if not domain.startswith('http') else domain
    async with AsyncSession(impersonate="chrome120") as session:
        for path in PATH_PATTERNS:
            url = urljoin(base, path)
            try:
                resp = await session.get(url, timeout=8, allow_redirects=True)
                if resp.status_code == 200:
                    cleaned = clean_html(resp.text)
                    if cleaned:
                        return url, cleaned
            except Exception:
                continue
    return None, None

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: STEALTH PLAYWRIGHT (with JS evasion)
# ═══════════════════════════════════════════════════════════════════════════
async def try_stealth_playwright(domain, browser):
    base = f"https://{domain}" if not domain.startswith('http') else domain
    context = await browser.new_context()
    page = await context.new_page()
    await stealth_async(page)

    # Block useless resources to speed up
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
                cleaned = clean_html(raw_html)
                if cleaned:
                    await context.close()
                    return url, cleaned
        except Exception:
            continue
    await context.close()
    return None, None

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN HARVESTER
# ═══════════════════════════════════════════════════════════════════════════
async def harvest_domain(domain, browser, semaphore, pbar):
    async with semaphore:
        try:
            url, text = await try_stealth_http(domain)
            if not text:
                url, text = await try_stealth_playwright(domain, browser)

            if text:
                fname = os.path.join(OUTPUT_DIR, f"{domain.replace('.', '_')}.txt")
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(text)
                with open(LOG_FILE, 'a') as f:
                    f.write(domain + '\n')
        finally:
            pbar.update(1)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    print(f"🚀 Starting Indian Seeds Harvester on {len(REMAINING)} domains...")
    semaphore = asyncio.Semaphore(15)  # concurrency

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            executable_path='/usr/bin/brave-browser'  # adjust if needed
        )

        with tqdm(total=len(REMAINING), desc="Harvesting Indian Seeds") as pbar:
            tasks = [harvest_domain(d, browser, semaphore, pbar) for d in REMAINING]
            await asyncio.gather(*tasks)

    print(f"✅ Indian seeds harvest complete. Policies saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    asyncio.run(main())