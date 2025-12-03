import re
import socket
import urllib.parse
import requests
from bs4 import BeautifulSoup
import whois
from datetime import datetime, timezone

def extract_features(url):
    features = []

    # Parse domain
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc
    now = datetime.now(timezone.utc)

    # Feature 1: Having IP address
    try:
        socket.inet_aton(url)
        features.append(1)
    except:
        features.append(-1)

    # Feature 2: URL Length
    features.append(1 if len(url) >= 75 else 0 if len(url) >= 54 else -1)

    # Feature 3: Shortening service
    shortening_services = r"bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|buff\.ly"
    features.append(1 if re.search(shortening_services, url) else -1)

    # Feature 4: '@' symbol
    features.append(1 if '@' in url else -1)

    # Feature 5: Redirecting using '//'
    features.append(1 if url.count('//') > 1 else -1)

    # Feature 6: Prefix/Suffix with '-'
    features.append(1 if '-' in domain else -1)

    # Feature 7: Subdomain count
    subdomains = domain.split('.')
    features.append(1 if len(subdomains) > 3 else 0 if len(subdomains) == 3 else -1)

    # Feature 8: SSL final state
    features.append(1 if url.startswith("https") else -1)

    # Feature 9: Domain registration length
    domain_info = None
    try:
        domain_info = whois.whois(domain)
        expiration_date = domain_info.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if expiration_date:
            registration_length = (expiration_date - now).days
            features.append(1 if registration_length > 365 else -1)
        else:
            print("WHOIS warning: No expiration date")
            features.append(0)
    except Exception as e:
        print("WHOIS error:", e)
        features.append(0)

    # Feature 10–24: HTML-based features
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")

        # Feature 10: Favicon presence
        favicon = soup.find("link", rel=lambda x: x and "icon" in x.lower())
        features.append(1 if favicon else -1)

        # Feature 11: Port
        features.append(1 if ":443" in url or ":80" in url else -1)

        # Feature 12: HTTPS token in domain
        features.append(1 if "https" in domain else -1)

        # Feature 13–20: Placeholders
        features.extend([-1] * 8)

        # Feature 21: onmouseover
        features.append(1 if "onmouseover" in response.text else -1)

        # Feature 22: RightClick disabled
        features.append(1 if "event.button==2" in response.text else -1)

        # Feature 23: Pop-up window
        features.append(1 if "window.open" in response.text else -1)

        # Feature 24: Iframe usage
        iframes = soup.find_all("iframe")
        features.append(1 if len(iframes) > 0 else -1)
    except Exception as e:
        print("Request error:", e)
        features.extend([-1] * 15)  # Covers 10–24 if request fails

    # Feature 25: Age of domain
    try:
        if domain_info:
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date:
                age = (now - creation_date).days
                features.append(1 if age > 180 else -1)
            else:
                print("WHOIS warning: No creation date")
                features.append(0)  # neutral fallback
        else:
          raise ValueError("No WHOIS data")
    except Exception as e:
        print("WHOIS error (creation date):", e)
        features.append(0)  # fallback if WHOIS fails entirely

    # Feature 26: DNS resolution
    try:
        socket.gethostbyname(domain)
        features.append(1)
    except:
        features.append(-1)

    # Feature 27: Web traffic (placeholder)
    # features.append(-1)
    def get_web_traffic(domain):
        try:
            # Use Alexa or similar API (or cached values)
            rank = get_rank(domain)  # hypothetical function
            return 1 if rank < 100000 else 0 if rank < 500000 else -1
        except:
            return 0  # neutral fallback
    features.append(get_web_traffic(domain))

    # Feature 28: Page rank (placeholder)
    # features.append(-1)
    def get_page_rank(domain):
        try:
            # Use Moz, Ahrefs, or cached heuristic
            score = get_rank_score(domain)  # hypothetical
            return 1 if score > 5 else 0 if score > 2 else -1
        except:
            return 0
    features.append(get_page_rank(domain))

    # Feature 29: Google index
    # features.append(1 if "google.com" in url else -1)
    # features.append(0)  # Placeholder for Google index status
    try:
        from googlesearch import search
        site = search(url, 5)
        features.append(1 if site else -1)
    except:
        features.append(0)

    # Feature 30: Phishing keyword detection
    phishing_keywords = ["secure", "login", "verify", "update", "account", "bank", "alert"]
    # features.append(1 if any(kw in url.lower() for kw in phishing_keywords) else -1)
    keyword_hits = sum(1 for kw in phishing_keywords if kw in url.lower())
    features.append(1 if keyword_hits >= 2 else 0 if keyword_hits == 1 else -1)

    # Final padding (if needed)
    while len(features) < 30:
        features.append(0)

    # ✅ Log how many features failed
    print("Missing features:", features.count(-1))
    features = [0 if f == -1 else f for f in features]

    
    assert len(features) == 30, f"Feature count mismatch: {len(features)}"

    return features