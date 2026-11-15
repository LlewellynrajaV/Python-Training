import os
import hmac
import hashlib
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

load_dotenv()
API_KEY = os.getenv("API_KEY", "default_key")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default_secret")
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

try:
    response = session.get("https://jsonplaceholder.typicode.com/users")
    users = response.json()
    for user in users[:5]:
        print(f"{user['id']} {user['name']}\n{user['email']}")

    first_user = users[0]
    city = first_user.get("address", {}).get("city", "Not Available")
    zip_code = first_user.get("address", {}).get("zipcode", "Not Available")
    company_name = first_user.get("company", {}).get("name", "Not Available")

    print(f"City: {city} | Zip: {zip_code} | Company: {company_name}")
except Exception as e:
    print(f"API Error: {e}")

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    computed = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)

secret = "my_secure_secret"
payload = b'{"event": "update"}'
signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

valid = verify_signature(payload, signature, secret)
tampered = verify_signature(b'{"event": "fake"}', signature, secret)
wrong_secret = verify_signature(payload, signature, "wrong_secret_key")

print(f"Valid: {valid}  Tampered: {tampered} | Wrong secret: {wrong_secret}")