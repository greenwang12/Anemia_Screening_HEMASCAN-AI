import os
import io
import base64
import uuid
import pytest
import requests
from PIL import Image, ImageDraw, ImageFilter

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
assert BASE_URL, "REACT_APP_BACKEND_URL is required"
API = f"{BASE_URL}/api"

ADMIN_EMAIL = "admin@anemiacheck.app"
ADMIN_PASSWORD = "Admin@123"


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def api_url():
    return API


@pytest.fixture
def api_client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


@pytest.fixture(scope="session")
def admin_token():
    r = requests.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, timeout=30)
    if r.status_code != 200:
        pytest.skip(f"Admin login failed: {r.status_code} {r.text}")
    return r.json()["token"]


@pytest.fixture
def admin_client(admin_token):
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json", "Authorization": f"Bearer {admin_token}"})
    return s


@pytest.fixture(scope="session")
def user_token():
    email = f"TEST_nonadmin_{uuid.uuid4().hex[:8]}@example.com"
    r = requests.post(f"{API}/auth/register",
                      json={"name": "NonAdmin", "email": email, "password": "Passw0rd!"}, timeout=30)
    if r.status_code != 200:
        pytest.skip(f"Non-admin register failed: {r.status_code} {r.text}")
    return r.json()["token"]


@pytest.fixture
def user_client(user_token):
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json", "Authorization": f"Bearer {user_token}"})
    return s


def _make_tissue_image(color=(220, 120, 110), size=(256, 256)) -> str:
    """Create a textured pinkish JPEG with real visual features. Returns base64."""
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            shade = ((i * j) % 40) - 20
            r = max(0, min(255, color[0] + shade))
            g = max(0, min(255, color[1] + shade // 2))
            b = max(0, min(255, color[2] + shade // 2))
            draw.rectangle([i, j, i + 7, j + 7], fill=(r, g, b))
    draw.ellipse([40, 60, 220, 200], outline=(150, 60, 60), width=3)
    draw.line([(10, 10), (240, 240)], fill=(120, 60, 60), width=2)
    draw.line([(240, 10), (10, 240)], fill=(120, 60, 60), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture(scope="session")
def sample_eye_image_b64():
    return _make_tissue_image(color=(210, 110, 110))


@pytest.fixture(scope="session")
def sample_nail_image_b64():
    return _make_tissue_image(color=(230, 170, 160))


@pytest.fixture(scope="session")
def blurry_image_b64():
    """Heavily blurred + low-quality JPEG to fail the Laplacian variance check."""
    img = Image.new("RGB", (256, 256), (200, 130, 130))
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 80, 200, 200], fill=(180, 100, 100))
    # Heavy gaussian blur to crush edges
    img = img.filter(ImageFilter.GaussianBlur(radius=18))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=8)
    return base64.b64encode(buf.getvalue()).decode("ascii")
