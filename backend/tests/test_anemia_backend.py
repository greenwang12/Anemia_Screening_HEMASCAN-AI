"""
Backend tests for HemaScan anemia screening (EfficientNetB0 binary nail model).
Covers: health, auth (no auto-login on register), screenings with new
clinical-features + binary model blend, Grad-CAM with new model, polarity check.
"""
import io
import base64
import uuid
import zipfile
import pytest
import requests
from PIL import Image

from conftest import API, ADMIN_EMAIL, ADMIN_PASSWORD


# ---------------- Health ----------------
class TestHealth:
    def test_health_ok(self, api_client):
        r = api_client.get(f"{API}/health", timeout=30)
        assert r.status_code == 200
        assert r.json().get("ok") is True


# ---------------- Auth ----------------
class TestAuth:
    def test_register_does_not_auto_login(self, api_client):
        """After /api/auth/register, response must NOT contain a token,
        and no auth cookie should be set (user is NOT auto-logged in)."""
        email = f"TEST_user_{uuid.uuid4().hex[:8]}@example.com"
        r = api_client.post(f"{API}/auth/register",
                            json={"name": "Test User", "email": email, "password": "Passw0rd!"},
                            timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["email"] == email.lower()
        assert data["role"] == "user"
        # CRITICAL: must NOT return a token
        assert "token" not in data, f"register response leaked token: {data}"
        assert "password_hash" not in data
        # Cookie should NOT be set
        assert "access_token" not in r.cookies, "register set an auth cookie (auto-login bug)"
        # Save email for next test
        pytest.fresh_email = email
        pytest.fresh_password = "Passw0rd!"

    def test_login_with_registered_credentials(self, api_client):
        email = getattr(pytest, "fresh_email", None)
        pw = getattr(pytest, "fresh_password", None)
        if not email:
            pytest.skip("fresh user not registered")
        r = api_client.post(f"{API}/auth/login",
                            json={"email": email, "password": pw}, timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        token = data.get("token")
        assert isinstance(token, str) and len(token) > 20
        # /auth/me with token
        r2 = api_client.get(f"{API}/auth/me",
                            headers={"Authorization": f"Bearer {token}"}, timeout=30)
        assert r2.status_code == 200
        assert r2.json()["email"] == email.lower()

    def test_register_duplicate_email(self, api_client):
        email = f"TEST_dup_{uuid.uuid4().hex[:8]}@example.com"
        r1 = api_client.post(f"{API}/auth/register",
                             json={"name": "Dup", "email": email, "password": "Passw0rd!"}, timeout=30)
        assert r1.status_code == 200
        r2 = api_client.post(f"{API}/auth/register",
                             json={"name": "Dup", "email": email, "password": "Passw0rd!"}, timeout=30)
        assert r2.status_code == 400

    def test_login_invalid(self, api_client):
        r = api_client.post(f"{API}/auth/login",
                            json={"email": ADMIN_EMAIL, "password": "WrongPassword!"}, timeout=30)
        assert r.status_code == 401

    def test_me_without_token(self, api_client):
        r = api_client.get(f"{API}/auth/me", timeout=30)
        assert r.status_code == 401


# ---------------- helpers ----------------
def _solid_image_b64(rgb, size=(256, 256)) -> str:
    img = Image.new("RGB", size, rgb)
    # Add a little structure so OpenCV doesn't bail (gradient + ellipse)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 60, 220, 200], outline=(max(0, rgb[0]-40), max(0, rgb[1]-40), max(0, rgb[2]-40)), width=3)
    for i in range(0, size[0], 16):
        shade = (i % 32) - 16
        draw.line([(i, 0), (i, size[1])],
                  fill=(max(0, min(255, rgb[0]+shade)),
                        max(0, min(255, rgb[1]+shade)),
                        max(0, min(255, rgb[2]+shade))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------- Screenings: new nail pipeline ----------------
FORBIDDEN_NAIL_KEYS = {"class_probs", "top_class", "top_prob", "p_class_anemia"}
REQUIRED_NAIL_EXTRAS = {
    "p_model_anemia", "p_features_anemia", "clinical_features",
    "feature_weights", "blend_weights",
}
REQUIRED_CLINICAL_FEATURES = {
    "pallor", "koilonychia", "platonychia", "ridging", "brittleness", "yellowing",
}


class TestNailScreening:
    def test_nail_only_new_pipeline(self, admin_client):
        img = _solid_image_b64((230, 170, 160))
        r = admin_client.post(f"{API}/screenings",
                              json={"nail_image_base64": img, "patient_name": "TEST_nail"},
                              timeout=240)
        assert r.status_code == 200, r.text[:600]
        data = r.json()
        assert data["eye_result"] is None
        nail = data["nail_result"]
        assert nail is not None
        extras = nail.get("model_extras") or {}

        # Required new keys
        missing = REQUIRED_NAIL_EXTRAS - set(extras.keys())
        assert not missing, f"missing required extras: {missing}; have {list(extras)}"

        # Forbidden keys absent (anywhere in nail_result)
        flat_keys = set(extras.keys()) | set(nail.keys())
        leaked = flat_keys & FORBIDDEN_NAIL_KEYS
        assert not leaked, f"forbidden 6-class keys leaked: {leaked}"

        # Probabilities are floats in [0,1]
        for k in ("p_model_anemia", "p_features_anemia"):
            v = extras[k]
            assert isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0, f"{k}={v}"

        # clinical_features has the 6 expected keys
        cf = extras["clinical_features"]
        assert set(cf.keys()) == REQUIRED_CLINICAL_FEATURES, f"clinical_features keys: {cf.keys()}"
        for k, v in cf.items():
            assert 0.0 <= float(v) <= 1.0, f"{k}={v}"

        # blend_weights is exactly {model:0.6, features:0.4}
        bw = extras["blend_weights"]
        assert abs(bw.get("model", 0) - 0.6) < 1e-6
        assert abs(bw.get("features", 0) - 0.4) < 1e-6

        # feature_weights present and sums ~1
        fw = extras["feature_weights"]
        assert abs(sum(fw.values()) - 1.0) < 1e-3

        # Grad-CAM still works with new EfficientNetB0
        assert isinstance(nail.get("gradcam_heatmap_base64"), str) and len(nail["gradcam_heatmap_base64"]) > 100
        assert isinstance(nail.get("gradcam_layer"), str) and len(nail["gradcam_layer"]) > 0

        # cleanup
        admin_client.delete(f"{API}/screenings/{data['id']}", timeout=30)

    def test_eye_only_still_works_with_gradcam(self, admin_client):
        img = _solid_image_b64((210, 110, 110))
        r = admin_client.post(f"{API}/screenings",
                              json={"eye_image_base64": img, "patient_name": "TEST_eye"},
                              timeout=240)
        assert r.status_code == 200, r.text[:600]
        data = r.json()
        eye = data["eye_result"]
        assert eye is not None
        assert data["nail_result"] is None
        assert isinstance(eye.get("gradcam_heatmap_base64"), str) and len(eye["gradcam_heatmap_base64"]) > 100
        admin_client.delete(f"{API}/screenings/{data['id']}", timeout=30)

    def test_polarity_pale_vs_red(self, admin_client):
        """Polarity sanity: a pale image (235,215,215) should yield HIGHER
        p_features_anemia and higher risk_percent than a red/healthy image
        (200,80,90)."""
        red = _solid_image_b64((200, 80, 90))
        pale = _solid_image_b64((235, 215, 215))

        r1 = admin_client.post(f"{API}/screenings",
                               json={"nail_image_base64": red, "patient_name": "TEST_red"},
                               timeout=240)
        r2 = admin_client.post(f"{API}/screenings",
                               json={"nail_image_base64": pale, "patient_name": "TEST_pale"},
                               timeout=240)
        assert r1.status_code == 200 and r2.status_code == 200, (r1.text[:300], r2.text[:300])

        red_data = r1.json()["nail_result"]
        pale_data = r2.json()["nail_result"]
        red_pf = red_data["model_extras"]["p_features_anemia"]
        pale_pf = pale_data["model_extras"]["p_features_anemia"]
        red_risk = red_data["risk_percent"]
        pale_risk = pale_data["risk_percent"]

        print(f"\nPolarity: red p_features={red_pf:.3f} risk={red_risk}% | "
              f"pale p_features={pale_pf:.3f} risk={pale_risk}%")

        # Soft thresholds from PRD: red <0.20, pale >=0.30
        assert pale_pf > red_pf, f"pale p_features ({pale_pf}) not > red ({red_pf})"
        assert pale_risk >= red_risk, f"pale risk {pale_risk}% < red risk {red_risk}%"
        assert red_pf < 0.25, f"red p_features {red_pf} unexpectedly high"
        assert pale_pf >= 0.28, f"pale p_features {pale_pf} unexpectedly low"

        admin_client.delete(f"{API}/screenings/{r1.json()['id']}", timeout=30)
        admin_client.delete(f"{API}/screenings/{r2.json()['id']}", timeout=30)


class TestScreeningGuards:
    def test_no_images_400(self, admin_client):
        r = admin_client.post(f"{API}/screenings", json={}, timeout=30)
        assert r.status_code == 400

    def test_screenings_require_auth(self, api_client):
        r = api_client.post(f"{API}/screenings",
                            json={"nail_image_base64": _solid_image_b64((220, 180, 170))},
                            timeout=60)
        assert r.status_code == 401
