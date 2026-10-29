import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["dept_projects"] >= 10


def test_duplicate_idea_scores_low_novelty():
    r = client.post("/api/check", json={
        "title": "Face Recognition Attendance System",
        "abstract": "A camera recognises student faces and automatically marks classroom attendance, replacing manual roll calls.",
        "check_external": False})
    d = r.json()
    assert d["novelty_score"] < 70, d
    assert d["closest_matches"][0]["source"] == "Department"


def test_fresh_idea_scores_higher():
    r = client.post("/api/check", json={
        "title": "Beehive Health Monitoring with Acoustic Sensors",
        "abstract": "Microphones inside beehives and a model that detects queen loss and disease from buzzing patterns to alert beekeepers early.",
        "check_external": False})
    d = r.json()
    assert d["novelty_score"] > 40, d
