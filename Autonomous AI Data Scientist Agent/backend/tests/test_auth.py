import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import auth, store
from app.main import _dist, app


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_db_path = auth.DB_PATH
        self.old_storage = store.STORAGE
        self.old_auth_storage = auth.DATASET_STORAGE
        root = Path(self.temp_dir.name)
        auth.DB_PATH = root / "auth.sqlite3"
        store.STORAGE = root / "datasets"
        store.STORAGE.mkdir()
        auth.DATASET_STORAGE = store.STORAGE
        auth.init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        auth.DB_PATH = self.old_db_path
        store.STORAGE = self.old_storage
        auth.DATASET_STORAGE = self.old_auth_storage
        self.temp_dir.cleanup()

    def register(self, client, email="person@example.com"):
        return client.post(
            "/api/auth/register",
            headers={"Origin": "http://127.0.0.1:5173"},
            json={
                "name": "Test Person",
                "email": email,
                "password": "a-long-test-password",
            },
        )

    def test_password_hash_is_not_the_password(self):
        encoded = auth._password_hash("correct horse battery staple")
        self.assertNotIn("correct horse", encoded)
        self.assertTrue(auth.verify_password("correct horse battery staple", encoded))
        self.assertFalse(auth.verify_password("incorrect password", encoded))

    def test_authentication_and_dataset_owner_isolation(self):
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(self.client.get("/api/datasets/aaaaaaaaaaaa").status_code, 401)

        registered = self.register(self.client)
        self.assertEqual(registered.status_code, 201)
        cookie = registered.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)

        uploaded = self.client.post(
            "/api/upload",
            files={"file": ("sample.csv", b"name,value\na,1\nb,2\n", "text/csv")},
        )
        self.assertEqual(uploaded.status_code, 200)
        dataset_id = uploaded.json()["meta"]["id"]
        self.assertEqual(self.client.get(f"/api/datasets/{dataset_id}").status_code, 200)

        other_client = TestClient(app)
        try:
            self.assertEqual(self.register(other_client, "other@example.com").status_code, 201)
            self.assertEqual(other_client.get(f"/api/datasets/{dataset_id}").status_code, 404)
        finally:
            other_client.close()

        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(
            self.client.post("/api/auth/login", json={
                "email": "person@example.com",
                "password": "a-long-test-password",
            }, headers={"Origin": "http://127.0.0.1:5173"}).status_code,
            200,
        )
        self.assertEqual(self.client.get(f"/api/datasets/{dataset_id}").status_code, 200)

    def test_mutation_rejects_untrusted_browser_origins(self):
        response = self.client.post(
            "/api/auth/login",
            headers={"Origin": "https://untrusted.example"},
            json={"email": "person@example.com", "password": "wrong"},
        )
        self.assertEqual(response.status_code, 403)

    def test_password_change_validates_current_password_and_revokes_other_sessions(self):
        self.assertEqual(self.register(self.client).status_code, 201)
        other_client = TestClient(app)
        try:
            login = other_client.post(
                "/api/auth/login",
                json={"email": "person@example.com", "password": "a-long-test-password"},
            )
            self.assertEqual(login.status_code, 200)

            wrong_password = self.client.post(
                "/api/auth/change-password",
                json={"current_password": "incorrect-password", "new_password": "another-long-password"},
            )
            self.assertEqual(wrong_password.status_code, 400)

            same_password = self.client.post(
                "/api/auth/change-password",
                json={"current_password": "a-long-test-password", "new_password": "a-long-test-password"},
            )
            self.assertEqual(same_password.status_code, 400)

            changed = self.client.post(
                "/api/auth/change-password",
                json={"current_password": "a-long-test-password", "new_password": "another-long-password"},
            )
            self.assertEqual(changed.status_code, 200)
            self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
            self.assertEqual(other_client.get("/api/auth/me").status_code, 401)

            old_password_login = other_client.post(
                "/api/auth/login",
                json={"email": "person@example.com", "password": "a-long-test-password"},
            )
            self.assertEqual(old_password_login.status_code, 401)
            new_password_login = other_client.post(
                "/api/auth/login",
                json={"email": "person@example.com", "password": "another-long-password"},
            )
            self.assertEqual(new_password_login.status_code, 200)
        finally:
            other_client.close()

    def test_production_routes_fall_back_to_the_spa(self):
        if not _dist.is_dir():
            self.skipTest("frontend production build is unavailable")

        response = self.client.get("/login", headers={"Accept": "text/html"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
