"""Regression tests: read-only trials observe one committed snapshot per call.

Each of the five read-only trial endpoints (selection-plans,
cost-attributions, repeat-run-trials, reversal-plans, release-forecast) must
read the dataset caps, granted requests, reservations and candidate
validation rows from a single SQLite commit snapshot, so a ledger change
committing mid-request (a cap transfer, a reversal, a new admission) cannot
mix, say, a pre-transfer cap with post-transfer occupancy into one answer.
The next call must still see the later commit.
"""
from __future__ import annotations

import itertools
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-snapshot-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "snapshot.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dsnap-{name}-{next(_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


async def _submit(client, dataset_id: str, request_id: str,
                  epsilon: float, delta: float = 1e-8):
    return await client.post(
        f"/datasets/{dataset_id}/requests",
        json={"request_id": request_id, "query_type": "gaussian_mean",
              "epsilon": epsilon, "delta": delta},
    )


async def _inject_commit(dataset_id: str, request_id: str) -> None:
    """Commit a brand-new granted request straight into the ledger.

    Simulates a concurrent admission that commits *while* a read-only trial
    of the same dataset is still reading.
    """
    async with db.connect() as conn:
        await db.begin_immediate(conn)
        await db.insert_request(
            conn,
            dataset_id=dataset_id,
            request_id=request_id,
            query_type="injected",
            epsilon=0.3,
            delta=1e-8,
            status="granted",
            reason=None,
            epsilon_before=0.0,
            epsilon_after=0.3,
            rdp_best_order=8192.0,
            created_at="2026-09-26T00:00:00+00:00",
        )
        await conn.commit()


def _arm_mid_request_commit(monkeypatch, dataset_id: str, request_id: str) -> None:
    """Fire a one-shot committed write right after the trial's first read.

    Wraps ``db.require_dataset`` -- the first statement every read-only trial
    issues -- so the concurrent commit lands between the cap read and the
    request/reservation reads of the same request.
    """
    real_require_dataset = db.require_dataset
    fired = False

    async def wrapper(conn, ds_id):
        row = await real_require_dataset(conn, ds_id)
        nonlocal fired
        if not fired and ds_id == dataset_id:
            fired = True
            await _inject_commit(ds_id, request_id)
        return row

    monkeypatch.setattr(db, "require_dataset", wrapper)


def _projection(entries, target_delta=1e-6) -> float:
    _, eps, _ = rdp.evaluate_entries(entries, target_delta)
    return max(0.0, eps)


# ------------------------------------------------------------- db primitive --


async def test_read_snapshot_pins_one_commit_state(client):
    ds = _new_dataset_id("dblevel")
    await _register(client, ds, eps=1.0)

    async with db.read_snapshot() as conn:
        row = await db.get_dataset_row(conn, ds)
        assert row["epsilon_cap"] == 1.0
        # A concurrent writer commits while this snapshot is open: the open
        # snapshot keeps observing the pre-commit state ...
        async with db.connect() as writer:
            await db.begin_immediate(writer)
            await db.update_dataset_epsilon_cap(
                writer, dataset_id=ds, epsilon_cap=0.5
            )
            await writer.commit()
        row = await db.get_dataset_row(conn, ds)
        assert row["epsilon_cap"] == 1.0

    # ... while the next snapshot sees everything committed since.
    async with db.read_snapshot() as conn:
        row = await db.get_dataset_row(conn, ds)
        assert row["epsilon_cap"] == 0.5


# -------------------------------------------------------------- endpoints --


async def test_selection_plan_reads_single_snapshot(client, monkeypatch):
    ds = _new_dataset_id("sp")
    await _register(client, ds, eps=1.0)
    _arm_mid_request_commit(monkeypatch, ds, "req-injected")

    payload = {"candidates": [
        {"candidate_id": "c1", "epsilon": 0.5, "delta": 1e-8, "value": 5},
    ]}
    r = await client.post(f"/datasets/{ds}/selection-plans", json=payload)
    assert r.status_code == 200
    body = r.json()
    # Cap and committed occupancy both predate the injected commit.
    assert body["epsilon_cap"] == 1.0
    assert body["epsilon_committed"] == pytest.approx(0.0)
    assert body["selected_candidate_ids"] == ["c1"]

    # A fresh call sees the committed request.
    r = await client.post(f"/datasets/{ds}/selection-plans", json=payload)
    assert r.status_code == 200
    assert r.json()["epsilon_committed"] == pytest.approx(
        _projection([(0.3, 1e-8)])
    )


async def test_repeat_run_trial_reads_single_snapshot(client, monkeypatch):
    ds = _new_dataset_id("rr")
    await _register(client, ds, eps=1.0)
    _arm_mid_request_commit(monkeypatch, ds, "req-injected")

    payload = {"items": [{"epsilon": 0.1, "delta": 1e-8}], "max_runs": 10}
    r = await client.post(f"/datasets/{ds}/repeat-run-trials", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["epsilon_cap"] == 1.0
    assert body["epsilon_committed"] == pytest.approx(0.0)

    r = await client.post(f"/datasets/{ds}/repeat-run-trials", json=payload)
    assert r.status_code == 200
    assert r.json()["epsilon_committed"] == pytest.approx(
        _projection([(0.3, 1e-8)])
    )


async def test_release_forecast_reads_single_snapshot(client, monkeypatch):
    ds = _new_dataset_id("fc")
    await _register(client, ds, eps=1.0)
    _arm_mid_request_commit(monkeypatch, ds, "req-injected")

    deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    r = await client.get(
        f"/datasets/{ds}/budget/release-forecast",
        params={"deadline": deadline},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["initial"]["epsilon_consumed"] == pytest.approx(0.0)
    assert body["at_deadline"]["epsilon_consumed"] == pytest.approx(0.0)

    r = await client.get(
        f"/datasets/{ds}/budget/release-forecast",
        params={"deadline": deadline},
    )
    assert r.status_code == 200
    assert r.json()["initial"]["epsilon_consumed"] == pytest.approx(
        _projection([(0.3, 1e-8)])
    )


async def test_cost_attribution_reads_single_snapshot(client, monkeypatch):
    ds = _new_dataset_id("ca")
    await _register(client, ds, eps=1.0)
    r = await _submit(client, ds, "req-a", 0.2)
    assert r.status_code == 201
    _arm_mid_request_commit(monkeypatch, ds, "req-injected")

    payload = {"request_ids": ["req-a"]}
    r = await client.post(f"/datasets/{ds}/cost-attributions", json=payload)
    assert r.status_code == 200
    body = r.json()
    # The baseline excludes the injected commit: only req-a exists here.
    assert body["epsilon_baseline"] == pytest.approx(0.0)
    assert body["epsilon_full_group"] == pytest.approx(
        _projection([(0.2, 1e-8)])
    )

    r = await client.post(f"/datasets/{ds}/cost-attributions", json=payload)
    assert r.status_code == 200
    assert r.json()["epsilon_baseline"] == pytest.approx(
        _projection([(0.3, 1e-8)])
    )


async def test_reversal_plan_reads_single_snapshot(client, monkeypatch):
    ds = _new_dataset_id("rp")
    await _register(client, ds, eps=1.0)
    r = await _submit(client, ds, "req-a", 0.2)
    assert r.status_code == 201
    _arm_mid_request_commit(monkeypatch, ds, "req-injected")

    payload = {
        "item": {"epsilon": 0.1, "delta": 1e-8},
        "request_ids": ["req-a"],
    }
    r = await client.post(f"/datasets/{ds}/reversal-plans", json=payload)
    assert r.status_code == 200
    body = r.json()
    # Committed occupancy predates the injected commit: req-a alone.
    assert body["epsilon_committed"] == pytest.approx(
        _projection([(0.2, 1e-8)])
    )
    assert body["feasible"] is True

    r = await client.post(f"/datasets/{ds}/reversal-plans", json=payload)
    assert r.status_code == 200
    assert r.json()["epsilon_committed"] == pytest.approx(
        _projection([(0.2, 1e-8), (0.3, 1e-8)])
    )
