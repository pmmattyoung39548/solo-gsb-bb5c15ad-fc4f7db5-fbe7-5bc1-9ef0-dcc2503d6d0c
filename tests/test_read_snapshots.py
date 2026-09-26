"""Concurrency regression tests for the read-only trial endpoints.

A read-only trial reads the dataset cap, the granted/non-reversed requests and
the currently valid reservations through several SELECTs.  In WAL mode every
SELECT outside an explicit transaction starts against the latest committed
snapshot, so a reversal or a cap transfer committing *between* two of those
reads used to mix commit states (old cap with new occupancy) and produced an
unexecutable plan.

These tests pin the exact race deterministically: a gate is entered right
after the trial's first read (which, inside the fixed read transaction, is
what establishes the SQLite snapshot), a writer then commits a reversal plus
an epsilon-cap transfer, and only then is the trial allowed to finish its
remaining reads.  A correct answer must describe the *old* committed ledger
end to end; the very next call must describe the *new* one.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-snapshots-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "snapshots.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, ledger  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dssn-{name}-{next(_counter)}"


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


@pytest_asyncio.fixture()
async def changing_ledger(client):
    """A dataset whose cap drops while a 0.8-epsilon request is reversed.

    Returns ``(source_ds, target_ds, tag)``.  Before the mid-trial commit the
    source dataset has cap 1.0 and one granted pure-DP request ``big`` worth
    0.8 epsilon.  The writer reverses ``big`` (occupancy becomes 0) and
    transfers 0.6 epsilon cap away (cap becomes 0.4, still positive and
    covering the now-empty occupancy); the target dataset receives the cap.
    """
    tag = next(_counter)
    source = f"dssn-src-{tag}"
    target = f"dssn-dst-{tag}"
    await _register(client, source, eps=1.0, delta=1e-6)
    await _register(client, target, eps=1.0, delta=1e-6)
    r = await client.post(
        f"/datasets/{source}/requests",
        json={"request_id": "big", "query_type": "q",
              "epsilon": 0.8, "delta": 0.0},
    )
    assert r.status_code == 201 and r.json()["allowed"]
    return source, target, tag


class _FirstReadGate:
    """Block inside require_dataset after the trial dataset's first read.

    The trial endpoints read the dataset row first; inside the fixed read
    transaction that SELECT establishes the shared SQLite snapshot.  The gate
    fires once for the trial dataset, lets the test driver commit a ledger
    change, and only then releases the trial to perform its remaining reads.
    """

    def __init__(self, monkeypatch, trial_dataset: str):
        self._real = db.require_dataset
        self._trial_dataset = trial_dataset
        self.entered = asyncio.Event()
        self.proceed = asyncio.Event()
        self._fired = False
        monkeypatch.setattr(db, "require_dataset", self._wrapped)
        monkeypatch.setattr(ledger.db, "require_dataset", self._wrapped)

    async def _wrapped(self, conn, dataset_id: str):
        row = await self._real(conn, dataset_id)
        if not self._fired and dataset_id == self._trial_dataset:
            self._fired = True
            self.entered.set()
            await asyncio.wait_for(self.proceed.wait(), timeout=10)
        return row

    async def commit_change_then_release(self, client, source: str, target: str):
        """Reverse the 0.8 request and move 0.6 cap out, then release the trial."""
        await asyncio.wait_for(self.entered.wait(), timeout=10)
        rev = await client.post(
            f"/datasets/{source}/requests/big/reversals",
            json={"reversal_id": f"rev-{source}", "reason": "mid-trial cleanup"},
        )
        assert rev.status_code == 201
        xfer = await client.post(
            "/budget-transfers",
            json={
                "transfer_id": f"xfer-{source}",
                "source_dataset_id": source,
                "target_dataset_id": target,
                "epsilon_amount": 0.6,
            },
        )
        assert xfer.status_code == 201 and xfer.json()["allowed"]
        assert xfer.json()["source"]["epsilon_cap"] == pytest.approx(0.4)
        self.proceed.set()


# ------------------------------------------------------------ selection plan --


@pytest.mark.asyncio
async def test_selection_plan_uses_one_commit_snapshot(
    client, changing_ledger, monkeypatch
):
    source, target, _ = changing_ledger
    gate = _FirstReadGate(monkeypatch, source)
    candidates = [
        {"candidate_id": "c", "epsilon": 0.5, "delta": 0.0, "value": 1}
    ]

    trial = asyncio.create_task(
        client.post(f"/datasets/{source}/selection-plans",
                    json={"candidates": candidates})
    )
    await gate.commit_change_then_release(client, source, target)
    response = await asyncio.wait_for(trial, timeout=10)

    # The whole answer describes the pre-commit ledger: cap 1.0 still holds and
    # the 0.8 request still occupies it, so the 0.5 candidate (1.3 projected)
    # is infeasible.  Old cap + new occupancy would have wrongly admitted it.
    assert response.status_code == 200
    body = response.json()
    assert body["epsilon_cap"] == pytest.approx(1.0)
    assert body["epsilon_committed"] == pytest.approx(0.8)
    assert body["projected_epsilon"] == pytest.approx(0.8)
    assert body["selected_candidate_ids"] == []
    assert body["unselected_candidate_ids"] == ["c"]

    # The next call opens a fresh snapshot and sees the reversed request and
    # the 0.4 cap; the same candidate is infeasible against the new ledger.
    after = (
        await client.post(
            f"/datasets/{source}/selection-plans",
            json={"candidates": candidates},
        )
    ).json()
    assert after["epsilon_cap"] == pytest.approx(0.4)
    assert after["epsilon_committed"] == pytest.approx(0.0)
    assert after["selected_candidate_ids"] == []


# ------------------------------------------------------------ repeat-run trial --


@pytest.mark.asyncio
async def test_repeat_run_trial_uses_one_commit_snapshot(
    client, changing_ledger, monkeypatch
):
    source, target, _ = changing_ledger
    gate = _FirstReadGate(monkeypatch, source)
    payload = {"items": [{"epsilon": 0.5, "delta": 0.0}], "max_runs": 2}

    trial = asyncio.create_task(
        client.post(f"/datasets/{source}/repeat-run-trials", json=payload)
    )
    await gate.commit_change_then_release(client, source, target)
    response = await asyncio.wait_for(trial, timeout=10)

    # Pre-commit ledger: 0.8 committed, even one 0.5 run projects to 1.3 > 1.0.
    # Old cap with new occupancy would have wrongly reported two fitting runs.
    assert response.status_code == 200
    body = response.json()
    assert body["epsilon_cap"] == pytest.approx(1.0)
    assert body["epsilon_committed"] == pytest.approx(0.8)
    assert body["max_complete_runs"] == 0
    assert body["at_max_runs"] is False
    assert body["projection"]["runs"] == 0
    assert body["next_run_projection"]["runs"] == 1
    assert body["next_run_projection"]["epsilon_projected"] == pytest.approx(1.3)
    assert body["next_run_projection"]["feasible"] is False

    # The next call sees the new 0.4 cap (zero runs fit there too).
    after = (
        await client.post(
            f"/datasets/{source}/repeat-run-trials", json=payload
        )
    ).json()
    assert after["epsilon_cap"] == pytest.approx(0.4)
    assert after["epsilon_committed"] == pytest.approx(0.0)
    assert after["max_complete_runs"] == 0


# ------------------------------------------------------------ release forecast --


@pytest.mark.asyncio
async def test_release_forecast_uses_one_commit_snapshot(
    client, changing_ledger, monkeypatch
):
    source, target, _ = changing_ledger
    # A valid reservation locking another 0.2 (committed 0.8 + 0.2 == cap).
    r = await client.post(
        f"/datasets/{source}/reservations",
        json={"reservation_id": "res", "epsilon": 0.2, "delta": 0.0,
              "ttl_seconds": 3600},
    )
    assert r.status_code == 201 and r.json()["allowed"]

    gate = _FirstReadGate(monkeypatch, source)
    deadline = datetime.now(timezone.utc) + timedelta(hours=2)

    trial = asyncio.create_task(
        client.get(
            f"/datasets/{source}/budget/release-forecast",
            params={"deadline": deadline.isoformat()},
        )
    )
    await gate.commit_change_then_release(client, source, target)
    response = await asyncio.wait_for(trial, timeout=10)

    # Pre-commit snapshot: the 0.8 request is still consumed, the reservation
    # valid, and nothing is left of the 1.0 cap; after the simulated release
    # 0.2 is free.  New occupancy (reversed request) would have reported 0
    # consumed and 0.8 remaining instead.
    assert response.status_code == 200
    body = response.json()
    assert body["initial"]["epsilon_consumed"] == pytest.approx(0.8)
    assert body["initial"]["active_reservations"] == 1
    assert body["initial"]["epsilon_remaining"] == pytest.approx(0.0)
    assert len(body["events"]) == 1
    assert body["events"][0]["released_reservation_ids"] == ["res"]
    assert body["at_deadline"]["epsilon_consumed"] == pytest.approx(0.8)
    assert body["at_deadline"]["epsilon_remaining"] == pytest.approx(0.2)
    assert body["at_deadline"]["active_reservations"] == 0

    # Strictly read-only: the forecast never swept the reservation itself.
    async with db.connect() as conn:
        row = await db.get_reservation(conn, source, "res")
        assert row["status"] == "granted"
        assert row["expired_at"] is None

    # The next call sees the reversed request under the new 0.4 cap while the
    # still-valid reservation keeps locking its 0.2.
    after = (
        await client.get(
            f"/datasets/{source}/budget/release-forecast",
            params={"deadline": deadline.isoformat()},
        )
    ).json()
    assert after["initial"]["epsilon_consumed"] == pytest.approx(0.0)
    assert after["initial"]["active_reservations"] == 1
    assert after["initial"]["epsilon_remaining"] == pytest.approx(0.2)


# ------------------------------------------------------------ cost attribution --


@pytest.mark.asyncio
async def test_cost_attribution_uses_one_commit_snapshot(
    client, changing_ledger, monkeypatch
):
    source, target, _ = changing_ledger
    gate = _FirstReadGate(monkeypatch, source)

    trial = asyncio.create_task(
        client.post(
            f"/datasets/{source}/cost-attributions",
            json={"request_ids": ["big"]},
        )
    )
    await gate.commit_change_then_release(client, source, target)
    response = await asyncio.wait_for(trial, timeout=10)

    # Pre-commit snapshot: "big" is still an eligible granted request, so the
    # attribution succeeds under the 1.0 cap.  Reading the candidate set from
    # the post-reversal snapshot would instead 409 as already_reversed.
    assert response.status_code == 200
    body = response.json()
    assert body["epsilon_cap"] == pytest.approx(1.0)
    assert body["epsilon_baseline"] == pytest.approx(0.0)
    assert body["epsilon_full_group"] == pytest.approx(0.8)
    assert body["epsilon_increment"] == pytest.approx(0.8)
    assert body["attributed_total_epsilon"] == pytest.approx(0.8)
    assert body["attributions"][0]["request_id"] == "big"
    assert body["attributions"][0]["attributed_epsilon"] == pytest.approx(0.8)

    # The next call sees the reversal and rejects the same selection.
    after = await client.post(
        f"/datasets/{source}/cost-attributions",
        json={"request_ids": ["big"]},
    )
    assert after.status_code == 409
    assert after.json()["error"] == "request_already_reversed"


# ------------------------------------------------------------ reversal plan --


@pytest.mark.asyncio
async def test_reversal_plan_uses_one_commit_snapshot(
    client, changing_ledger, monkeypatch
):
    source, target, _ = changing_ledger
    gate = _FirstReadGate(monkeypatch, source)
    payload = {
        "item": {"epsilon": 0.5, "delta": 0.0},
        "request_ids": ["big"],
    }

    trial = asyncio.create_task(
        client.post(f"/datasets/{source}/reversal-plans", json=payload)
    )
    await gate.commit_change_then_release(client, source, target)
    response = await asyncio.wait_for(trial, timeout=10)

    # Pre-commit snapshot: the pending item needs "big" reversed (0.8 + 0.5 >
    # 1.0, but 0.5 fits afterwards).  A post-reversal candidate read would
    # have 409'd as request_already_reversed.
    assert response.status_code == 200
    body = response.json()
    assert body["epsilon_cap"] == pytest.approx(1.0)
    assert body["epsilon_committed"] == pytest.approx(0.8)
    assert body["feasible"] is True
    assert body["suggested_request_ids"] == ["big"]
    assert body["reversal_count"] == 1
    assert body["epsilon_after_reversal"] == pytest.approx(0.0)
    assert body["epsilon_with_pending"] == pytest.approx(0.5)

    # The next call sees the reversal: "big" is no longer a candidate.
    after = await client.post(
        f"/datasets/{source}/reversal-plans", json=payload
    )
    assert after.status_code == 409
    assert after.json()["error"] == "request_already_reversed"
