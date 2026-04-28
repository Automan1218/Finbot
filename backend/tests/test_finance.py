from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.main import app


@pytest.fixture
async def client(db_session, finance_setup):
    user, team, token = finance_setup

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=headers
    ) as c:
        yield c, team
    app.dependency_overrides.clear()


# ---------- Accounts ----------

@pytest.mark.asyncio
async def test_create_account(client):
    c, team = client
    tid = str(team.id)
    resp = await c.post(
        "/finance/accounts",
        json={"team_id": tid, "name": "ICBC", "type": "bank", "currency": "CNY", "balance_fen": 100000},
        params={"team_id": tid},
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "ICBC"
    assert resp.json()["balance_fen"] == 100000


@pytest.mark.asyncio
async def test_list_accounts(client):
    c, team = client
    tid = str(team.id)
    await c.post(
        "/finance/accounts",
        json={"team_id": tid, "name": "CMB", "type": "credit", "currency": "CNY", "balance_fen": 0},
        params={"team_id": tid},
    )
    resp = await c.get("/finance/accounts", params={"team_id": tid})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_update_account(client):
    c, team = client
    tid = str(team.id)
    create = await c.post(
        "/finance/accounts",
        json={"team_id": tid, "name": "Cash", "type": "cash", "currency": "CNY", "balance_fen": 0},
        params={"team_id": tid},
    )
    acc_id = create.json()["id"]
    resp = await c.patch(
        f"/finance/accounts/{acc_id}",
        json={"balance_fen": 5000},
        params={"team_id": tid},
    )
    assert resp.status_code == 200
    assert resp.json()["balance_fen"] == 5000


# ---------- Categories ----------

@pytest.mark.asyncio
async def test_create_category(client):
    c, team = client
    tid = str(team.id)
    resp = await c.post(
        "/finance/categories",
        json={"team_id": tid, "name": "Food", "icon": "F"},
        params={"team_id": tid},
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "Food"


@pytest.mark.asyncio
async def test_list_categories(client):
    c, team = client
    tid = str(team.id)
    await c.post(
        "/finance/categories",
        json={"team_id": tid, "name": "Transport"},
        params={"team_id": tid},
    )
    resp = await c.get("/finance/categories", params={"team_id": tid})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


# ---------- Transactions ----------

@pytest.fixture
async def acc_cat(client):
    c, team = client
    tid = str(team.id)
    acc_resp = await c.post(
        "/finance/accounts",
        json={"team_id": tid, "name": "TxAcc", "type": "bank", "currency": "CNY", "balance_fen": 0},
        params={"team_id": tid},
    )
    cat_resp = await c.post(
        "/finance/categories",
        json={"team_id": tid, "name": "TxCat"},
        params={"team_id": tid},
    )
    return acc_resp.json()["id"], cat_resp.json()["id"]


@pytest.mark.asyncio
async def test_create_transaction(client, acc_cat):
    c, team = client
    tid = str(team.id)
    acc_id, cat_id = acc_cat
    resp = await c.post(
        "/finance/transactions",
        json={
            "team_id": tid,
            "account_id": acc_id,
            "category_id": cat_id,
            "amount_fen": 3500,
            "direction": "expense",
            "description": "Starbucks",
            "transaction_date": str(date.today()),
        },
        params={"team_id": tid},
    )
    assert resp.status_code == 201
    assert resp.json()["amount_fen"] == 3500


@pytest.mark.asyncio
async def test_list_transactions(client, acc_cat):
    c, team = client
    tid = str(team.id)
    acc_id, _ = acc_cat
    await c.post(
        "/finance/transactions",
        json={"team_id": tid, "account_id": acc_id, "amount_fen": 500,
              "direction": "expense", "transaction_date": str(date.today())},
        params={"team_id": tid},
    )
    resp = await c.get("/finance/transactions", params={"team_id": tid})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_soft_delete_transaction(client, acc_cat):
    c, team = client
    tid = str(team.id)
    acc_id, _ = acc_cat
    create = await c.post(
        "/finance/transactions",
        json={"team_id": tid, "account_id": acc_id, "amount_fen": 200,
              "direction": "expense", "transaction_date": str(date.today())},
        params={"team_id": tid},
    )
    tx_id = create.json()["id"]
    del_resp = await c.delete(f"/finance/transactions/{tx_id}", params={"team_id": tid})
    assert del_resp.status_code == 204

    list_resp = await c.get("/finance/transactions", params={"team_id": tid})
    assert all(t["id"] != tx_id for t in list_resp.json())


# ---------- Budgets ----------

@pytest.mark.asyncio
async def test_create_budget(client):
    c, team = client
    tid = str(team.id)
    cat = await c.post(
        "/finance/categories",
        json={"team_id": tid, "name": "BudgetCat"},
        params={"team_id": tid},
    )
    resp = await c.post(
        "/finance/budgets",
        json={"team_id": tid, "category_id": cat.json()["id"],
              "amount_fen": 20000, "period": "monthly", "alert_threshold": 0.8},
        params={"team_id": tid},
    )
    assert resp.status_code == 201
    assert resp.json()["amount_fen"] == 20000


@pytest.mark.asyncio
async def test_budget_usage(client, acc_cat):
    c, team = client
    tid = str(team.id)
    acc_id, cat_id = acc_cat
    budget_resp = await c.post(
        "/finance/budgets",
        json={"team_id": tid, "category_id": cat_id, "amount_fen": 10000, "period": "monthly"},
        params={"team_id": tid},
    )
    assert budget_resp.status_code == 201
    budget_id = budget_resp.json()["id"]

    await c.post(
        "/finance/transactions",
        json={"team_id": tid, "account_id": acc_id, "category_id": cat_id,
              "amount_fen": 4000, "direction": "expense",
              "transaction_date": str(date.today())},
        params={"team_id": tid},
    )

    resp = await c.get(f"/finance/budgets/{budget_id}/usage", params={"team_id": tid})
    assert resp.status_code == 200
    data = resp.json()
    assert data["spent_fen"] == 4000
    assert abs(data["usage_ratio"] - 0.4) < 0.001
