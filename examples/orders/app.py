"""Tiny Orders API used as the System Under Test for SpecRunGraph demos.

Seeded defects (intentional, so generated tests have something to find):
  1. POST /v1/orders with qty > 100 crashes with HTTP 500 (spec says 400 MAX_ORDER_LIMIT).
  2. GET /v1/orders/{id} returns the field `state` instead of `status` (schema mismatch).
  3. DELETE /v1/orders/{id} on an unknown id returns 200 instead of 404.

Run:  uv run uvicorn examples.orders.app:app --port 8000
Auth: Bearer token "customer-token" (customer) or "admin-token" (admin).
"""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Orders API", version="1.0.0")

PRODUCTS = {
    "SKU-ACTIVE-1": {"sku": "SKU-ACTIVE-1", "name": "Coffee beans", "price": 12.5, "active": True},
    "SKU-ACTIVE-2": {"sku": "SKU-ACTIVE-2", "name": "Tea leaves", "price": 7.0, "active": True},
    "SKU-RETIRED": {"sku": "SKU-RETIRED", "name": "Old mug", "price": 3.0, "active": False},
}
ORDERS: dict[str, dict] = {}
IDEMPOTENCY: dict[str, str] = {}
TOKENS = {"customer-token": "customer", "admin-token": "admin"}


def error(status: int, code: str, message: str, field: str | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if field:
        body["error"]["field"] = field
    return JSONResponse(status_code=status, content=body)


def current_role(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing token")
    role = TOKENS.get(authorization.removeprefix("Bearer ").strip())
    if role is None:
        raise HTTPException(status_code=401, detail="invalid token")
    return role


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    code = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND"}.get(exc.status_code, "ERROR")
    return error(exc.status_code, code, str(exc.detail))


class OrderItem(BaseModel):
    sku: str
    qty: int


class CreateOrder(BaseModel):
    items: list[OrderItem] = Field(min_length=1)


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/products")
def list_products(_: str = Depends(current_role)):
    return {"data": [p for p in PRODUCTS.values() if p["active"]]}


@app.post("/v1/orders", status_code=201)
def create_order(
    body: CreateOrder,
    role: str = Depends(current_role),
    idempotency_key: str | None = Header(default=None),
):
    if idempotency_key and idempotency_key in IDEMPOTENCY:
        return {"data": ORDERS[IDEMPOTENCY[idempotency_key]]}
    total = 0.0
    for i, item in enumerate(body.items):
        if item.qty == 0:
            return error(422, "MIN_QTY_EXCEEDED", "qty must be at least 1", f"items[{i}].qty")
        if item.qty < 0:
            return error(422, "NEGATIVE_QTY", "qty must be positive", f"items[{i}].qty")
        if item.qty > 100:
            raise RuntimeError("order limit check not implemented")  # defect 1 -> HTTP 500
        product = PRODUCTS.get(item.sku)
        if product is None or not product["active"]:
            return error(422, "UNKNOWN_SKU", f"sku {item.sku} is not available", f"items[{i}].sku")
        total += product["price"] * item.qty
    order_id = str(uuid.uuid4())
    order = {
        "id": order_id,
        "status": "DRAFT",
        "items": [i.model_dump() for i in body.items],
        "total": round(total, 2),
        "owner": role,
    }
    ORDERS[order_id] = order
    if idempotency_key:
        IDEMPOTENCY[idempotency_key] = order_id
    return {"data": order}


@app.get("/v1/orders/{order_id}")
def get_order(order_id: str, _: str = Depends(current_role)):
    order = ORDERS.get(order_id)
    if order is None:
        return error(404, "NOT_FOUND", "order not found")
    data = dict(order)
    data["state"] = data.pop("status")  # defect 2 -> schema mismatch
    return {"data": data}


@app.post("/v1/orders/{order_id}/confirm")
def confirm_order(order_id: str, _: str = Depends(current_role)):
    order = ORDERS.get(order_id)
    if order is None:
        return error(404, "NOT_FOUND", "order not found")
    if order["status"] != "DRAFT":
        return error(409, "INVALID_STATE", f"order is {order['status']}")
    order["status"] = "CONFIRMED"
    return {"data": order}


@app.delete("/v1/orders/{order_id}")
def delete_order(order_id: str, role: str = Depends(current_role)):
    if role != "admin" and order_id in ORDERS and ORDERS[order_id]["owner"] != role:
        return error(403, "FORBIDDEN", "not your order")
    ORDERS.pop(order_id, None)  # defect 3 -> unknown id still 200
    return {"data": {"id": order_id, "deleted": True}}
