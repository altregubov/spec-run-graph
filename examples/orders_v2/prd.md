# Orders — Product Requirements

Domain: customer orders for the demo shop. Customers authenticate with a bearer token.
Test users: a standard customer (`CUSTOMER_TOKEN`) and an administrator (`ADMIN_TOKEN`).
Test data: `SKU-ACTIVE-1` is an active product; `SKU-RETIRED` is no longer sold.

## PRD-ORD-40 Browse products
Customers can list all products that are currently on sale. Retired products must not be listed.

## PRD-ORD-42 Create draft order
An authenticated customer can create an order with one or more line items (SKU + quantity).
A new order is always created in `DRAFT` status and its total is the sum of price × quantity.
Clients may send an `Idempotency-Key` header; repeating the same request with the same key must
return the original order instead of creating a duplicate.

## PRD-ORD-43 Quantity validation
- A quantity of 0 is rejected with HTTP 422 and error code `MIN_QTY_EXCEEDED`.
- A negative quantity is rejected with HTTP 422 and error code `NEGATIVE_QTY`.
- A quantity above 50 per line is rejected with HTTP 400 and error code `MAX_ORDER_LIMIT`.
- Every validation error names the offending field, e.g. `items[0].qty`.

## PRD-ORD-44 Unknown or retired products
Ordering a SKU that does not exist or is retired is rejected with HTTP 422 and error code `UNKNOWN_SKU`.

## PRD-ORD-45 Confirm order
A customer confirms a `DRAFT` order, which moves it to `CONFIRMED`. Confirming an order that is not
in `DRAFT` status fails with HTTP 409 `INVALID_STATE`.

## PRD-ORD-46 View order
A customer can fetch an order by id and sees its status, items and total. Unknown ids return 404.

## PRD-ORD-47 Delete order
Customers may delete their own orders; administrators may delete any order. Deleting another
customer's order returns 403. Deleting an unknown order returns 404.
A `CONFIRMED` order can no longer be deleted (HTTP 409 `INVALID_STATE`).

## PRD-ORD-48 Order history
Customers can list their own order history, newest first, paginated by 20.

## PRD-ORD-49 Security
All order and product endpoints require a valid bearer token; missing or invalid tokens return 401.
