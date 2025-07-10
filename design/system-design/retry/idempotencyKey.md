Here’s a list of **system design use cases where idempotency keys are essential**, especially to ensure **safety, consistency, and correctness** in distributed or unreliable environments.

---

## ✅ What is an Idempotency Key?

An **idempotency key** is a unique identifier sent with a request to guarantee that even if the **same request is submitted multiple times**, the operation is only **performed once**.

---

## 📋 Common System Designs that Use Idempotency Keys

|  # | System Design Scenario                                     | Why Idempotency is Needed                                                                 |
| -: | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
|  1 | **Payment gateways** (e.g., Stripe, Razorpay, PayPal)      | Prevent duplicate charges if user hits “Pay” twice or network retries                     |
|  2 | **Order creation in e-commerce**                           | Avoid placing multiple orders if the API is retried due to timeout                        |
|  3 | **Account signup / registration APIs**                     | Prevent multiple user accounts for the same email/phone                                   |
|  4 | **Message processing in queues** (e.g., Kafka, SQS)        | Ensure that duplicate message delivery doesn’t cause duplicate processing                 |
|  5 | **Webhook receivers**                                      | Avoid acting multiple times when the same webhook is sent multiple times by a third party |
|  6 | **Email/SMS notification services**                        | Prevent duplicate sending of emails/SMS in case of retries                                |
|  7 | **Database write APIs** (POST/PUT/DELETE endpoints)        | Ensure safe writes with client-generated request IDs                                      |
|  8 | **Booking/reservation systems** (flights, hotels, tickets) | Prevent double-booking due to retries                                                     |
|  9 | **Bank transfers or fund withdrawals**                     | Prevent double withdrawals or transfers when requests are resent                          |
| 10 | **Inventory adjustment in retail systems**                 | Avoid incorrect stock deduction from repeated API hits                                    |
| 11 | **Blockchain transaction broadcasting**                    | Prevent multiple broadcasts of the same signed transaction                                |
| 12 | **Document uploads to file storage APIs**                  | Avoid storing multiple copies of the same upload on retries                               |

---

## 🧠 Design Considerations When Using Idempotency

* Store the idempotency key and result in a persistent store (e.g., Redis, PostgreSQL).
* If the same idempotency key comes in again:

  * Return the same response.
  * Skip execution logic.
* Expire keys after a certain time if needed.
* Ensure keys are **unique per user and operation type** to avoid false positives.

---

## 🧰 Common Libraries / Services Supporting It

| Tool                          | Idempotency Support                       |
| ----------------------------- | ----------------------------------------- |
| **Stripe API**                | `Idempotency-Key` header                  |
| **AWS SQS FIFO queues**       | `MessageDeduplicationId`                  |
| **PostgREST / JSON API**      | Use `Prefer: resolution=merge-duplicates` |
| **GraphQL mutation patterns** | Client-side mutation keys                 |

---

## 🔚 Summary

Idempotency keys are a **best practice in any system with retry logic**, financial/state-changing operations, or distributed communication.

Would you like a Spring Boot implementation example with Redis/PostgreSQL idempotency cache?
