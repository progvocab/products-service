Excellent question — WhatsApp actually uses **all three**, but for **different purposes** depending on the platform and network state. Let’s break it down clearly 👇

---

### 🧩 1. **WhatsApp on Mobile (Android & iOS)**

#### ✅ Uses **MQTT** (over TCP)

* WhatsApp’s core messaging protocol is **MQTT (Message Queuing Telemetry Transport)** — a lightweight publish-subscribe protocol.
* It’s used for:

  * **Sending and receiving messages in real time**
  * **Delivery receipts** (✓ ✓)
  * **Presence updates** (online, typing, etc.)
* MQTT is **optimized for mobile** networks — it minimizes data and battery usage, unlike plain WebSockets.

**Why MQTT instead of WebSocket?**

| Feature                          | MQTT                        | WebSocket                |
| -------------------------------- | --------------------------- | ------------------------ |
| Bandwidth Usage                  | Very low                    | Higher                   |
| Battery Optimization             | Built-in keep-alive control | No standard keep-alive   |
| QoS Levels (Guaranteed delivery) | Yes (QoS 0,1,2)             | No                       |
| Network reconnection handling    | Automatic                   | Manual logic needed      |
| Packet Size                      | Smaller binary packets      | Larger text-based frames |

So WhatsApp keeps a **persistent MQTT connection** when the app is in foreground or active in the background (depending on the OS).

---

### 📱 2. **When the App is Closed or Killed**

#### ✅ Uses **Push Notifications (APNs / FCM)**

* **iOS:** Apple Push Notification Service (APNs)
* **Android:** Firebase Cloud Messaging (FCM)

Since iOS (and newer Android versions) **don’t allow background network sockets**, WhatsApp can’t keep MQTT connected when the app is closed.
So they use **push notifications** to:

* Wake the app when a new message arrives
* Show a message preview on the lock screen
* Re-establish the MQTT connection once the app is opened

---

### 💻 3. **WhatsApp Web / Desktop**

#### ✅ Uses **WebSocket**

* The WhatsApp Web client connects to your phone via a **WebSocket** bridge.
* The phone acts as the “primary device”, keeping the MQTT session alive.
* Messages flow:

  ```
  Web → WebSocket → Phone → MQTT → WhatsApp Server
  ```
* In the new **multi-device mode**, each device maintains its own **MQTT session** with WhatsApp’s servers, but still communicates with the web client via WebSockets.

---

### ⚙️ Summary

| Platform                            | Protocol Used                  | Purpose                                     |
| ----------------------------------- | ------------------------------ | ------------------------------------------- |
| Android / iOS (Active)              | MQTT                           | Real-time messaging, presence               |
| Android / iOS (Background / Killed) | Push Notification (FCM / APNs) | Wake up / Notify                            |
| WhatsApp Web                        | WebSocket                      | Sync with phone or direct server connection |
| Multi-device mode                   | MQTT (per device)              | Independent message sync                    |

---

Would you like me to show a **diagram** of WhatsApp’s end-to-end communication flow (with MQTT, Push, and WebSocket connections)?
