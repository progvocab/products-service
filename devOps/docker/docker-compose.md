To stop Docker containers started using `docker compose up`, you can do it in a few different ways:

---

## ✅ Option 1: Graceful Stop (Recommended)

If you ran:

```bash
docker compose up
```

in a **terminal (foreground)**, just press:

```
CTRL + C
```

This will gracefully stop all services.

---

## ✅ Option 2: Stop and Clean Up (Background / Detached Mode)

If you ran it with `-d` (detached mode):

```bash
docker compose up -d
```

Then stop it with:

```bash
docker compose down
```

This:

* Stops all running containers in the compose file
* Removes the containers and associated networks

---

## ✅ Option 3: Stop Without Removing Containers

If you only want to **stop** the containers (not delete them):

```bash
docker compose stop
```

You can later restart them with:

```bash
docker compose start
```

---

## 🧼 Bonus: Remove All Containers (if needed)

If you want to **clean up everything**:

```bash
docker compose down --volumes --remove-orphans
```

---

Let me know if you want to remove only one container or check the running ones.

