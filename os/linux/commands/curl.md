Yes — you can, but there are several ways to do it and important caveats. Below I’ll show practical methods using `curl` (quick hacks) and then recommend real load-testing tools that are far better for performance testing. I’ll also explain what each approach measures and what to watch out for.

---

## Quick `curl` methods (simple, convenient)

### 1) Using `xargs` (concurrency control)

Run **100 requests total** with **50 concurrent** at a time:

```bash
seq 1 100 | xargs -n1 -P50 -I{} \
  curl -s -o /dev/null -w "%{http_code} %{time_total}\n" "http://example.com/api"
```

* `-P50` → up to 50 parallel processes.
* `-n1` → pass one arg per `curl` invocation.
* `-s -o /dev/null -w` → silent, discard body, print status + total time.

---

### 2) Simple shell loop with background jobs

Fire off 100 requests as background jobs (best-effort concurrency = as many as system allows), then `wait`:

```bash
for i in $(seq 1 100); do
  curl -s -o /dev/null -w "%{http_code} %{time_total}\n" "http://example.com/api" &
done
wait
```

* Easy, but you have little control over max concurrent requests.

---

### 3) `curl` built-in parallel (modern curl)

If your `curl` supports `--parallel` (curl ≥ 7.66) you can do:

```bash
# create a list of URLs (100 lines)
for i in $(seq 1 100); do echo "http://example.com/api"; done > urls.txt

# run up to 50 parallel transfers
curl --parallel --parallel-max 50 --silent --show-error --write-out "%{http_code} %{time_total}\n" --output /dev/null -K urls.txt
```

Or inline (shell brace expansion may be used if multiple distinct URLs):

```bash
curl --parallel --parallel-max 50 http://example.com/api http://example.com/other -o /dev/null
```

`--parallel` is convenient but mainly for fetching many different URLs; using it to stress a single endpoint requires listing that URL many times.

---

## Measuring useful metrics with `curl`

Use `-w` format tokens. Common useful ones:

* `%{http_code}` — response status
* `%{time_total}` — total request time
* `%{time_connect}` — time to TCP connect
* `%{time_starttransfer}` — time to first byte (TTFB)

Example:

```bash
curl -s -o /dev/null -w "code=%{http_code} time=%{time_total}s\n" "http://example.com/api"
```

Aggregate results (example: count success/fail):

```bash
seq 1 100 | xargs -n1 -P20 -I{} \
  curl -s -o /dev/null -w "%{http_code}\n" "http://example.com/api" \
| sort | uniq -c
```

---

## Important caveats using `curl` for performance testing

1. **Not a full-featured load tester** — curl is fine for quick smoke tests but lacks realistic concurrency control, ramping, latency histograms, and accurate throughput measurements.

2. **Client machine limits** — your CPU, network, ephemeral ports, ulimit for open files, and default TCP settings will cap throughput. You may saturate the client, not the server.

3. **Connection reuse / HTTP/1.1 vs HTTP/2** — repeated curl invocations may create new TCP/TLS handshakes unless you use persistent connections (hard with separate curl processes). This can artificially increase latency.

4. **No latency distributions** — curl prints per-request times but you’ll have to aggregate to get percentiles (p50/p95/p99).

5. **Risk of accidentally DoS-ing a service** — be careful with endpoints you don’t own or without permission.

---

## Recommended load-testing tools (use these for real testing)

* **wrk** (very popular C tool): high performance, Lua scripting, throughput & latency. Example:

  ```
  wrk -t8 -c200 -d30s http://example.com/api
  ```

  * `-t8` = 8 threads, `-c200` = 200 concurrent connections, `-d30s` = 30 seconds.

* **hey** (Go, from rakyll): simple to use.

  ```
  hey -n 10000 -c 100 http://example.com/api
  ```

  * `-n` = total requests, `-c` = concurrency.

* **k6** (JavaScript scripting, modern features, cloud option): for complex scenarios and metrics.

  ```js
  // script.js
  import http from 'k6/http';
  import { sleep } from 'k6';
  export default function () {
    http.get('http://example.com/api');
    sleep(1);
  }
  ```

  then run: `k6 run --vus 100 --duration 30s script.js`

* **vegeta** (attacker-style tool): supports constant-rate attacks and produces nice latency percentiles.

  ```
  echo "GET http://example.com/api" | vegeta attack -rate=200 -duration=30s | vegeta report
  ```

* **ApacheBench (ab)** — older but still used:

  ```
  ab -n 10000 -c 100 http://example.com/api
  ```

---

## Which to use when?

* Quick single-shot concurrency test → `curl` with `xargs` or `&` is OK.
* Real benchmarking (throughput, p95/p99 latency, ramping, long duration) → **wrk**, **hey**, **k6**, or **vegeta**.

---

## Example recommended workflow (real testing)

1. Start small: low concurrency, short duration, sanity check.
2. Ramp up concurrency gradually to find saturation point.
3. Monitor the server (CPU, RAM, threads, GC, network) while testing.
4. Use a tool that reports percentiles (p50/p95/p99) — curl alone does not.
5. Repeat tests and average results.

---

## Quick cheat-sheet commands

* `curl` parallel (xargs):

  ```bash
  seq 1 1000 | xargs -n1 -P100 -I{} curl -s -o /dev/null -w "%{time_total}\n" "http://example.com/api" > times.txt
  # then analyze times.txt (e.g., compute percentiles)
  ```

* `hey` example:

  ```bash
  hey -n 10000 -c 200 -q 10 http://example.com/api
  ```

* `wrk` example:

  ```bash
  wrk -t4 -c200 -d60s http://example.com/api
  ```

* `vegeta` example:

  ```bash
  echo "GET http://example.com/api" | vegeta attack -rate=500 -duration=60s | vegeta report
  ```

---

If you want, I can:

* generate a **ready-to-run script** using `curl` + `xargs` that collects times and computes p50/p95/p99, or
* prepare a **wrk/hey/k6 command** tuned to your target (you can tell me total requests, concurrency, and whether you want a duration-based or total-request test).

Which would you prefer?
