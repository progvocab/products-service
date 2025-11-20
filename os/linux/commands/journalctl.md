### What Is `journalctl` in Linux

`journalctl` is the command-line tool used to **query and view logs** collected by **systemd-journald**, the system logging service. Unlike traditional syslog, journald stores logs in a **binary structured format**, supports metadata (PID, UID, systemd unit, cgroup), and integrates tightly with services managed by systemd. `journalctl` reads these logs directly from `/var/log/journal/` and displays them in human-readable form.



### Basic Usage to View Logs

#### View All Logs

```bash
journalctl
```

#### Follow Logs in Real Time (like `tail -f`)

```bash
journalctl -f
```

#### View Logs for a Specific Service

```bash
journalctl -u nginx.service
journalctl -u docker.service
```

#### View Logs Since a Specific Time

```bash
journalctl --since "2025-01-01 10:00"
journalctl --since "2 hours ago"
```

#### View Logs for Current Boot

```bash
journalctl -b
```

#### View Logs for a Previous Boot

```bash
journalctl -b -1
```



### Filtering Examples

#### Filter by Priority (Error, Warning, Info)

```bash
journalctl -p err
journalctl -p warning
journalctl -p info
```

#### Filter by PID

```bash
journalctl _PID=1234
```

#### Filter by Process or Executable

```bash
journalctl _COMM=sshd
```



### How `journalctl` Works Internally

1. **systemd-journald** collects logs from:

   * kernel (`dmesg`)
   * services managed by systemd
   * syslog
   * standard output/error of services
   * audit messages
2. Logs are stored as **binary journal files** in:

   * `/run/log/journal/` (volatile)
   * `/var/log/journal/` (persistent)
3. `journalctl` reads these files, applies filters, and formats output.
4. Pagination is handled by **less**, unless disabled with `--no-pager`.



### Example: Viewing Logs for a Java App Managed by systemd

```bash
journalctl -u custom-app.service -f
```

systemd-journald captures the app’s stdout/stderr, timestamps it, adds metadata (PID, UID, cgroup), and displays it.



### Viewing Logs in JSON (for log aggregation)

```bash
journalctl -u nginx.service -o json-pretty
```

More :  log flow from kernel/app → journald → journalctl.
