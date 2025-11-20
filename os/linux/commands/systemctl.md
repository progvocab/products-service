### What Is `systemctl` in Linux

`systemctl` is the **command-line interface for systemd**, used to control and inspect **services**, **units**, **system state**, and **startup behavior**. It interacts with the **systemd PID 1 process**, which is responsible for service management, logging integration, device management, and boot control. `systemctl` does not operate directly on services—it **sends requests to systemd**, and systemd performs the actual actions.

---

### Key Commands and Their Meaning

#### Start, Stop, Restart, Reload a Service

```bash
systemctl start nginx
systemctl stop nginx
systemctl restart nginx
systemctl reload nginx
```

* `systemctl` sends a DBus request → **systemd** starts/stops the service.

#### Enable or Disable Service on Boot

```bash
systemctl enable nginx
systemctl disable nginx
```

* Creates or removes symlinks under `/etc/systemd/system/*.wants/`.

#### Check Service Status

```bash
systemctl status nginx
```

* Shows logs from **journald**, service state, PID, and uptime.

#### List All Services

```bash
systemctl list-units --type=service
```

#### Check System State

```bash
systemctl is-active sshd
systemctl is-enabled sshd
systemctl is-failed
```

#### Restart Entire System or Power Off

```bash
systemctl reboot
systemctl poweroff
```

* `systemd` executes shutdown targets.

---

### How `systemctl` Works Internally

1. You run `systemctl <command>`.
2. It communicates with **systemd** (PID 1) over **DBus**.
3. systemd reads the corresponding **unit file** from:

   * `/etc/systemd/system/`
   * `/usr/lib/systemd/system/`
4. systemd executes service actions (start/stop) using cgroups and process tracking.
5. Logs collected in **journald** appear in `systemctl status`.

---

### Example Unit File

```ini
# /etc/systemd/system/custom-app.service
[Unit]
Description=Custom Java Application

[Service]
ExecStart=/usr/bin/java -jar /opt/app/app.jar
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Enable and Start

```bash
systemctl enable custom-app
systemctl start custom-app
```

More : explain **systemd boot targets**, **journald**, or show a **Mermaid diagram** of how systemctl → systemd → cgroups interacts.
