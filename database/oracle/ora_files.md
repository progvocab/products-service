In Oracle, **`tnsnames.ora`**, **`sqlnet.ora`**, and **`listener.ora`** are key configuration files used for network communication between Oracle clients and databases. Here's a breakdown of each:

---

### **1. `tnsnames.ora`**

**Purpose:**  
Defines **TNS aliases** that map to network connection details (host, port, service name) for Oracle databases.

**Location:**
```bash
$ORACLE_HOME/network/admin/
```

**Example:**
```ora
MYDB =
  (DESCRIPTION =
    (ADDRESS = (PROTOCOL = TCP)(HOST = dbserver.example.com)(PORT = 1521))
    (CONNECT_DATA =
      (SERVICE_NAME = orclpdb1.example.com)
    )
  )
```

**Use Case:**  
Used by Oracle tools like SQL*Plus or applications to **resolve a TNS alias** to connection parameters.

---

### **2. `sqlnet.ora`**

**Purpose:**  
Defines **network configuration parameters**, such as naming methods (like TNSNAMES or LDAP), encryption, and timeout settings.

**Example:**
```ora
NAMES.DIRECTORY_PATH = (TNSNAMES, EZCONNECT)

SQLNET.ENCRYPTION_SERVER = required
SQLNET.CRYPTO_CHECKSUM_SERVER = required
SQLNET.EXPIRE_TIME = 10
```

**Use Case:**  
- Control how client resolves service names  
- Enable encryption and security settings  
- Manage connection expiration and keepalives

---

### **3. `listener.ora`**

**Purpose:**  
Configures the **Oracle Listener**, which accepts incoming client connections and routes them to the appropriate database service.

**Example:**
```ora
LISTENER =
  (DESCRIPTION_LIST =
    (DESCRIPTION =
      (ADDRESS = (PROTOCOL = TCP)(HOST = dbserver)(PORT = 1521))
    )
  )

SID_LIST_LISTENER =
  (SID_LIST =
    (SID_DESC =
      (SID_NAME = orcl)
      (ORACLE_HOME = /u01/app/oracle/product/21.0.0/dbhome_1)
    )
  )
```

**Use Case:**  
Defines which databases the listener serves and on what network interface and port.

---

### **4. `ldap.ora`** (optional)

**Purpose:**  
Used when Oracle uses **LDAP directory naming** (like Oracle Internet Directory) instead of `tnsnames.ora`.

---

### **5. `cman.ora`** (optional)

**Purpose:**  
Configures Oracle **Connection Manager**, a lightweight connection proxy to manage traffic and multiplex connections.

---

### **Summary Table**

| File             | Purpose                         | Client or Server | Required |
|------------------|----------------------------------|------------------|----------|
| `tnsnames.ora`   | Defines connection aliases       | Client           | No (if using EZCONNECT or LDAP) |
| `sqlnet.ora`     | Sets network/security policies   | Client & Server  | Yes      |
| `listener.ora`   | Configures database listener     | Server           | Yes      |
| `ldap.ora`       | LDAP connection info             | Client           | Optional |
| `cman.ora`       | Connection Manager config        | Middle-tier      | Optional |

---

Let me know if you'd like to see a working config for secure client-to-server connections!