### What is the dig command

The **dig** command (**Domain Information Groper**) is a powerful DNS lookup tool used in Linux to query DNS servers and retrieve detailed information about domain names, IP addresses, records, and DNS resolution paths.

### Why dig is used

* To check DNS records
* To debug DNS resolution issues
* To verify that DNS zones are configured correctly
* To check which DNS server is responding
* To see TTL (Time To Live) of records
* To perform reverse lookups and trace DNS paths

### Basic syntax

```
dig <domain>
```

Example:

```
dig google.com
```

### What dig shows (important sections)

When you run `dig google.com`, the output is split into main sections:

#### Header

Shows status, opcode, response codes.
Example:
`status: NOERROR` means the query succeeded.

#### QUESTION SECTION

The query you asked for:

```
;google.com.    IN   A
```

#### ANSWER SECTION

Contains the actual DNS records returned:

```
google.com.  300  IN  A  142.250.195.78
```

#### AUTHORITY SECTION

Shows which DNS server is authoritative for the domain:

```
ns1.google.com.
```

#### ADDITIONAL SECTION

Provides related info, often IPs of name servers.

### Common dig options

#### Query a specific record type

```
dig google.com A
dig google.com AAAA
dig google.com MX
dig google.com CNAME
dig google.com TXT
dig google.com NS
```

#### Query a specific DNS server

```
dig @8.8.8.8 google.com
```

#### Reverse DNS lookup

```
dig -x 142.250.195.78
```

#### Short answer only (clean output)

```
dig +short google.com
```

Output:

```
142.250.195.78
```

#### Display only the answer section

```
dig +answer google.com
```

#### Show TTL

Every DNS record returned includes a TTL field. E.g.:

```
google.com. 300 IN A 142.250.195.78
```

#### Trace DNS resolution path (like how DNS finds the domain)

```
dig +trace google.com
```

This performs a DNS lookup starting from the root servers.

#### Increase verbosity

```
dig +noall +answer google.com
```

### Examples

#### Check MX (mail server) records

```
dig gmail.com MX
```

#### Check NS (nameserver) records

```
dig google.com NS
```

#### Check TXT records (SPF, DKIM, verification)

```
dig google.com TXT
```

#### Check SOA record

```
dig google.com SOA
```

### Important fields in the answer section

| Field | Meaning                        |
| ----- | ------------------------------ |
| Name  | The domain queried             |
| TTL   | Time record stays in DNS cache |
| Class | Usually IN (Internet)          |
| Type  | A, AAAA, MX, NS, etc.          |
| Data  | The actual value               |

### Why dig is preferred over nslookup

| Feature              | dig | nslookup   |
| -------------------- | --- | ---------- |
| Modern tool          | Yes | Deprecated |
| More detailed output | Yes | Limited    |
| Supports +trace      | Yes | No         |
| Better for debugging | Yes | No         |

### When dig is useful

* Website not resolving
* DNS propagation check
* SSL certificate hostname mismatch
* Email delivery issues (check MX, SPF, DKIM)
* Debug slow DNS lookup
* Investigate CDN routing

If you want, I can also explain **nslookup**, **host**, or give **real troubleshooting examples using dig**.
