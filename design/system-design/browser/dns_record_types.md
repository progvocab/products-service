Below is a complete list of commonly used DNS record types with **2–3 line explanations** and **examples**.
All headings follow your requirement (**H3 only**).

### A Record

Maps a domain name to an IPv4 address. Used for directing browsers to the server hosting a website.
Example: `example.com. IN A 93.184.216.34`

### AAAA Record

Maps a domain name to an IPv6 address. Similar to A record but for IPv6-enabled systems.
Example: `example.com. IN AAAA 2606:2800:220:1:248:1893:25c8:1946`

### CNAME Record

Creates an alias from one domain to another. The target must be a canonical domain name, not an IP.
Example: `www.example.com. IN CNAME example.com.`

### MX Record

Specifies mail servers responsible for receiving email for a domain, with priority values.
Example: `example.com. IN MX 10 mail.example.com.`

### TXT Record

Stores arbitrary text data such as SPF, DKIM, verification codes, or metadata.
Example: `example.com. IN TXT "v=spf1 include:_spf.google.com ~all"`

### NS Record

Defines the authoritative name servers for a domain. The child zone delegates control to these servers.
Example: `example.com. IN NS ns1.example.com.`

### SOA Record

Start of Authority record that defines the primary name server, serial number, refresh/retry intervals.
Example:
`example.com. IN SOA ns1.example.com. admin.example.com. 2025010101 3600 600 604800 3600`

### PTR Record

Used for reverse DNS lookup, mapping an IP address back to a domain name (opposite of A record).
Example: `34.216.184.93.in-addr.arpa. IN PTR example.com.`

### SRV Record

Specifies hostname and port for a service (e.g., SIP, XMPP, LDAP). Enables service-based discovery.
Example: `_sip._tcp.example.com. IN SRV 10 60 5060 sipserver.example.com.`

### CAA Record

Specifies which Certificate Authorities (CAs) are permitted to issue certificates for the domain.
Example: `example.com. IN CAA 0 issue "letsencrypt.org"`

### SPF Record (Deprecated but still seen)

Used to define permitted email-sending servers; now implemented via TXT.
Example: `example.com. IN SPF "v=spf1 a mx ip4:203.0.113.10 -all"`

### DKIM Record

Stores public key used for validating DKIM-signed emails. Stored as TXT under a selector.
Example: `selector._domainkey.example.com. IN TXT "k=rsa; p=MIIBIjAN..."`

### DMARC Record

Defines domain-level email authentication policy using SPF and DKIM alignment.
Example: `_dmarc.example.com. IN TXT "v=DMARC1; p=reject; rua=mailto:admin@example.com"`

### NAPTR Record

Used for regex-based service discovery; common with SIP and ENUM telephony.
Example:
`example.com. IN NAPTR 100 10 "U" "SIP+D2U" "!^.*$!sip:info@example.com!" .`

### LOC Record

Stores geographical location (latitude, longitude, altitude) for a domain.
Example:
`example.com. IN LOC 37 48 10.000 N 122 24 10.000 W 10.00m`

### CERT Record

Stores certificates, PGP keys, or related security objects.
Example: `example.com. IN CERT PGP 0 0 (BASE64...)`

### HINFO Record

Provides details about the host’s CPU and OS, rarely used due to security concerns.
Example: `example.com. IN HINFO "Intel" "Linux"`

### DNAME Record

Redirects an entire subtree of the DNS namespace (like CNAME for a whole domain).
Example: `old.example.com. IN DNAME new.example.net.`

### DNSKEY Record

Holds public keys for DNSSEC signature validation.
Example: `example.com. IN DNSKEY 257 3 8 (BASE64 KEY...)`

### DS Record

Delegation signer record used in DNSSEC to link parent/child zones securely.
Example: `example.com. IN DS 12345 8 1 A1B2C3D4E5...`

### RRSIG Record

Stores cryptographic signatures for DNSSEC-validated records.
Example: `example.com. IN RRSIG A 8 2 3600 (BASE64...)`

### NSEC / NSEC3 Record

Used by DNSSEC to prove non-existence of records; prevents zone enumeration (NSEC3).
Example: `example.com. IN NSEC blog.example.com. A NS SOA`

### TLSA Record

Used with DANE to bind a TLS certificate to a domain via DNSSEC.
Example:
`_443._tcp.example.com. IN TLSA 3 1 1 (HASH)`

---

If you want, I can also create a **table with all record types**, or explain **how recursive vs authoritative servers process each record inside Linux (BIND/CoreDNS)**.

