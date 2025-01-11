Identity providers (IdPs) manage user authentication and authorization. While SAML and OIDC are common standards, many identity providers use other protocols or frameworks. Here’s a list of identity providers (and associated protocols/standards) beyond SAML and OIDC:

---

### **Identity Providers Using LDAP and Kerberos**  
1. **Microsoft Active Directory (AD)**  
   - Supports LDAP and Kerberos.  
   - Common in enterprise environments for Windows systems.  

2. **OpenLDAP**  
   - An open-source LDAP server.  
   - Often used in Linux and Unix-based environments.

3. **FreeIPA**  
   - Combines LDAP, Kerberos, and DNS for identity management.  
   - Typically used in Red Hat ecosystems.  

---

### **Identity Providers Using Proprietary Protocols**  
1. **AWS IAM (Identity and Access Management)**  
   - AWS-specific authentication and authorization system.  
   - No direct support for SAML/OIDC, but integrates through federation.

2. **Google Workspace (formerly G Suite)**  
   - Provides OAuth 2.0 and proprietary APIs for user authentication.  

3. **Apple ID**  
   - Proprietary protocols for Apple services and devices.

---

### **Social and Consumer Identity Providers**  
1. **OAuth-based Providers**  
   - **Facebook Login**  
   - **Google Sign-In**  
   - **Twitter OAuth**  

2. **Passwordless Providers**  
   - **Magic.link**: Uses email-based one-time links.  
   - **Yubico**: Hardware authentication with proprietary APIs.

---

### **PKI (Public Key Infrastructure) Providers**  
1. **X.509 Certificate Authorities**  
   - Authenticates users via digital certificates.  
   - Common in secure enterprise environments.  

2. **Venafi**  
   - Manages machine and user identities via certificates.

---

### **Custom and Legacy Authentication Protocols**  
1. **Radius Servers**  
   - Used for network access authentication (e.g., Cisco ISE, FreeRADIUS).  

2. **CAS (Central Authentication Service)**  
   - Open-source single sign-on protocol.  

3. **TACACS+**  
   - Proprietary Cisco protocol for access control.

---

### **Decentralized and Blockchain-Based Identity Providers**  
1. **Self-Sovereign Identity (SSI)**  
   - Examples: Sovrin, uPort, Civic.  

2. **Blockchain-Based Solutions**  
   - Examples: DID (Decentralized Identifiers) providers, Hyperledger Indy.  

---

### **Multi-Factor Authentication (MFA) Providers**  
1. **Duo Security**  
   - Provides two-factor authentication (2FA) and integrates with multiple systems.  

2. **Authy**  
   - Offers OTP and other authentication mechanisms.

---

### **Other Protocols and Frameworks**  
1. **WS-Federation**  
   - Used by Microsoft Active Directory Federation Services (AD FS).  

2. **SCIM (System for Cross-domain Identity Management)**  
   - Focused on user provisioning rather than authentication.  

3. **PAM (Pluggable Authentication Modules)**  
   - Common in Unix/Linux for system authentication.  

---

These providers and protocols cater to various use cases, from enterprise to consumer applications. Let me know if you'd like help with specific ones!
