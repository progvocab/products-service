The `javax` package in Java is a **namespace** that stands for "**Java Extension**." Originally, it was meant for APIs that were **extensions** to the core Java platform (`java.*`), but over time many of these became standard parts of Java.

---

### Origin and Evolution

* Introduced with Java 2 (JDK 1.2).
* "javax" was used to keep backward compatibility when promoting extension packages to core Java.
* Many `javax` packages were later **migrated** to `jakarta.*` (starting in Jakarta EE 9) after the Java EE platform was transferred from Oracle to the Eclipse Foundation.

---

### Common `javax` Packages

Here are some widely used packages under `javax`:

| Package             | Purpose                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| `javax.servlet`     | For developing web components like Servlets and Filters.                 |
| `javax.persistence` | JPA (Java Persistence API) for ORM and database interactions.            |
| `javax.validation`  | Bean Validation API (e.g., for annotating model validation constraints). |
| `javax.mail`        | For sending and receiving emails.                                        |
| `javax.sql`         | JDBC extension for DataSource and pooled connections.                    |
| `javax.annotation`  | Common annotations like `@PostConstruct`, `@PreDestroy`.                 |
| `javax.xml`         | XML parsing and transformation (DOM, SAX, XPath, etc.).                  |
| `javax.ws.rs`       | JAX-RS for building RESTful web services.                                |
| `javax.faces`       | JSF (JavaServer Faces) for building UI components in Java web apps.      |

---

### Transition to `jakarta`

As of Jakarta EE 9:

* All `javax.*` packages are now `jakarta.*`.
* Example: `javax.persistence` → `jakarta.persistence`.

This means:

* Java EE 8 and below use `javax`.
* Jakarta EE 9+ uses `jakarta`.

---

### Usage in Modern Java

In modern Spring Boot or Jakarta EE projects:

* You may still see `javax` if using older dependencies (e.g., `javax.servlet-api`).
* For newer libraries, use `jakarta.*` instead.

Let me know if you want examples for a specific `javax` package.
