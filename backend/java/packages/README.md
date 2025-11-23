## Package Info

A **`package-info.java`** file is a special Java file used to attach **documentation, annotations, and package-level comments** to a package.

It does **not** contain classes; it exists only to provide metadata for a package.

---

# ✅ **Why `package-info.java` Exists**

Java packages cannot directly have:

* Javadoc comments
* Package-level annotations
* Descriptions of architecture or usage
* Deprecation annotations
* Nullability annotations
* Module-level metadata (pre-Java 9)

`package-info.java` solves this.

---

# ✅ **Typical Contents of package-info.java**

### ✔️ 1. Package-level Javadoc

```java
/**
 * This package contains the service layer implementations.
 * It handles business logic and integrates with repositories.
 */
package com.example.service;
```

---

### ✔️ 2. Package-level annotations

Example: marking an entire package as `@NonnullByDefault`:

```java
@ParametersAreNonnullByDefault
package com.example.util;

import javax.annotation.ParametersAreNonnullByDefault;
```

---

### ✔️ 3. Deprecating a whole package

```java
/**
 * @deprecated use com.example.newapi.* instead
 */
@Deprecated
package com.example.oldapi;
```

---

### ✔️ 4. Hibernate/JPA annotations

You can mark a package with a specific naming strategy:

```java
@org.hibernate.annotations.TypeDef(
    name = "json",
    typeClass = JsonType.class
)
package com.example.domain;
```

---

# 📝 **Where to place package-info.java**

Each package gets its **own** file:

```
src/main/java/
    com/example/service/package-info.java
    com/example/util/package-info.java
```

Each file contains exactly **one** `package` declaration and optional annotations.

---

# ❗ Important Rules

* A package-info.java file **must contain ONLY**:

  * Javadoc comment
  * Annotations
  * The `package` statement
* No classes inside
* No multiple packages
* The file name must be exactly **`package-info.java`**

---

# ⭐ Why is it useful?

| Purpose                           | Example                            |
| --------------------------------- | ---------------------------------- |
| Add Javadoc for the whole package | Explain the architecture or intent |
| Add default null-safety           | `@ParametersAreNonnullByDefault`   |
| Add custom annotations            | Logging, tracing, validation       |
| Mark package deprecated           | `@Deprecated`                      |
| Add ORM defaults                  | JPA/Hibernate types                |

---

# If you want:

I can show examples for:

✔ Spring Boot package-level configuration
✔ Lombok `@NonNull` package defaults
✔ Package-level logging annotations
✔ Real-world architecture documentation using package-info.java


## Package Level Annotations 

Below is a **comprehensive table of commonly used package-level annotations in Java**, including **core Java**, **Jakarta/Javax**, **Spring**, **Hibernate**, **Lombok**, and **JSR-305 / Checker Framework**.

(And as per your preference, the table starts after a blank line.)

---

# 📘 **Package-Level Annotations (Complete Reference Table)**

Below: Annotation → What it does → When to use → Example

---

| Annotation                                                  | Package / Library          | What It Does                                                                          | When to Use                                           | Example in `package-info.java`                                  |
| ----------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| **@Deprecated**                                             | Java Standard              | Marks entire package as deprecated                                                    | The entire package should no longer be used           | `@Deprecated package com.example.old;`                          |
| **@Generated**                                              | Java Standard              | Indicates the package’s code is generated                                             | Auto-generated API clients, DTOs                      | `@Generated("tool") package com.gen;`                           |
| **@ParametersAreNonnullByDefault**                          | JSR-305 (javax.annotation) | All parameters in all methods of this package are non-null unless annotated otherwise | Enforce default null-safety                           | `@ParametersAreNonnullByDefault package com.app.api;`           |
| **@ReturnValuesAreNonnullByDefault**                        | JSR-305                    | All return values are non-null by default                                             | Prevent null return bugs                              | `@ReturnValuesAreNonnullByDefault package com.app.service;`     |
| **@FieldsAreNonnullByDefault**                              | JSR-305                    | All fields in all classes of the package default to non-null                          | Strong null safety                                    | `@FieldsAreNonnullByDefault package com.domain;`                |
| **@Nonnull**                                                | javax.annotation / JSR-305 | Applied at package-level to indicate non-null defaults                                | When entire package should treat values as non-null   | `@Nonnull package com.secure;`                                  |
| **@CheckReturnValue**                                       | ErrorProne / JSR-305       | Methods’ return values must not be ignored                                            | Useful for validation methods, builder patterns       | `@CheckReturnValue package com.example.validators;`             |
| **@TypeQualifierDefault**                                   | JSR-305                    | Defines custom default type qualifiers                                                | Used with custom nullability or compliance frameworks | Used with other null annotations                                |
| **@ParametersAreNullableByDefault**                         | JSR-305                    | Opposite of nonnull – all parameters nullable by default                              | Legacy code where null is common                      | `@ParametersAreNullableByDefault package legacy;`               |
| **@EntityListeners**                                        | JPA / Jakarta Persistence  | Apply entity listeners to all entities in this package                                | When entities share common audit logic                | `@EntityListeners(AuditListener.class) package com.app.domain;` |
| **@SequenceGenerator**                                      | JPA                        | Defines a sequence, used by entities inside the package                               | Common ID generation package-wide                     | `@SequenceGenerator(name="idGen", ...)`                         |
| **@NamedQueries**                                           | JPA                        | Defines named queries package-wide                                                    | Centralize query definitions                          | Large domain models                                             |
| **@TypeDef**                                                | Hibernate                  | Defines a custom Hibernate type available package-wide                                | For JSON, XML, or custom value objects                | `@TypeDef(name="json", typeClass=JsonType.class)`               |
| **@TypeDefs**                                               | Hibernate                  | Multiple Hibernate type definitions                                                   | When several custom types used in package             | Multiple `@TypeDef` entries                                     |
| **@FilterDef**                                              | Hibernate                  | Defines Hibernate filters                                                             | Soft deletes, multi-tenancy                           | `@FilterDef(name="activeFilter", ...)`                          |
| **@FilterDefs**                                             | Hibernate                  | Multiple filter definitions                                                           | Apply filters across package                          | Combined filters                                                |
| **@LombokGetter`/`@LombokSetter` (via @Getter @Setter)**    | Lombok                     | Set default getters/setters for all classes in the package                            | Reduce boilerplate in package                         | `@Getter @Setter package com.model;`                            |
| **@Value**                                                  | Lombok                     | Make all classes immutable by default                                                 | Functional, immutable packages                        | `@Value package com.dto;`                                       |
| **@Log4j2 / @Slf4j**                                        | Lombok                     | Enable logging for every class in the package                                         | Auto-add logs to all classes                          | `@Slf4j package com.example.*;`                                 |
| **@SpringBootApplication** *(not allowed at package level)* | —                          | ✘ Not allowed                                                                         | —                                                     | —                                                               |
| **@Configuration** *(not allowed at package level)*         | —                          | ✘ Not allowed                                                                         | —                                                     | —                                                               |
| **@NonNullApi**                                             | Spring Framework           | Treat all package elements as non-null unless annotated otherwise                     | Strong null-safety in Spring apps                     | `@NonNullApi package com.app.api;`                              |
| **@NonNullFields**                                          | Spring Framework           | All fields in this package are non-null                                               | Domain model packages                                 | `@NonNullFields package com.app.domain;`                        |
| **@ParametersAreNonnullByDefault**                          | Spring (JSR-305 aligned)   | Same purpose as JSR-305 version                                                       | Default null safety                                   | Identical usage                                                 |
| **@XmlSchema**                                              | JAXB                       | Configures XML namespace for entire package                                           | Govern XML bindings                                   | `@XmlSchema(namespace="http://example.com")`                    |
| **@XmlAccessorType**                                        | JAXB                       | Controls whether fields or getters form XML elements                                  | Centralize XML config                                 | `@XmlAccessorType(XmlAccessType.FIELD)`                         |
| **@XmlJavaTypeAdapters**                                    | JAXB                       | Register custom XML adapters package-wide                                             | For complex type bindings                             | Complex XML models                                              |

---

# ⭐ Notes

### ✔ These annotations must appear **before** the `package` declaration:

```java
/**
 * Javadoc for the package.
 */
@Nonnull
@ParametersAreNonnullByDefault
package com.example.myapp;
```

### ✔ You cannot place classes in `package-info.java`

### ✔ It is ideal for:

* Architecture documentation
* Null-safety defaults
* Hibernate, JPA, JAXB package configs
* Deprecation
* Logging defaults (Lombok)

---

# If you want:

I can generate:

✔ Full sample `package-info.java` for Spring Boot or JPA
✔ Package-level null safety best practices
✔ A skeleton file for your actual project structure

Just tell me!

