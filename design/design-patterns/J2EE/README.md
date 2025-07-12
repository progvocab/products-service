The book **“Core J2EE Patterns: Best Practices and Design Strategies”** by **Deepak Alur, John Crupi, and Dan Malks** defines **15 core design patterns** that were essential for building scalable, maintainable **J2EE (Java EE)** applications.

These patterns are categorized into **three tiers** reflecting the traditional **Java EE architecture**:

---

## 🧱 1. **Presentation Tier Patterns**

These patterns deal with handling client requests and controlling the view layer (usually via JSP/Servlets).

| Pattern Name            | Purpose                                                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Intercepting Filter** | Applies reusable filters (e.g., logging, authentication) before requests reach the servlet.                            |
| **Front Controller**    | Centralizes request handling into a single controller servlet.                                                         |
| **View Helper**         | Encapsulates logic in JSPs using helpers (like custom tags or JavaBeans) to separate business logic from presentation. |
| **Composite View**      | Creates complex views from modular subviews (composing reusable JSPs).                                                 |
| **Dispatcher View**     | Controller dispatches to view after business processing; view may contain logic.                                       |
| **Service to Worker**   | Controller handles logic and delegates view rendering to separate view components.                                     |

---

## ⚙️ 2. **Business Tier Patterns**

These handle the core business logic, business objects, transactions, and communication between components.

| Pattern Name                  | Purpose                                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| **Business Delegate**         | Hides the complexity of remote EJB calls; acts as a client-side proxy to business services.    |
| **Session Facade**            | Provides a unified interface to a set of EJBs; central point for business logic orchestration. |
| **Application Service**       | Encapsulates business logic in a POJO-like service, can be called from different tiers.        |
| **Service Locator**           | Reduces overhead of JNDI lookups for EJBs or services; provides a caching mechanism.           |
| **Transfer Object**           | Bundles data into a serializable object (DTO) to reduce remote call overhead.                  |
| **Transfer Object Assembler** | Combines multiple transfer objects into one to simplify data transfer to client layers.        |

---

## 🗃️ 3. **Integration Tier Patterns**

These patterns manage data access and communication with persistent storage or legacy systems.

| Pattern Name                 | Purpose                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| **Data Access Object (DAO)** | Abstracts and encapsulates all access to the data source; centralizes database interaction. |
| **Service Activator**        | Listens for asynchronous messages (e.g., via JMS) and triggers backend logic.               |
| **Domain Store**             | Separates domain object persistence logic from the object itself (like ORM mapping).        |

---

## 🧠 Summary Table

### 🔄 Presentation Tier

| Pattern             | Role                               |
| ------------------- | ---------------------------------- |
| Intercepting Filter | Preprocessing/filtering requests   |
| Front Controller    | Centralized request entry point    |
| View Helper         | Logic extraction from JSP          |
| Composite View      | Reusable view fragments            |
| Dispatcher View     | Logic then forward to view         |
| Service to Worker   | Controller + separate view handler |

---

### 🧠 Business Tier

| Pattern                   | Role                                     |
| ------------------------- | ---------------------------------------- |
| Business Delegate         | Hides remote service complexity          |
| Session Facade            | Simplifies and centralizes EJB access    |
| Application Service       | Stateless service layer (non-EJB)        |
| Service Locator           | Caches and locates services (EJBs, etc.) |
| Transfer Object (DTO)     | Data carrier between layers              |
| Transfer Object Assembler | Composes complex DTOs                    |

---

### 📦 Integration Tier

| Pattern                  | Role                               |
| ------------------------ | ---------------------------------- |
| DAO (Data Access Object) | Abstract DB interaction            |
| Service Activator        | React to async events (JMS)        |
| Domain Store             | Handles object persistence mapping |

---

## 🔚 Why These Patterns Matter (Even Today)

Although the book is from the **J2EE** (pre-Spring Boot) era, the patterns are still **fundamental principles** used in:

* Spring Boot (e.g., `@Service`, `@Controller`, `@Repository`)
* REST APIs
* Clean/Hexagonal Architecture
* Microservices (Session Facade → API Gateway, DTO still valid)

---

Would you like examples of these patterns implemented in **modern Spring Boot code**?
