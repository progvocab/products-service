Here’s a complete example of a **Book Library Microservice** using the **Command Query Responsibility Segregation (CQRS)** pattern in Spring Boot:

---

### 🧱 Technologies Used

* Java 17+
* Spring Boot
* Spring Data JPA (for query model)
* MapStruct (for DTO mapping)
* H2/PostgreSQL
* Lombok

---

### 📚 CQRS Structure Overview

| Layer       | Responsibility                          |
| ----------- | --------------------------------------- |
| **Command** | Write operations (create/update/delete) |
| **Query**   | Read operations (fetch by id, list)     |

---

### 🗂️ Project Structure

```
book-library-cqrs/
├── controller/
│   ├── BookCommandController.java
│   └── BookQueryController.java
├── command/
│   ├── BookCommandService.java
│   └── BookCommandRepository.java
├── query/
│   ├── BookQueryService.java
│   └── BookQueryRepository.java
├── dto/
│   ├── BookCreateRequest.java
│   ├── BookUpdateRequest.java
│   └── BookResponse.java
├── model/
│   └── Book.java
└── BookLibraryApplication.java
```

---

### ✅ 1. Entity

```java
package com.example.model;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Getter @Setter @NoArgsConstructor @AllArgsConstructor
public class Book {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String title;
    private String author;
    private int year;
}
```

---

### ✅ 2. DTOs

```java
package com.example.dto;

public record BookCreateRequest(String title, String author, int year) {}
public record BookUpdateRequest(String title, String author, int year) {}
public record BookResponse(Long id, String title, String author, int year) {}
```

---

### ✅ 3. Command Repository & Service

```java
package com.example.command;

import com.example.model.Book;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BookCommandRepository extends JpaRepository<Book, Long> {}
```

```java
package com.example.command;

import com.example.dto.BookCreateRequest;
import com.example.dto.BookUpdateRequest;
import com.example.model.Book;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class BookCommandService {

    private final BookCommandRepository repository;

    public Book create(BookCreateRequest request) {
        Book book = new Book(null, request.title(), request.author(), request.year());
        return repository.save(book);
    }

    public Book update(Long id, BookUpdateRequest request) {
        Book book = repository.findById(id).orElseThrow();
        book.setTitle(request.title());
        book.setAuthor(request.author());
        book.setYear(request.year());
        return repository.save(book);
    }

    public void delete(Long id) {
        repository.deleteById(id);
    }
}
```

---

### ✅ 4. Query Repository & Service

```java
package com.example.query;

import com.example.model.Book;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BookQueryRepository extends JpaRepository<Book, Long> {}
```

```java
package com.example.query;

import com.example.dto.BookResponse;
import com.example.model.Book;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class BookQueryService {

    private final BookQueryRepository repository;

    public BookResponse findById(Long id) {
        Book b = repository.findById(id).orElseThrow();
        return new BookResponse(b.getId(), b.getTitle(), b.getAuthor(), b.getYear());
    }

    public List<BookResponse> findAll() {
        return repository.findAll().stream()
            .map(b -> new BookResponse(b.getId(), b.getTitle(), b.getAuthor(), b.getYear()))
            .toList();
    }
}
```

---

### ✅ 5. REST Controllers

```java
package com.example.controller;

import com.example.command.BookCommandService;
import com.example.dto.BookCreateRequest;
import com.example.dto.BookUpdateRequest;
import com.example.model.Book;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/commands/books")
@RequiredArgsConstructor
public class BookCommandController {

    private final BookCommandService service;

    @PostMapping
    public Book create(@RequestBody BookCreateRequest request) {
        return service.create(request);
    }

    @PutMapping("/{id}")
    public Book update(@PathVariable Long id, @RequestBody BookUpdateRequest request) {
        return service.update(id, request);
    }

    @DeleteMapping("/{id}")
    public void delete(@PathVariable Long id) {
        service.delete(id);
    }
}
```

```java
package com.example.controller;

import com.example.dto.BookResponse;
import com.example.query.BookQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/queries/books")
@RequiredArgsConstructor
public class BookQueryController {

    private final BookQueryService service;

    @GetMapping
    public List<BookResponse> findAll() {
        return service.findAll();
    }

    @GetMapping("/{id}")
    public BookResponse findById(@PathVariable Long id) {
        return service.findById(id);
    }
}
```

---

### ✅ 6. Application Main Class

```java
@SpringBootApplication
public class BookLibraryApplication {
    public static void main(String[] args) {
        SpringApplication.run(BookLibraryApplication.class, args);
    }
}
```

---

### ✅ 7. `application.properties`

```properties
spring.datasource.url=jdbc:h2:mem:books
spring.datasource.driver-class-name=org.h2.Driver
spring.datasource.username=sa
spring.datasource.password=

spring.jpa.hibernate.ddl-auto=update
spring.h2.console.enabled=true
```

---

Would you like me to upload this to a GitHub repo or ZIP the complete working project?
