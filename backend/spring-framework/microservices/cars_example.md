Here’s a complete breakdown of how to build a **Books Microservice** using the **CQRS (Command Query Responsibility Segregation)** pattern in **Spring Boot**.

---

## ✅ CQRS Overview

* **Command Side**: Handles `create`, `update`, `delete` (write operations).
* **Query Side**: Handles `get`, `search`, `list` (read operations).
* Each side has its own controller, service, and optionally separate databases.

---

## 📦 Project Structure

```
books-cqrs/
├── command/
│   ├── controller/
│   ├── service/
│   ├── dto/
│   └── entity/
├── query/
│   ├── controller/
│   ├── service/
│   └── dto/
├── repository/
├── BooksApplication.java
└── application.properties
```

---

## 📘 Entity Class

```java
package com.example.books.command.entity;

import jakarta.persistence.*;
import java.time.LocalDate;

@Entity
@Table(name = "books")
public class Book {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String title;
    private String author;
    private LocalDate publishedDate;

    // Getters and Setters
}
```

---

## 📝 Command DTO

```java
package com.example.books.command.dto;

import java.time.LocalDate;

public class BookCommandDTO {
    private String title;
    private String author;
    private LocalDate publishedDate;
}
```

---

## 🧠 Query DTO

```java
package com.example.books.query.dto;

import java.time.LocalDate;

public class BookQueryDTO {
    private Long id;
    private String title;
    private String author;
    private LocalDate publishedDate;
}
```

---

## ⚙️ Repository (Shared)

```java
package com.example.books.repository;

import com.example.books.command.entity.Book;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BookRepository extends JpaRepository<Book, Long> {
}
```

---

## 🚀 Command Service

```java
package com.example.books.command.service;

import com.example.books.command.dto.BookCommandDTO;
import com.example.books.command.entity.Book;
import com.example.books.repository.BookRepository;
import org.springframework.stereotype.Service;

@Service
public class BookCommandService {
    private final BookRepository repository;

    public BookCommandService(BookRepository repository) {
        this.repository = repository;
    }

    public Book createBook(BookCommandDTO dto) {
        Book book = new Book();
        book.setTitle(dto.getTitle());
        book.setAuthor(dto.getAuthor());
        book.setPublishedDate(dto.getPublishedDate());
        return repository.save(book);
    }

    public void deleteBook(Long id) {
        repository.deleteById(id);
    }
}
```

---

## 🔍 Query Service

```java
package com.example.books.query.service;

import com.example.books.query.dto.BookQueryDTO;
import com.example.books.repository.BookRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class BookQueryService {
    private final BookRepository repository;

    public BookQueryService(BookRepository repository) {
        this.repository = repository;
    }

    public List<BookQueryDTO> getAllBooks() {
        return repository.findAll().stream().map(book -> {
            BookQueryDTO dto = new BookQueryDTO();
            dto.setId(book.getId());
            dto.setTitle(book.getTitle());
            dto.setAuthor(book.getAuthor());
            dto.setPublishedDate(book.getPublishedDate());
            return dto;
        }).collect(Collectors.toList());
    }
}
```

---

## 🌐 Command Controller

```java
package com.example.books.command.controller;

import com.example.books.command.dto.BookCommandDTO;
import com.example.books.command.service.BookCommandService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/books/command")
public class BookCommandController {

    private final BookCommandService service;

    public BookCommandController(BookCommandService service) {
        this.service = service;
    }

    @PostMapping
    public ResponseEntity<?> create(@RequestBody BookCommandDTO dto) {
        return ResponseEntity.ok(service.createBook(dto));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        service.deleteBook(id);
        return ResponseEntity.noContent().build();
    }
}
```

---

## 🌐 Query Controller

```java
package com.example.books.query.controller;

import com.example.books.query.service.BookQueryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/books/query")
public class BookQueryController {

    private final BookQueryService service;

    public BookQueryController(BookQueryService service) {
        this.service = service;
    }

    @GetMapping
    public ResponseEntity<?> getAll() {
        return ResponseEntity.ok(service.getAllBooks());
    }
}
```

---

## ⚙️ `application.properties`

```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/booksdb
spring.datasource.username=postgres
spring.datasource.password=postgres
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
```

---

## 🧪 Sample Endpoints

| Method | Endpoint                  | Action        |
| ------ | ------------------------- | ------------- |
| POST   | `/api/books/command`      | Create book   |
| DELETE | `/api/books/command/{id}` | Delete book   |
| GET    | `/api/books/query`        | Get all books |

---

Would you like:

* A GitHub repo or downloadable ZIP?
* The project with MongoDB or event sourcing?
* Docker + Swagger integration?
