Excellent — let’s dive into **`Content-Disposition`**, a key HTTP header that controls how browsers (or clients) handle a file or response from a server.

---

## 🧩 What is `Content-Disposition`?

`Content-Disposition` is an **HTTP response header** used to specify how content should be **displayed**, **stored**, or **handled** by the client.

It tells the browser:

> “Should this file be displayed inline in the browser or downloaded as an attachment?”

---

## 🧱 Syntax

```http
Content-Disposition: <type>; filename="<filename>"
```

### Common Types

| Type         | Meaning                                                            |
| ------------ | ------------------------------------------------------------------ |
| `inline`     | Display the content directly in the browser (if supported).        |
| `attachment` | Prompt the user to **download** the file instead of displaying it. |

---

## 🧮 Example 1 — View Inline (like PDF or image)

If you want the browser to display the file:

```http
Content-Disposition: inline; filename="document.pdf"
```

✅ The browser opens `document.pdf` in a new tab if it supports PDFs.

---

## 🧮 Example 2 — Force Download

To force a download dialog:

```http
Content-Disposition: attachment; filename="report.csv"
```

✅ Browser pops up *“Save as”* dialog box, suggesting `report.csv` as filename.

---

## 🧮 Example 3 — API Download Response

If you’re building an API that lets users download a file:

```http
HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: attachment; filename="invoice_2025.pdf"
```

When users hit this endpoint, their browser will download `invoice_2025.pdf`.

---

## ⚙️ Technical Breakdown

| Parameter   | Description                                         |
| ----------- | --------------------------------------------------- |
| `type`      | Either `inline` or `attachment`.                    |
| `filename`  | Suggested name for the file when downloaded.        |
| `filename*` | RFC 5987 encoding for UTF-8 or non-ASCII filenames. |

### Example with UTF-8 filename:

```http
Content-Disposition: attachment; filename="report.pdf"; filename*=UTF-8''r%C3%A9sum%C3%A9.pdf
```

This ensures that filenames with accents or non-English characters appear correctly.

---

## 🧩 Example in Backend Code

### **Spring Boot / Java**

```java
@GetMapping("/download")
public ResponseEntity<Resource> download() {
    Resource file = new FileSystemResource("/tmp/report.pdf");
    return ResponseEntity.ok()
        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"report.pdf\"")
        .contentType(MediaType.APPLICATION_PDF)
        .body(file);
}
```

### **Express.js / Node.js**

```js
app.get('/download', (req, res) => {
  res.download('/tmp/report.pdf', 'report.pdf');
});
```

Node.js sets:

```
Content-Disposition: attachment; filename="report.pdf"
```

---

## 🧠 Why It Matters

| Use Case           | Purpose                                                |
| ------------------ | ------------------------------------------------------ |
| File downloads     | Control how browsers download API responses            |
| Invoices / Reports | Suggest proper filenames dynamically                   |
| Inline preview     | Allow in-browser display (e.g., PDFs, images)          |
| Security           | Prevent file name injection (sanitize filename values) |

---

## ⚠️ Security Notes

* Always **sanitize filenames** to prevent path traversal attacks (`../../etc/passwd`).
* Avoid exposing server paths.
* Use quoted values for filenames:
  ✅ `filename="safe.pdf"`
  ❌ `filename=safe.pdf` (less safe in some cases)

---

## 🧮 Summary Table

| Mode           | Header                                                         | Browser Behavior                   |
| -------------- | -------------------------------------------------------------- | ---------------------------------- |
| Inline         | `Content-Disposition: inline; filename="file.pdf"`             | Display file in tab (if supported) |
| Attachment     | `Content-Disposition: attachment; filename="file.pdf"`         | Force download prompt              |
| UTF-8 Filename | `Content-Disposition: attachment; filename*=UTF-8''résumé.pdf` | Preserve special chars             |

---

Would you like me to show how browsers like **Chrome**, **Firefox**, and **Safari** behave differently with `Content-Disposition` (especially for inline PDFs and images)?
