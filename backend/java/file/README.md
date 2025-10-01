
---

## 1. **Byte Stream (raw data)**

* Works with **binary data** (`.jpg`, `.mp3`, `.zip`, etc.).
* Reads/Writes **8-bit bytes**.
* Classes:

  * Input → `FileInputStream`
  * Output → `FileOutputStream`

👉 Example:

```java
FileInputStream in = new FileInputStream("image.jpg");
FileOutputStream out = new FileOutputStream("copy.jpg");

int data;
while ((data = in.read()) != -1) {
    out.write(data);
}
in.close();
out.close();
```

---

## 2. **Character Stream (text data)**

* Works with **Unicode characters** (16-bit).
* Best for **text files** (`.txt`, `.csv`, `.xml`).
* Classes:

  * Input → `FileReader`
  * Output → `FileWriter`

👉 Example:

```java
FileReader reader = new FileReader("input.txt");
FileWriter writer = new FileWriter("output.txt");

int ch;
while ((ch = reader.read()) != -1) {
    writer.write(ch);
}
reader.close();
writer.close();
```

---

## 3. **Buffering**

* Reading/writing **one byte/char at a time** is slow.
* **Buffered Streams/Writers** use an **internal buffer** to reduce disk I/O.
* Classes:

  * `BufferedInputStream`, `BufferedOutputStream`
  * `BufferedReader`, `BufferedWriter`

👉 Example:

```java
BufferedReader br = new BufferedReader(new FileReader("input.txt"));
BufferedWriter bw = new BufferedWriter(new FileWriter("output.txt"));

String line;
while ((line = br.readLine()) != null) {
    bw.write(line);
    bw.newLine();
}
bw.flush();  // forces buffered data to file
br.close();
bw.close();
```

---

## 4. **`flush()`**

* Ensures **all buffered data is written** to the destination immediately.
* Important when writing partial data that must be visible before closing.

---

✅ **Summary**

* **Byte Streams** = binary data, use `InputStream` / `OutputStream`.
* **Character Streams** = text data, use `Reader` / `Writer`.
* **Buffers** improve performance by batching I/O.
* **flush()** ensures no data is stuck in buffer before closing.

---
