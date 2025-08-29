Perfect 🔥 — let’s go deep into the **HTML Parser** and its **system design inside a browser**.
The HTML parser is one of the **first components** to run when a response with `Content-Type: text/html` arrives.

---

# 🔹 Role of the HTML Parser

* Converts raw **HTML text (bytes)** into a structured **DOM (Document Object Model) tree**.
* Handles malformed HTML gracefully (browsers are very forgiving).
* Triggers loading of external resources (CSS, JS, images).
* Works **incrementally** (can start rendering before the full HTML arrives → progressive rendering).

---

# 🔹 High-Level System Design

```
Network Response (HTML bytes)
         |
   Character Decoder (UTF-8, etc.)
         |
      Tokenizer (Lexical Analysis)
         |
     Tree Builder (Parsing Rules)
         |
     DOM Construction (Node Tree)
         |
  Interaction with CSS Parser + JS Engine
```

---

# 🔹 Step-by-Step Pipeline

### 1. **Input Stream & Character Decoding**

* Browser receives HTML bytes → decodes them into characters (using `Content-Type` charset or `<meta charset>`).

### 2. **Tokenizer (Lexical Analysis)**

* Splits characters into **tokens** like:

  * Start tags (`<html>`, `<body>`)
  * End tags (`</body>`)
  * Attributes (`class="header"`)
  * Text nodes (`Hello World`)
* Works as a **state machine** (HTML5 spec defines this in detail).

Example:

```html
<p class="msg">Hi</p>
```

→ Tokens:

* StartTag(p, class="msg")
* Text("Hi")
* EndTag(p)

### 3. **Tree Builder**

* Applies **parsing rules** (defined in HTML5 spec) to tokens.
* Builds **DOM Tree** by inserting nodes in the correct hierarchy.
* Handles **error recovery**:

  * `<b><i>Text</b></i>` → browser auto-corrects nesting.

### 4. **DOM Construction**

* Result is the **DOM tree**, which represents the HTML document structure.

Example DOM for above snippet:

```
Document
 └── html
      └── body
           └── p (class="msg")
                 └── "Hi"
```

---

# 🔹 Interaction with Other Components

1. **CSS Parser**

   * When `<link rel="stylesheet">` or `<style>` is found → HTML parser pauses DOM construction for **CSS fetch & parse**.
   * DOM + CSSOM combine into the **Render Tree**.

2. **JavaScript Engine**

   * When `<script>` tag appears:

     * If **synchronous**: parser pauses → JS executed → JS may modify DOM.
     * If **async/defer**: script downloaded in parallel, executed later.

3. **Preload Scanner**

   * Runs in parallel, scanning ahead for resources (CSS, JS, images) to start downloading early (optimization).

4. **Incremental Parsing**

   * Browser does not wait for the full HTML file → starts parsing + rendering progressively.

---

# 🔹 Error Handling (Robustness)

Browsers must handle malformed HTML:

* Missing tags: `<p>text` → auto-closes `<p>`.
* Misnested tags: `<b><i>Text</b></i>` → corrected into `<b><i>Text</i></b>`.
* Invalid attributes ignored.

This is why browsers behave differently from XML parsers (XML is strict).

---

# 🔹 System Design Challenges

1. **Streaming & Incremental Parsing** → Parse while downloading to reduce First Paint time.
2. **Spec Compliance** → Must follow the HTML Living Standard (quirks included).
3. **Error Recovery** → Handle broken HTML from the web.
4. **Coordination with CSS & JS** → Parser often pauses/resumes execution.
5. **Security** → Prevent injection attacks (e.g., script injection).

---

# 🔹 Simplified Example Walkthrough

Input:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Test</title>
    <script src="app.js"></script>
  </head>
  <body>
    <h1>Hello</h1>
  </body>
</html>
```

1. Tokenizer → generates tokens (`StartTag(html)`, `StartTag(head)`, …).
2. Tree Builder → builds DOM incrementally.
3. On `<script src="app.js">` → parser **pauses**, loads & executes JS.
4. JS may add/remove DOM nodes (`document.write`, `appendChild`).
5. Parser resumes → continues building DOM.
6. Final DOM is passed to **Render Tree construction**.

---

✅ In short:

* HTML Parser is a **streaming state machine**.
* Converts HTML → **DOM tree** while interacting with **CSS Parser** & **JS Engine**.
* Designed to be **forgiving, incremental, and fast**.

---

Would you like me to also explain **how DOM + CSSOM merge into the Render Tree and affect rendering performance (e.g., reflow/repaint system design)?** That’s the next step after HTML parsing.
