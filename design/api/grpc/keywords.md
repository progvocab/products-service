### **Protocol Buffers (ProtoBuf) Syntax and Keywords Explained**  

A **ProtoBuf (.proto) file** defines structured data models using a **schema-based approach**. It includes **messages, enums, services**, and other elements used for serialization and communication, particularly in **gRPC APIs**.  

---

## **1. Basic Syntax of a Proto File**  

```proto
syntax = "proto3";  // Specify ProtoBuf version

package example;  // Optional: Define package name

import "google/protobuf/timestamp.proto";  // Import built-in types

// Define an enumeration
enum Status {
  UNKNOWN = 0;
  ACTIVE = 1;
  INACTIVE = 2;
}

// Define a message (data structure)
message User {
  int32 id = 1;  // Unique field number
  string name = 2;
  repeated string emails = 3;  // List of emails
  Status status = 4;  // Enum field
  optional string nickname = 5;  // Optional field
  google.protobuf.Timestamp created_at = 6;  // Timestamp field
}

// Define a gRPC service
service UserService {
  rpc GetUser (UserRequest) returns (UserResponse);
}

// Request and response messages
message UserRequest {
  int32 id = 1;
}

message UserResponse {
  User user = 1;
}
```

---

## **2. Keywords and Features in ProtoBuf**  

### **A. `syntax` (Version Specification)**
```proto
syntax = "proto3";  // Use ProtoBuf version 3 (default for gRPC)
```
- `proto2` → Older version, requires specifying **required/optional** fields  
- `proto3` → Simplified syntax, fields are **optional by default**  

---

### **B. `package` (Namespace Definition)**
```proto
package example;
```
- Helps avoid **name conflicts** when using multiple ProtoBuf files  
- Equivalent to **namespaces in C++/Java**  

---

### **C. `import` (Importing Other Proto Files)**
```proto
import "google/protobuf/timestamp.proto";  // Import built-in type
import "common.proto";  // Import custom proto file
```
- Imports another `.proto` file for **reuse of messages and types**  
- Can import **Google-provided types** (`timestamp.proto`, `empty.proto`)  

---

### **D. `message` (Defining Data Structures)**
```proto
message User {
  int32 id = 1;
  string name = 2;
}
```
- Represents **data models** (like a class or struct)  
- Fields must have **unique numbers** for backward compatibility  

#### **Field Numbering Rules**  
✅ `1-15` → Optimized for small fields  
✅ `16-2047` → Slightly larger encoding  
❌ `19000-19999` → Reserved for internal use  

---

### **E. Data Types in ProtoBuf**  

| **Type**      | **Example**           | **Description**               |
|--------------|----------------------|------------------------------|
| `int32`      | `int32 age = 1;`      | 32-bit integer               |
| `int64`      | `int64 timestamp = 2;` | 64-bit integer               |
| `float`      | `float price = 3;`     | 32-bit floating-point        |
| `double`     | `double amount = 4;`   | 64-bit floating-point        |
| `bool`       | `bool is_active = 5;`  | Boolean value                |
| `string`     | `string name = 6;`     | UTF-8 text string            |
| `bytes`      | `bytes image = 7;`     | Raw binary data              |

Example:
```proto
message Product {
  int32 id = 1;
  float price = 2;
  bool available = 3;
}
```

---

### **F. `enum` (Defining Enumerations)**
```proto
enum Status {
  UNKNOWN = 0;
  ACTIVE = 1;
  INACTIVE = 2;
}
```
- **First value must be `0`**  
- Useful for **defining fixed sets of values**  
- Defaults to `0` if not set  

Example usage:
```proto
message User {
  Status status = 1;
}
```

---

### **G. `repeated` (Defining Lists)**
```proto
message User {
  repeated string emails = 1;
}
```
- Creates a **list/array** of elements  
- Default is an **empty list**  

Example with integers:
```proto
message Scores {
  repeated int32 marks = 1;
}
```

---

### **H. `optional` (Optional Fields)**
```proto
message User {
  optional string nickname = 1;
}
```
- Used in **proto3** for fields that may not always exist  
- Reduces payload size if the field is missing  

---

### **I. `oneof` (Mutually Exclusive Fields)**
```proto
message Contact {
  oneof contact_info {
    string email = 1;
    string phone = 2;
  }
}
```
- Only **one field can be set at a time**  
- Saves memory if only **one option** is needed  

Example:
```proto
message Notification {
  oneof method {
    string email = 1;
    string sms = 2;
  }
}
```

---

### **J. `map` (Key-Value Pairs)**
```proto
message User {
  map<string, int32> scores = 1;
}
```
- Defines **key-value pairs**  
- Keys must be **strings or integers**  

Example:
```proto
message Config {
  map<string, string> settings = 1;
}
```

---

### **K. `service` (gRPC Service Definition)**
```proto
service UserService {
  rpc GetUser (UserRequest) returns (UserResponse);
}
```
- Defines **gRPC API services**  
- Uses `rpc` (Remote Procedure Call) for defining methods  

Example:
```proto
service OrderService {
  rpc GetOrder (OrderRequest) returns (OrderResponse);
  rpc StreamOrders (OrderRequest) returns (stream OrderResponse);
}
```

---

### **L. Streaming in gRPC**
- **Client Streaming** → Multiple requests, single response  
- **Server Streaming** → Single request, multiple responses  
- **Bidirectional Streaming** → Multiple requests and responses  

Example:
```proto
service ChatService {
  rpc ChatStream (stream ChatMessage) returns (stream ChatResponse);
}
```

---

## **3. Reserved Keywords in ProtoBuf**  

✅ **Avoid using these words as field names**:  
- `reserved` → Reserve field numbers to avoid conflicts  
- `extend` → Used in `proto2` for extending messages  
- `option` → Used for setting ProtoBuf options  

Example:
```proto
message OldMessage {
  reserved 1, 2, 3;
  reserved "old_field";
}
```

---

## **4. Advanced Features**  

| Feature      | Usage Example |
|-------------|--------------|
| **Default Values** | `int32 age = 1 [default = 18];` |
| **Nested Messages** | `message Address { string city = 1; }` |
| **Extensions (proto2)** | `extend Message { string extra = 100; }` |

---

## **Conclusion**  
- **ProtoBuf is compact, fast, and strongly typed**  
- **Used in gRPC, IoT, and high-performance APIs**  
- **Supports enums, lists, key-value pairs, streaming, and services**  

Would you like a **real-world ProtoBuf + gRPC example** or a **comparison with Avro/Thrift**?