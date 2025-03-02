### **What is Nginx?**  

**Nginx (pronounced "engine-x")** is a high-performance, open-source **web server, reverse proxy, load balancer, and HTTP cache**. It is widely used for **serving static content, handling high-traffic websites, and improving application scalability**.  

---

## **1. Key Features of Nginx**  

✅ **Web Server** – Serves static files like HTML, CSS, JavaScript, and images.  
✅ **Reverse Proxy** – Forwards client requests to backend servers (e.g., Tomcat, Node.js, Python).  
✅ **Load Balancer** – Distributes traffic across multiple backend servers.  
✅ **HTTP Cache** – Stores and delivers frequently requested content to reduce load.  
✅ **Security & Rate Limiting** – Protects against DDoS attacks and limits excessive requests.  
✅ **Supports HTTPS/SSL** – Encrypts web traffic using TLS.  

---

## **2. How Nginx Works**  

### **A. Basic Architecture**  
- **Clients (Browsers, Mobile Apps)** → Send HTTP/HTTPS requests.  
- **Nginx** → Receives requests and processes them.  
- **Backend Servers (Tomcat, Node.js, Python, PHP, etc.)** → Nginx forwards requests to backend servers if needed.  
- **Databases (MySQL, PostgreSQL, MongoDB)** → The backend fetches data from the database.  

---

## **3. Nginx as a Web Server**  

Nginx can serve **static files** (HTML, CSS, images, videos) efficiently.  

### **Example Configuration for Static Website**  
```nginx
server {
    listen 80;
    server_name example.com;
    
    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```
✅ **Handles HTTP requests and serves static content.**  

---

## **4. Nginx as a Reverse Proxy**  

A **reverse proxy** sits between clients and backend servers, forwarding requests.  

### **Example: Forwarding Requests to a Backend (Tomcat, Node.js, or Spring Boot)**
```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
✅ **Client → Nginx → Backend (Tomcat, Node.js, etc.)**  

---

## **5. Nginx as a Load Balancer**  

Nginx can distribute traffic across multiple servers for **high availability & scalability**.  

### **Example: Load Balancing Between Multiple Backend Servers**
```nginx
upstream backend_servers {
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}

server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://backend_servers;
    }
}
```
✅ **Balances traffic across multiple servers.**  

---

## **6. Nginx for HTTPS & SSL/TLS**  

Nginx supports HTTPS by enabling SSL/TLS encryption.  

### **Example: Enabling HTTPS with SSL Certificate**
```nginx
server {
    listen 443 ssl;
    server_name example.com;

    ssl_certificate /etc/nginx/ssl/example.crt;
    ssl_certificate_key /etc/nginx/ssl/example.key;

    location / {
        proxy_pass http://localhost:8080;
    }
}
```
✅ **Ensures secure HTTPS communication.**  

---

## **7. Caching in Nginx**  

Nginx can cache static files or API responses for better performance.  

### **Example: Caching Static Files**
```nginx
location /static/ {
    expires 30d;
    add_header Cache-Control "public, max-age=2592000";
}
```
✅ **Reduces load and improves speed.**  

---

## **8. Security Features in Nginx**  

Nginx provides **security mechanisms** to protect web applications.  

### **Example: Rate Limiting to Prevent DDoS Attacks**
```nginx
limit_req_zone $binary_remote_addr zone=limit_per_ip:10m rate=5r/s;

server {
    location /api/ {
        limit_req zone=limit_per_ip burst=10;
        proxy_pass http://backend_servers;
    }
}
```
✅ **Limits requests to prevent server overload.**  

---

## **9. Deploying Nginx in Docker**  

Nginx can be deployed inside **Docker containers** for easy scalability.  

### **Example: Dockerfile for Nginx**
```dockerfile
FROM nginx:latest
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```
✅ **Lightweight & scalable deployment using containers.**  

---

## **10. Summary**  

| **Feature**          | **Nginx Functionality** |
|----------------------|----------------------|
| **Web Server**       | Serves static files like HTML, CSS, JS |
| **Reverse Proxy**    | Forwards requests to backend servers |
| **Load Balancer**    | Distributes traffic to multiple servers |
| **HTTPS/SSL**        | Encrypts web traffic |
| **Caching**         | Improves performance |
| **Security**         | Prevents DDoS attacks & applies rate limiting |
| **Docker Support**   | Can be deployed using Docker & Kubernetes |

Would you like a more **detailed implementation** for any specific use case?