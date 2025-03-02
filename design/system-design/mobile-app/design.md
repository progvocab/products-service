### **System Design for Mobile App with Push Notifications (Spring Boot + MongoDB + AWS EKS + FCM)**  

This system design will outline the **architecture**, **components**, **data flow**, and **scalability considerations** for a **mobile app** that receives push notifications from a **Spring Boot backend** deployed in **AWS EKS** with **MongoDB**.

---

# **1. High-Level Architecture**
### **Key Components**:
1. **Mobile App (React Native)**
   - Uses Firebase Cloud Messaging (FCM) to receive push notifications  
   - Authenticates with AWS Cognito  
   - Calls backend APIs via Apollo Client (GraphQL)  
   - Subscribes to topics for notifications  

2. **Backend (Spring Boot in AWS EKS)**
   - Exposes GraphQL API via AWS AppSync  
   - Handles business logic & authentication  
   - Stores data in MongoDB (hosted on AWS DocumentDB)  
   - Sends notifications using FCM  

3. **AWS Services**
   - **AWS EKS**: Hosts the Spring Boot backend  
   - **AWS AppSync**: GraphQL API for the mobile app  
   - **AWS Cognito**: User authentication (OIDC)  
   - **AWS S3**: Stores user profile images or documents  
   - **AWS Secret Manager**: Stores FCM API keys securely  

---

# **2. System Architecture Diagram**
```
                          +---------------------------+
                          |        Mobile App         |
                          | (React Native + Apollo)   |
                          |                           |
                          +------------+-------------+
                                       |
                  +--------------------+---------------------+
                  |                    |                     |
    +------------+-----------+ +--------+-----------+ +------v------+
    | AWS Cognito (OIDC)     | | AWS AppSync (GraphQL) | | AWS S3      |
    | (Auth & Token Issuer)  | | Handles GraphQL APIs | | File Storage |
    +------------+-----------+ +--------+-----------+ +------^------+
                  |                    |                     |
       +---------v---------+ +----------+----------+ +-------+-------+
       | Spring Boot API   | | Firebase Cloud Messaging | | AWS Secret Mgr |
       | (Push Service)    | | (Push Notification)    | | (Secure Keys)   |
       +-------------------+ +-----------------------+ +----------------+
                  |
      +----------------------------+
      | AWS DocumentDB (MongoDB)    |
      | Stores user & app data      |
      +----------------------------+
```
---

# **3. Data Flow**
### **User Registration & Authentication**
1. **User signs up/login** via React Native app  
2. App sends credentials to **AWS Cognito (OIDC)**  
3. Cognito validates & returns a **JWT Token**  
4. Mobile app stores the token & uses it for all API requests  

### **Fetching & Displaying Data**
1. Mobile app sends GraphQL queries via **AWS AppSync**  
2. AppSync fetches data from **Spring Boot Backend**  
3. Backend retrieves data from **MongoDB (AWS DocumentDB)**  
4. Data is returned to the mobile app & displayed  

### **Push Notification Flow**
1. Mobile app registers with **FCM** and gets a **device token**  
2. App sends the token to **Spring Boot Backend** for storage  
3. When an **important event** occurs (e.g., new project assigned), backend:  
   - Retrieves the user’s **FCM token**  
   - Sends a push notification using **Firebase Cloud Messaging (FCM)**  
4. Mobile app receives & displays the notification  

---

# **4. Database Schema (MongoDB)**
### **Users Collection**
```json
{
  "_id": "user123",
  "name": "John Doe",
  "email": "john@example.com",
  "role": "Engineer",
  "deviceToken": "fcmtoken123", 
  "departmentId": "dept456"
}
```
### **Projects Collection**
```json
{
  "_id": "project789",
  "name": "AI Research",
  "departmentId": "dept456",
  "assignedTo": ["user123", "user456"]
}
```

---

# **5. Scalability & Security Considerations**
### **Scalability**
✅ **AWS EKS (Kubernetes)**: Scales Spring Boot backend  
✅ **AWS AppSync (GraphQL)**: Efficient data fetching  
✅ **MongoDB (AWS DocumentDB)**: Handles millions of users  
✅ **Firebase FCM**: Scalable push notifications  

### **Security**
✅ **AWS Cognito (OIDC)**: Secure user authentication  
✅ **JWT Authentication**: Secure API access  
✅ **AWS Secret Manager**: Secure FCM keys & database credentials  
✅ **Role-Based Access Control (RBAC)**: Restrict access in GraphQL  

---

# **6. API Endpoints**
### **GraphQL Query (Get User Details)**
```graphql
query GetUser($id: ID!) {
  getUser(id: $id) {
    name
    email
    role
  }
}
```
### **REST API (Send Notification)**
```http
POST /notifications/send
Content-Type: application/json
Authorization: Bearer <JWT>
{
  "token": "fcmtoken123",
  "title": "New Project Assigned",
  "body": "You have been assigned to AI Research"
}
```

---

# **7. Deployment Strategy**
### **Step 1: Containerize Spring Boot Backend**
```dockerfile
FROM openjdk:17-jdk-slim
COPY target/*.jar app.jar
ENTRYPOINT ["java", "-jar", "app.jar"]
```
### **Step 2: Deploy on AWS EKS using Kubernetes**
- Create a **Deployment YAML** for Spring Boot:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
        - name: backend
          image: myrepo/springboot-app:latest
          ports:
            - containerPort: 8080
```
- Expose using a **LoadBalancer**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8080
  selector:
    app: backend
```

---

# **8. Summary**
✅ **React Native App** → Uses FCM for push notifications  
✅ **Spring Boot Backend** → Sends push notifications via FCM  
✅ **AWS Cognito** → Secure authentication with OIDC  
✅ **AWS AppSync** → GraphQL API for mobile app  
✅ **AWS EKS (Kubernetes)** → Scalable backend deployment  
✅ **MongoDB (AWS DocumentDB)** → Stores user & app data  

---

# **Next Steps**
🔹 **Implement Role-Based Access Control (RBAC) in AppSync**  
🔹 **Enable Offline Sync in App with AWS Amplify**  
🔹 **Set up AWS Lambda for event-driven notifications**  

Would you like a **detailed implementation guide for deploying on AWS EKS?**