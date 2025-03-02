To send **push notifications** to mobile users from a **Spring Boot backend (deployed in AWS EKS with MongoDB)**, you can use **AWS SNS (Simple Notification Service) or Firebase Cloud Messaging (FCM)**. Here’s how you can implement this:

---

## **1. Choose a Push Notification Provider**
### **Option 1: Firebase Cloud Messaging (FCM)**
- ✅ Best for **iOS & Android**
- ✅ Free & scalable  
- ✅ Provides **topic-based** & **direct device messaging**
- 🔹 Requires **FCM SDK** integration in the mobile app  

### **Option 2: AWS SNS (Simple Notification Service)**
- ✅ Good for sending notifications to **APNs (iOS) and FCM (Android)**
- ✅ Supports **SMS & Email alerts**  
- 🔹 More setup required for mobile push  

For **mobile apps**, **FCM is preferred** due to its reliability and ease of use.

---

## **2. Set Up Firebase Cloud Messaging (FCM)**
### **Step 1: Create an FCM Project**
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project
3. Add **iOS/Android app** and download the **`google-services.json`** (for Android) or **`GoogleService-Info.plist`** (for iOS)
4. Enable **Cloud Messaging API** under Firebase settings  

### **Step 2: Get FCM Server Key**
1. In Firebase Console, go to **Project Settings > Cloud Messaging**
2. Copy the **Server Key** (This is needed for backend integration)

---

## **3. Implement Push Notification in Spring Boot Backend**
### **Step 1: Add Dependencies**
Add the following dependencies to your `pom.xml`:
```xml
<dependency>
    <groupId>com.google.firebase</groupId>
    <artifactId>firebase-admin</artifactId>
    <version>9.2.0</version>
</dependency>
```
For **Gradle**:
```gradle
implementation 'com.google.firebase:firebase-admin:9.2.0'
```

---

### **Step 2: Configure Firebase in Spring Boot**
1. Place the **`google-services.json`** in `src/main/resources/`
2. Initialize Firebase in a `FirebaseConfig` class:

```java
import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;
import java.io.FileInputStream;
import java.io.IOException;

@Configuration
public class FirebaseConfig {

    @PostConstruct
    public void init() throws IOException {
        FileInputStream serviceAccount =
                new FileInputStream("src/main/resources/google-services.json");

        FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .build();

        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.initializeApp(options);
        }
    }
}
```

---

### **Step 3: Create a Push Notification Service**
```java
import com.google.firebase.messaging.FirebaseMessaging;
import com.google.firebase.messaging.Message;
import com.google.firebase.messaging.Notification;
import org.springframework.stereotype.Service;

@Service
public class PushNotificationService {

    public void sendPushNotification(String token, String title, String body) {
        Message message = Message.builder()
                .setToken(token)
                .setNotification(Notification.builder()
                        .setTitle(title)
                        .setBody(body)
                        .build())
                .build();

        try {
            String response = FirebaseMessaging.getInstance().send(message);
            System.out.println("Successfully sent message: " + response);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

---

### **Step 4: Create a Controller to Send Notifications**
```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/notifications")
public class NotificationController {

    @Autowired
    private PushNotificationService pushNotificationService;

    @PostMapping("/send")
    public String sendNotification(@RequestParam String token, 
                                   @RequestParam String title, 
                                   @RequestParam String body) {
        pushNotificationService.sendPushNotification(token, title, body);
        return "Notification sent!";
    }
}
```
- This API allows you to send notifications to a specific **device token**.

---

## **4. How Mobile App Receives Notifications**
### **React Native (Expo) FCM Integration**
1. Install dependencies:
```bash
npm install firebase @react-native-firebase/messaging
```
2. Initialize Firebase in **React Native App**:
```javascript
import messaging from '@react-native-firebase/messaging';

async function requestUserPermission() {
  const authStatus = await messaging().requestPermission();
  if (authStatus === messaging.AuthorizationStatus.AUTHORIZED) {
    console.log('Notification permission granted.');
  }
}

async function getDeviceToken() {
  const token = await messaging().getToken();
  console.log("FCM Token:", token);
  return token;
}

requestUserPermission();
getDeviceToken();
```
- Send the **FCM token** to your backend via an API.

---

## **5. Send Notifications to All Users (Topic-Based Messaging)**
Instead of sending to **individual device tokens**, use **topic-based messaging**:
### **Step 1: Subscribe Users to a Topic**
```java
import com.google.firebase.messaging.FirebaseMessaging;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PushNotificationService {

    public void subscribeToTopic(String token, String topic) {
        try {
            FirebaseMessaging.getInstance().subscribeToTopic(List.of(token), topic);
            System.out.println("Subscribed to topic: " + topic);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void sendNotificationToTopic(String topic, String title, String body) {
        Message message = Message.builder()
                .setTopic(topic)
                .setNotification(Notification.builder()
                        .setTitle(title)
                        .setBody(body)
                        .build())
                .build();

        try {
            String response = FirebaseMessaging.getInstance().send(message);
            System.out.println("Successfully sent topic message: " + response);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```
### **Step 2: Expose API to Subscribe to Topic**
```java
@RestController
@RequestMapping("/notifications")
public class NotificationController {

    @Autowired
    private PushNotificationService pushNotificationService;

    @PostMapping("/subscribe")
    public String subscribeToTopic(@RequestParam String token, @RequestParam String topic) {
        pushNotificationService.subscribeToTopic(token, topic);
        return "Subscribed to topic: " + topic;
    }

    @PostMapping("/sendToTopic")
    public String sendNotificationToTopic(@RequestParam String topic, 
                                          @RequestParam String title, 
                                          @RequestParam String body) {
        pushNotificationService.sendNotificationToTopic(topic, title, body);
        return "Notification sent to topic: " + topic;
    }
}
```
- Use this when you need **broadcast notifications**.

---

## **6. Deploy Spring Boot in AWS EKS**
- **Containerize Spring Boot** (`Dockerfile`):
```dockerfile
FROM openjdk:17-jdk-slim
COPY target/*.jar app.jar
ENTRYPOINT ["java", "-jar", "app.jar"]
```
- **Deploy to EKS with Helm or Kubernetes YAML**  
- Use **AWS Secrets Manager** to store **FCM credentials** securely.

---

## **7. Summary**
✅ **FCM Setup** → Created Firebase Project & Obtained API Key  
✅ **Spring Boot Integration** → Configured FCM & created notification service  
✅ **React Native App** → Integrated FCM SDK to receive push notifications  
✅ **Deployment in AWS EKS** → Containerized and deployed using Kubernetes  

---

### **Next Steps**
🔹 Implement **AWS SNS** for **multi-platform (SMS, Email, Push)** notifications  
🔹 Use **WebSockets** in Spring Boot for **real-time alerts**  
🔹 Secure API with **OAuth2 & JWT authentication**  

Would you like help in **deploying Spring Boot in EKS with Helm?**