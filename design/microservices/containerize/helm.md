Here's how you can create a basic **Helm chart** to deploy your Spring Boot Docker image to **Kubernetes**.

---

## **Step 1: Scaffold Helm Chart**

```bash
helm create springboot-app
```

This will create a structure like:

```
springboot-app/
  ├── Chart.yaml
  ├── values.yaml
  ├── templates/
       ├── deployment.yaml
       ├── service.yaml
       └── ...
```

---

## **Step 2: Customize Chart.yaml**

```yaml
# springboot-app/Chart.yaml
apiVersion: v2
name: springboot-app
description: A Helm chart for deploying Spring Boot app
version: 0.1.0
appVersion: "1.0"
```

---

## **Step 3: Set Values in values.yaml**

```yaml
# springboot-app/values.yaml
replicaCount: 2

image:
  repository: your-dockerhub-username/springboot-app
  pullPolicy: IfNotPresent
  tag: latest

service:
  type: ClusterIP
  port: 8080

ingress:
  enabled: false

resources: {}

nodeSelector: {}

tolerations: []

affinity: []
```

---

## **Step 4: Update deployment.yaml**

Edit `templates/deployment.yaml` to use the correct container port:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "springboot-app.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ include "springboot-app.name" . }}
  template:
    metadata:
      labels:
        app: {{ include "springboot-app.name" . }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - containerPort: 8080
```

---

## **Step 5: Deploy with Helm**

```bash
helm install springboot-app ./springboot-app
```

If you make changes, you can upgrade:

```bash
helm upgrade springboot-app ./springboot-app
```

---

Would you like me to generate a complete minimal Helm chart for you (as files) or include optional Ingress, HPA, or ConfigMaps?