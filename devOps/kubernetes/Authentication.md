### **🔹 OIDC Authentication & Certificate-Based User Management in Kubernetes (Kind Cluster)**  

To **secure Kubernetes access** in a **Kind cluster**, we’ll set up:  
1️⃣ **OIDC Authentication** (e.g., with Keycloak, Okta, or AWS Cognito)  
2️⃣ **Certificate-Based User Authentication** for additional security  

---

## **🔹 1. Setting Up OIDC Authentication in Kubernetes**
**🔹 Step 1: Install & Configure an OIDC Provider (Keycloak as Example)**  

### **1️⃣ Deploy Keycloak (OIDC Provider)**
```bash
docker run -d --name keycloak -p 8080:8080 -e KEYCLOAK_ADMIN=admin -e KEYCLOAK_ADMIN_PASSWORD=admin quay.io/keycloak/keycloak:latest start-dev
```
- Access Keycloak UI at **http://localhost:8080**  
- Create a new **realm** (`kubernetes-auth`)  
- Add a new **client** (`kube-client`) with:
  - **Client ID:** `kube-client`
  - **Client Protocol:** `openid-connect`
  - **Redirect URIs:** `https://kubernetes.default.svc/*`

---

### **2️⃣ Configure Kubernetes to Use OIDC**
Modify the **Kind cluster config** to enable OIDC:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: oidc-cluster
networking:
  podSubnet: "192.168.0.0/16"
  serviceSubnet: "10.96.0.0/12"
kubeadmConfigPatches:
- |
  apiVersion: kubeadm.k8s.io/v1beta3
  kind: ClusterConfiguration
  apiServer:
    extraArgs:
      oidc-issuer-url: "http://localhost:8080/realms/kubernetes-auth"
      oidc-client-id: "kube-client"
      oidc-username-claim: "preferred_username"
      oidc-groups-claim: "groups"
```

**Apply the Config:**  
```bash
kind create cluster --config=kind-oidc.yaml
```

---

## **🔹 2. Certificate-Based User Authentication**
**🔹 Step 1: Create a New Certificate Authority (CA)**
```bash
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -subj "/CN=K8s-CA" -days 365 -out ca.crt
```

**🔹 Step 2: Generate a User Certificate & Sign with CA**
```bash
openssl genrsa -out user.key 2048
openssl req -new -key user.key -subj "/CN=kind-user" -out user.csr
openssl x509 -req -in user.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out user.crt -days 365
```

**🔹 Step 3: Create a Kubernetes User with RBAC**
```bash
kubectl create rolebinding user-read-access --clusterrole=view --user=kind-user --namespace=default
```

**🔹 Step 4: Update kubeconfig for User**
```bash
kubectl config set-credentials kind-user --client-certificate=user.crt --client-key=user.key --embed-certs=true
kubectl config set-context kind-user-context --cluster=oidc-cluster --user=kind-user
kubectl config use-context kind-user-context
```

---

## **🔹 Summary**
✅ **OIDC Authentication** enables **Google, Keycloak, or Okta login**  
✅ **Certificates** ensure **secure user authentication**  
✅ **RBAC grants users specific permissions**  

Would you like **Helm-based Keycloak setup** or **AWS Cognito integration**? 🚀### **🔹 Deploy Keycloak on Kubernetes using Helm & Integrate OIDC with Kubernetes Authentication**  

We'll **deploy Keycloak** inside the **Kind cluster** using **Helm** and configure **OIDC authentication** for Kubernetes.  

---

## **🔹 1. Deploy Keycloak Using Helm in Kind Cluster**
### **1️⃣ Add Helm Repository & Install Keycloak**
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### **2️⃣ Deploy Keycloak**
```bash
helm install keycloak bitnami/keycloak --set auth.adminUser=admin --set auth.adminPassword=admin
```

- **Access Keycloak UI**:  
  ```bash
  kubectl port-forward svc/keycloak 8080:80
  ```
  - Open **http://localhost:8080** in a browser.
  - Login with **admin / admin**.

---

## **🔹 2. Configure Keycloak for OIDC**
### **1️⃣ Create a New Realm**
- In Keycloak UI, go to **Realm Settings** → **Create Realm** → Name: `kubernetes-auth`

### **2️⃣ Create OIDC Client for Kubernetes**
- **Clients → Create Client**
  - **Client ID**: `kube-client`
  - **Client Type**: `openid-connect`
  - **Root URL**: `https://kubernetes.default.svc`
  - **Valid Redirect URIs**: `https://kubernetes.default.svc/*`
  - **Save Changes**

### **3️⃣ Add a Test User**
- **Users → Add User**
  - **Username**: `kind-user`
  - **Set Password**: `mypassword`

---

## **🔹 3. Configure Kubernetes API Server for OIDC**
Modify **Kind cluster config** to enable OIDC:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: kind-oidc
kubeadmConfigPatches:
- |
  apiVersion: kubeadm.k8s.io/v1beta3
  kind: ClusterConfiguration
  apiServer:
    extraArgs:
      oidc-issuer-url: "http://keycloak.default.svc/realms/kubernetes-auth"
      oidc-client-id: "kube-client"
      oidc-username-claim: "preferred_username"
      oidc-groups-claim: "groups"
```

**Apply the Config:**  
```bash
kind create cluster --config=kind-oidc.yaml
```

---

## **🔹 4. Configure `kubectl` for OIDC Authentication**
1️⃣ **Obtain an OIDC Token** (Login to Keycloak)
```bash
TOKEN=$(curl -X POST "http://localhost:8080/realms/kubernetes-auth/protocol/openid-connect/token" \
 -d "client_id=kube-client" \
 -d "username=kind-user" \
 -d "password=mypassword" \
 -d "grant_type=password" | jq -r '.access_token')
```

2️⃣ **Update `kubectl` Config**
```bash
kubectl config set-credentials kind-user --token=$TOKEN
kubectl config set-context kind-context --cluster=kind-oidc --user=kind-user
kubectl config use-context kind-context
```

---

## **🔹 Summary**
✅ **Deploys Keycloak on Kubernetes**  
✅ **Integrates OIDC authentication with Kubernetes**  
✅ **Allows users to login via OIDC token**  

Would you like **OIDC authentication with AWS Cognito** as well? 🚀### **🔹 OIDC Authentication with AWS Cognito for Kubernetes (Kind Cluster)**  

We'll configure **AWS Cognito as an OIDC Provider** for authentication in a **Kind Kubernetes cluster**.  

---

## **🔹 1. Set Up AWS Cognito as an OIDC Provider**
### **1️⃣ Create a User Pool in AWS Cognito**
1. Open AWS **Cognito Console** → **Create a User Pool**  
2. **Name**: `kubernetes-auth-pool`  
3. Enable **username and password authentication**  
4. Add **an app client** (`kube-client`) with:  
   - **Allowed OAuth Flows**: Authorization Code Grant  
   - **Allowed OAuth Scopes**: `openid`, `email`, `profile`  
   - **Callback URLs**: `https://kubernetes.default.svc/callback`  

### **2️⃣ Get AWS Cognito OIDC Discovery URL**
```bash
export COGNITO_POOL_ID=<YourUserPoolId>
export AWS_REGION=<YourAWSRegion>
export OIDC_ISSUER="https://cognito-idp.$AWS_REGION.amazonaws.com/$COGNITO_POOL_ID"
```
- Example:  
  ```
  https://cognito-idp.us-east-1.amazonaws.com/us-east-1_ABC123
  ```

---

## **🔹 2. Configure Kind Cluster for OIDC**
Modify **Kind cluster configuration** to use Cognito:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: kind-oidc
kubeadmConfigPatches:
- |
  apiVersion: kubeadm.k8s.io/v1beta3
  kind: ClusterConfiguration
  apiServer:
    extraArgs:
      oidc-issuer-url: "$OIDC_ISSUER"
      oidc-client-id: "kube-client"
      oidc-username-claim: "email"
      oidc-groups-claim: "cognito:groups"
```

**Apply the Config:**  
```bash
kind create cluster --config=kind-oidc.yaml
```

---

## **🔹 3. Generate an OIDC Token for `kubectl` Authentication**
1️⃣ **Obtain an Access Token from AWS Cognito**
```bash
TOKEN=$(curl -X POST "https://$COGNITO_DOMAIN/oauth2/token" \
 -d "grant_type=password&client_id=kube-client&username=myuser&password=mypassword" \
 -H "Content-Type: application/x-www-form-urlencoded" | jq -r '.id_token')
```

2️⃣ **Configure `kubectl` for OIDC**
```bash
kubectl config set-credentials kind-user --token=$TOKEN
kubectl config set-context kind-context --cluster=kind-oidc --user=kind-user
kubectl config use-context kind-context
```

---

## **🔹 Summary**
✅ **AWS Cognito as OIDC provider for Kubernetes**  
✅ **OIDC authentication enabled in Kind cluster**  
✅ **Users log in with AWS Cognito credentials**  

Would you like **RBAC role-based access control** for users next? 🚀### **🔹 Implement RBAC for OIDC Users in Kubernetes (AWS Cognito + Kind Cluster)**  

After configuring **AWS Cognito as an OIDC provider**, we now enforce **Role-Based Access Control (RBAC)** so that users have specific permissions based on their group membership.

---

## **🔹 1. Define RBAC Roles in Kubernetes**
We'll create two groups in AWS Cognito:  
- **admin-group** → Full cluster admin access  
- **dev-group** → Read-only access  

### **1️⃣ Create a `ClusterRole` for Admins**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-admin-role
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
```

### **2️⃣ Create a `ClusterRole` for Developers (Read-Only)**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: dev-readonly-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "deployments"]
  verbs: ["get", "list"]
```

---

## **🔹 2. Bind OIDC Groups to Kubernetes Roles**
### **1️⃣ Bind the Admin Group to Full Access**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-binding
subjects:
- kind: Group
  name: admin-group  # Must match Cognito group name
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: cluster-admin-role
  apiGroup: rbac.authorization.k8s.io
```

### **2️⃣ Bind the Developer Group to Read-Only Access**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: dev-binding
subjects:
- kind: Group
  name: dev-group  # Must match Cognito group name
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: dev-readonly-role
  apiGroup: rbac.authorization.k8s.io
```

**Apply the RBAC Config:**  
```bash
kubectl apply -f admin-binding.yaml
kubectl apply -f dev-binding.yaml
```

---

## **🔹 3. Assign Users to AWS Cognito Groups**
1. **Go to AWS Cognito** → **User Groups**  
2. **Create `admin-group` & `dev-group`**  
3. **Assign users to the respective groups**  
4. **Ensure Cognito includes the `cognito:groups` claim** in the token

---

## **🔹 4. Test RBAC Permissions**
### **1️⃣ Authenticate as an Admin**
```bash
kubectl auth can-i create pods --as=admin@example.com
kubectl auth can-i delete deployments --as=admin@example.com
```
✅ Should return `yes`

### **2️⃣ Authenticate as a Developer**
```bash
kubectl auth can-i create pods --as=dev@example.com
kubectl auth can-i list services --as=dev@example.com
```
❌ Should return `no` for create  
✅ Should return `yes` for list  

---

## **🔹 Summary**
✅ **RBAC enforces user access control in Kubernetes**  
✅ **AWS Cognito groups (`admin-group`, `dev-group`) mapped to roles**  
✅ **OIDC users authenticate & get assigned permissions automatically**  

Would you like to **integrate external logging & monitoring (e.g., Prometheus, Grafana, AWS CloudWatch)?** 🚀