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

### **🔹 Implementing IAM-Based Access Control for Kubernetes with AWS IAM Roles**  
We’ll configure **AWS IAM authentication** for your **Kind Kubernetes cluster**, allowing users to authenticate using **IAM roles instead of static credentials**.

---

## **🔹 1. Install AWS IAM Authenticator**
### **1️⃣ Install `aws-iam-authenticator`**
```bash
curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.22.2/2021-11-02/bin/linux/amd64/aws-iam-authenticator
chmod +x aws-iam-authenticator
mv aws-iam-authenticator /usr/local/bin/
```
Verify Installation:
```bash
aws-iam-authenticator version
```

---

## **🔹 2. Enable IAM Authentication in Kind Cluster**
### **1️⃣ Modify `kind` Config to Use IAM Authenticator**
```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: kind-iam
kubeadmConfigPatches:
- |
  apiVersion: kubeadm.k8s.io/v1beta3
  kind: ClusterConfiguration
  apiServer:
    extraArgs:
      authentication-token-webhook-config-file: "/etc/kubernetes/aws-iam-authenticator.yaml"
nodes:
- role: control-plane
- role: worker
```

Create the **AWS IAM Authenticator Config File** (`aws-iam-authenticator.yaml`):
```yaml
apiVersion: v1
kind: Config
clusters:
  - name: kind-iam-cluster
    cluster:
      server: https://kind-control-plane:6443
users:
  - name: iam-auth-user
    user:
      exec:
        apiVersion: client.authentication.k8s.io/v1beta1
        command: aws-iam-authenticator
        args:
          - token
          - -i
          - kind-iam-cluster
contexts:
  - name: kind-iam-context
    context:
      cluster: kind-iam-cluster
      user: iam-auth-user
current-context: kind-iam-context
```

Apply Config:
```bash
kubectl apply -f aws-iam-authenticator.yaml
```

---

## **🔹 3. Map IAM Roles to Kubernetes RBAC**
### **1️⃣ Create IAM Role for Kubernetes Users**
Create an IAM Role:
```bash
aws iam create-role --role-name KubernetesAdmin \
  --assume-role-policy-document file://assume-role-policy.json
```

`assume-role-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:user/dev-user"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Attach a Policy to Allow Kubernetes Access:
```bash
aws iam attach-role-policy --role-name KubernetesAdmin \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy
```

### **2️⃣ Map IAM Role to Kubernetes RBAC**
Create a `ConfigMap` to Associate IAM Role with Kubernetes Group:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-auth
  namespace: kube-system
data:
  mapRoles: |
    - rolearn: arn:aws:iam::123456789012:role/KubernetesAdmin
      username: admin-user
      groups:
        - system:masters
```
Apply ConfigMap:
```bash
kubectl apply -f aws-auth-configmap.yaml
```

---

## **🔹 4. Authenticate with IAM Role**
### **1️⃣ Generate an AWS IAM Token**
```bash
TOKEN=$(aws eks get-token --cluster-name kind-iam-cluster --role KubernetesAdmin | jq -r .status.token)
```
### **2️⃣ Configure `kubectl` to Use the IAM Token**
```bash
kubectl config set-credentials iam-user --token=$TOKEN
kubectl config set-context kind-iam-context --cluster=kind-iam-cluster --user=iam-user
kubectl config use-context kind-iam-context
```

### **3️⃣ Test IAM-Based Access**
```bash
kubectl auth can-i get pods --as=admin-user
```

---

## **🔹 Summary**
✅ **AWS IAM authentication for Kubernetes**  
✅ **IAM roles mapped to Kubernetes RBAC**  
✅ **No need for static user credentials**  

Would you like **AWS OIDC Federated Authentication for Single Sign-On (SSO) with AWS IAM?** 🚀

### **🔹 Implementing AWS OIDC Federated Authentication for Kubernetes (SSO with AWS IAM)**  
We will integrate **AWS IAM OIDC authentication** with Kubernetes, allowing users to log in with **AWS SSO or an external identity provider (IdP)** (e.g., Okta, Google Workspace, Azure AD).

---

## **🔹 1. Set Up AWS OIDC Provider**
### **1️⃣ Create an OIDC Identity Provider in AWS IAM**
1. **Get the OIDC Issuer URL for Your IdP** (e.g., Okta, Google Workspace, Azure AD)
   ```bash
   aws eks describe-cluster --name my-cluster --query "cluster.identity.oidc.issuer" --output text
   ```
   Example Output:  
   ```
   https://oidc.eks.<region>.amazonaws.com/id/EXAMPLE-ID
   ```
2. **Create the OIDC Provider in AWS IAM**
   ```bash
   aws iam create-open-id-connect-provider \
       --url "https://oidc.eks.<region>.amazonaws.com/id/EXAMPLE-ID" \
       --client-id-list sts.amazonaws.com \
       --thumbprint-list <THUMBPRINT>
   ```
   - Replace `<THUMBPRINT>` with the fingerprint of the provider certificate.

---

## **🔹 2. Create an IAM Role for Kubernetes Users**
### **1️⃣ Create an IAM Role for Kubernetes Access**
```bash
aws iam create-role --role-name OIDC-Kubernetes-Access \
  --assume-role-policy-document file://assume-role-policy.json
```
`assume-role-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/oidc.eks.<region>.amazonaws.com/id/EXAMPLE-ID"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.<region>.amazonaws.com/id/EXAMPLE-ID:sub": "system:serviceaccount:kube-system:oidc-auth"
        }
      }
    }
  ]
}
```
Replace `<AWS_ACCOUNT_ID>` and `<region>` accordingly.

### **2️⃣ Attach Permissions to the IAM Role**
```bash
aws iam attach-role-policy --role-name OIDC-Kubernetes-Access \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy
```

---

## **🔹 3. Map the IAM Role to Kubernetes RBAC**
Create an `aws-auth` ConfigMap:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-auth
  namespace: kube-system
data:
  mapRoles: |
    - rolearn: arn:aws:iam::<AWS_ACCOUNT_ID>:role/OIDC-Kubernetes-Access
      username: oidc-user
      groups:
        - system:masters
```
Apply the ConfigMap:
```bash
kubectl apply -f aws-auth.yaml
```

---

## **🔹 4. Authenticate with OIDC & AWS IAM**
### **1️⃣ Generate an AWS STS Token for Authentication**
```bash
TOKEN=$(aws eks get-token --cluster-name my-cluster | jq -r .status.token)
```
### **2️⃣ Configure `kubectl` to Use OIDC Token**
```bash
kubectl config set-credentials oidc-user --token=$TOKEN
kubectl config set-context oidc-context --cluster=my-cluster --user=oidc-user
kubectl config use-context oidc-context
```
### **3️⃣ Verify Authentication**
```bash
kubectl auth can-i get pods --as=oidc-user
```

---

## **🔹 Summary**
✅ **AWS OIDC authentication for Kubernetes**  
✅ **Users log in via AWS SSO or an external IdP**  
✅ **IAM role-based Kubernetes RBAC access**  

Would you like **to enable multi-cluster authentication across AWS regions?** 🚀