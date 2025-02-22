# **Helm Architecture Overview**
Helm is a **package manager for Kubernetes** that simplifies application deployment by managing Kubernetes manifests using **Charts**. It provides a structured way to install, upgrade, and manage Kubernetes applications.

---

## **Helm Architecture Components**
Helm has a **client-server architecture**, with the following key components:

1. **Helm CLI (Client)**
2. **Helm Charts**
3. **Chart Repository**
4. **Kubernetes API Server**
5. **Release Management**

---

### **1. Helm CLI (Client)**
- The **Helm CLI** is a command-line tool used to interact with Helm.
- It performs actions such as:
  - Installing, upgrading, rolling back, and deleting releases.
  - Managing repositories (`helm repo add`, `helm repo update`).
  - Rendering templates (`helm template`).
- It directly communicates with the **Kubernetes API Server**.

#### **Example Helm Commands**
```sh
helm install myapp mychart/       # Install an application
helm upgrade myapp mychart/       # Upgrade an existing release
helm rollback myapp 1             # Rollback to a previous release
helm list                         # List installed releases
```

---

### **2. Helm Charts**
- **Charts** are Helm's packaging format, containing all Kubernetes manifests for an application.
- A Helm chart consists of:
  - **`Chart.yaml`** → Chart metadata.
  - **`values.yaml`** → Default configuration values.
  - **`templates/`** → Kubernetes manifest templates.
  - **`requirements.yaml`** → Lists dependencies.

#### **Example Chart Directory**
```
mychart/
│── Chart.yaml        # Metadata
│── values.yaml       # Default values
│── templates/        # Kubernetes templates
│── requirements.yaml # Dependencies
```

---

### **3. Chart Repository**
- A **Helm Chart Repository** is a location where Helm charts are stored and shared.
- Helm CLI pulls charts from repositories like **ArtifactHub, Bitnami, or private repos**.
- Default public repo: `https://charts.helm.sh/stable`

#### **Adding and Using a Chart Repo**
```sh
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm search repo bitnami/nginx
```

---

### **4. Kubernetes API Server**
- Helm interacts with **Kubernetes API Server** to deploy applications.
- When running `helm install`, the Helm client sends manifests to **Kubernetes API Server**, which schedules and runs resources.

#### **How Helm Interacts with Kubernetes**
1. Helm templates are rendered.
2. Kubernetes manifests are created.
3. Helm sends them to **Kubernetes API Server**.
4. Kubernetes schedules workloads accordingly.

---

### **5. Release Management**
- A **Release** is a deployed instance of a Helm chart.
- Each release has a **name** and a **revision number** (versioned for rollbacks).
- Helm stores release information in Kubernetes secrets/configmaps.

#### **Example Helm Releases**
```sh
helm list
NAME    NAMESPACE  REVISION  STATUS   CHART         APP VERSION
myapp   default    2         deployed mychart-1.0.0 1.16.0
```

- **`helm rollback myapp 1`** → Rolls back `myapp` to the first version.
- **`helm history myapp`** → Shows release history.

---

## **Helm Architecture Flow**
1. **User runs a Helm command (`helm install myapp mychart/`).**
2. **Helm CLI pulls the chart from the repository** (or local).
3. **Helm renders the templates** using `values.yaml`.
4. **Helm sends the manifest files** to Kubernetes API Server.
5. **Kubernetes schedules and runs** the resources.
6. **Helm stores release data** for future upgrades/rollbacks.

---

## **Summary of Helm Components**
| Component        | Description |
|-----------------|-------------|
| **Helm CLI** | Command-line tool for managing Helm charts/releases. |
| **Helm Charts** | Packages that define Kubernetes resources. |
| **Chart Repository** | Stores and distributes Helm charts. |
| **Kubernetes API Server** | Deploys workloads using Helm manifests. |
| **Release Management** | Handles installations, upgrades, and rollbacks. |

Helm simplifies Kubernetes deployments by **abstracting complexity** and **enabling reusable, configurable templates**.