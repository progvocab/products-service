# Kube System Namespace

```mermaid 
flowchart TD

    subgraph KS["kube-system Namespace"]
        A[kube-apiserver<br>Control Plane]
        B[kube-scheduler<br>Control Plane]
        C[kube-controller-manager<br>Control Plane]
        D[cloud-controller-manager<br>Cloud Integration]

        E[etcd<br>Cluster Store]

        F[kube-proxy<br>Node Networking]
        G[coredns<br>Cluster DNS]

        H[metrics-server<br>Resource Metrics]
        I[cluster-autoscaler<br>Node Autoscaling]

        J[calico/flannel/weave<br>CNI Plugins]

        K[kubelet-bootstrap<br>Bootstrap Tokens]
        L[addon-manager<br>Addon Installation]
    end
```


### Eks 

```mermaid 
flowchart TD

    subgraph CP["EKS Managed Control Plane (Not Visible as Pods)"]
        A[EKS API Server]
        B[EKS Controller Manager]
        C[EKS Scheduler]
        D[EKS etcd Cluster]
    end

    subgraph KS["kube-system Namespace (User-visible Pods on Worker Nodes)"]
        subgraph NET["Networking"]
            E[aws-node<br>Amazon VPC CNI]
            F[kube-proxy]
            G[coredns]
        end

        subgraph MET[Metrics / Autoscaling]
            H[metrics-server]
            I["cluster-autoscaler (runs in kube-system when installed)"]
        end

        subgraph ADD["EKS Add-ons"]
            J[aws-eks-node-agent]
            K[aws-eks-pod-identity-agent]
            L[ebs-csi-controller and ebs-csi-node]
            M[efs-csi-controller and efs-csi-node]
        end
    end

    CP --> KS
```


### **node-level diagram for Amazon EKS**

 showing exactly **which kube-system components run on each worker node**, how they interact with the **control plane**, and how **network traffic flows**.

```mermaid
flowchart TD

    subgraph CP["EKS Control Plane (Managed by AWS)"]
        A[EKS API Server]
        B[EKS Controller Manager]
        C[EKS Scheduler]
        D[EKS etcd]
    end

    subgraph N1["Worker Node 1"]
        E1[kubelet]
        F1[kube-proxy]
        G1[aws-node  VPC CNI]
        H1["coredns (if scheduled)"]
        I1[pods/workloads]
        J1[ebs-csi-node]
        K1[efs-csi-node]
    end

   

    A --> E1
  

    E1 --> F1
   

    F1 --> G1
    

    I1 --> G1
   

    G1 --> A
   
```

### What This Diagram Shows

* **kubelet** on each worker node communicates directly with the **EKS API Server**.
* **aws-node (VPC CNI)** manages ENIs and pod IP allocation on EC2 nodes.
* **kube-proxy** programs iptables rules for cluster networking.
* **coredns**, **CSI drivers**, and **workload pods** run as regular pods in `kube-system` or other namespaces.
* All pod-to-service and pod-to-pod traffic flows through **VPC CNI + kube-proxy**.

More :  **traffic flow from ALB → Node → Pod via VPC CNI**, or the **EKS IAM + Pod Identity integration flow**.
The **`kube-system`** namespace is a special Kubernetes namespace used to manage system components that are essential for the functioning of a Kubernetes cluster. It contains critical infrastructure components, controllers, and services that are necessary for cluster operations.

### Key Aspects of the `kube-system` Namespace:

1. **Purpose**:
   - The `kube-system` namespace is reserved for Kubernetes system services and components. It ensures that these essential services are logically separated from user-deployed applications and workloads, providing better organization and management.

2. **Common Components Deployed in `kube-system`**:
   - **API Server**: The Kubernetes API server that handles all REST operations and provides the interface for cluster interactions.
   - **Controller Manager**: Manages controllers that regulate the state of the cluster, such as the replication controller and endpoint controller.
   - **Scheduler**: Schedules pods on nodes based on resource availability and policies.
   - **etcd**: A key-value store that holds the entire cluster's state.
   - **CoreDNS**: Provides DNS services within the cluster for service discovery.
   - **Kube-proxy**: Maintains network rules on nodes to allow communication between pods and services.
   - **Metrics Server**: Provides resource metrics (CPU and memory) for pods and nodes used by Horizontal Pod Autoscaler and Kubernetes dashboard.

3. **Networking**:
   - Components like **CoreDNS** and **kube-proxy** help manage internal networking and service discovery, ensuring that pods can communicate with each other and with external services.

4. **Cluster Management**:
   - The controllers in the `kube-system` namespace handle various cluster management tasks such as node lifecycle management, maintaining desired states of resources, and managing the deployment of critical pods.

5. **Logging and Monitoring**:
   - Add-ons like **Fluentd**, **Heapster** (deprecated), and **Metrics Server** for cluster-level logging and resource monitoring are often found in the `kube-system` namespace.

6. **Security**:
   - Critical security components, such as network policies and cluster role bindings, may also be managed within the `kube-system` namespace to ensure secure operations of system components.

7. **Default Namespace for Cluster Add-ons**:
   - Many Kubernetes add-ons and plugins that enhance cluster capabilities, such as network plugins (Calico, Flannel), storage plugins, and Ingress controllers, are often deployed in the `kube-system` namespace.

### Managing `kube-system` Namespace:

- **Listing Resources**: You can inspect the resources running in the `kube-system` namespace with the following command:
  ```bash
  kubectl get all -n kube-system
  ```

- **Monitoring and Troubleshooting**: Since the `kube-system` namespace contains essential components, it's important to monitor and troubleshoot any issues in this namespace to ensure the cluster's health and stability.

### Best Practices:

1. **Avoid Manual Changes**: Be cautious when making changes in the `kube-system` namespace, as it can affect the entire cluster's functionality.
2. **Monitoring**: Regularly monitor the health and performance of components in the `kube-system` namespace.
3. **Isolation**: Keep application workloads and custom resources separate from the `kube-system` namespace to avoid conflicts and accidental disruptions.

The `kube-system` namespace is vital for the proper functioning of a Kubernetes cluster, as it hosts all the necessary system services and components that keep the cluster operational.