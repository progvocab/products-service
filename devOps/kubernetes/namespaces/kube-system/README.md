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
