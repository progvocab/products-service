# **Deploying Amazon EKS Using Terraform**  

Amazon **EKS (Elastic Kubernetes Service)** allows you to run Kubernetes clusters on AWS. **Terraform** automates EKS deployment, managing networking, worker nodes, IAM roles, and more.

---

## **1️⃣ Prerequisites**
Before you start, ensure you have:
✅ **Terraform** installed (`terraform -v`)  
✅ **AWS CLI** configured (`aws configure`)  
✅ **kubectl** installed (`kubectl version`)  
✅ **IAM permissions** to create EKS resources  

---

## **2️⃣ Steps to Deploy EKS Using Terraform**
### **Step 1: Define Provider & Variables** (`variables.tf`)
```hcl
variable "region" { default = "us-east-1" }
variable "cluster_name" { default = "my-eks-cluster" }
variable "node_instance_type" { default = "t3.medium" }
variable "desired_capacity" { default = 2 }
variable "max_capacity" { default = 3 }
variable "min_capacity" { default = 1 }
```

---

### **Step 2: Configure AWS Provider** (`provider.tf`)
```hcl
provider "aws" {
  region = var.region
}
```

---

### **Step 3: Create a VPC & Subnets** (`vpc.tf`)
```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "3.19.0"

  name = "eks-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
}
```

---

### **Step 4: Create the EKS Cluster** (`eks.tf`)
```hcl
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "18.20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.27"
  subnet_ids      = module.vpc.private_subnets
  vpc_id          = module.vpc.vpc_id

  enable_irsa = true  # IAM Roles for Service Accounts (IRSA)

  node_groups = {
    eks_nodes = {
      desired_capacity = var.desired_capacity
      max_capacity     = var.max_capacity
      min_capacity     = var.min_capacity

      instance_types = [var.node_instance_type]
      key_name       = "my-eks-keypair"
    }
  }
}
```

---

### **Step 5: Output the Cluster Information** (`outputs.tf`)
```hcl
output "cluster_id" {
  description = "EKS Cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_endpoint" {
  description = "EKS Cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "kubeconfig" {
  description = "Kubeconfig command"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_id}"
}
```

---

### **Step 6: Initialize, Apply & Connect**
Run the following commands in your Terraform directory:

```sh
terraform init      # Initialize Terraform
terraform apply -auto-approve  # Deploy EKS
```

Once completed, update your **kubectl** configuration:

```sh
aws eks update-kubeconfig --region us-east-1 --name my-eks-cluster
kubectl get nodes  # Verify nodes are ready
```

---

## **3️⃣ Summary**
| **Step** | **Description** |
|----------|---------------|
| **1** | Set up variables (`variables.tf`) |
| **2** | Configure AWS provider (`provider.tf`) |
| **3** | Create VPC (`vpc.tf`) |
| **4** | Deploy EKS cluster (`eks.tf`) |
| **5** | Output cluster details (`outputs.tf`) |
| **6** | Run `terraform init` & `terraform apply` |
| **7** | Connect using `kubectl` |

✅ **EKS is now deployed with Terraform!**  
Would you like to add **autoscaling, logging, or monitoring** to the setup? 🚀