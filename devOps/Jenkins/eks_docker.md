Here's a **Jenkinsfile** to deploy a Python application from a **GitHub repository** to **AWS EKS** using a **Docker container** and **Kubernetes manifests**.  

### **Prerequisites**  
Before running this pipeline, ensure:  
1. **Jenkins** has the required plugins:  
   - Pipeline  
   - Kubernetes CLI  
   - Docker Pipeline  
2. **AWS CLI** and **kubectl** are installed on the Jenkins agent.  
3. **Jenkins credentials** are set for:  
   - `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (for EKS)  
   - `DOCKER_CREDENTIALS_ID` (DockerHub or ECR login)  
   - `GITHUB_CREDENTIALS_ID` (for private repositories)  
4. **EKS Cluster** is configured, and `kubectl` is authenticated.  

---

### **Jenkinsfile**
```groovy
pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        EKS_CLUSTER_NAME = "my-eks-cluster"
        DOCKER_IMAGE = "my-python-app"
        DOCKER_TAG = "latest"
        DOCKER_REPO = "my-dockerhub-user/my-python-app"  // Change for ECR if needed
        KUBE_NAMESPACE = "default"
        KUBE_DEPLOYMENT = "python-app"
    }

    stages {
        stage('Clone Repository') {
            steps {
                script {
                    checkout([
                        $class: 'GitSCM',
                        branches: [[name: '*/main']], 
                        userRemoteConfigs: [[
                            url: 'https://github.com/your-repo/python-app.git',
                            credentialsId: 'GITHUB_CREDENTIALS_ID'
                        ]]
                    ])
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                withDockerRegistry([credentialsId: 'DOCKER_CREDENTIALS_ID', url: 'https://index.docker.io/v1/']) {
                    sh "docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_REPO}:${DOCKER_TAG}"
                    sh "docker push ${DOCKER_REPO}:${DOCKER_TAG}"
                }
            }
        }

        stage('Configure AWS and EKS') {
            steps {
                script {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'AWS_CREDENTIALS_ID']]) {
                        sh "aws eks --region ${AWS_REGION} update-kubeconfig --name ${EKS_CLUSTER_NAME}"
                    }
                }
            }
        }

        stage('Deploy to EKS') {
            steps {
                script {
                    sh """
                    kubectl set image deployment/${KUBE_DEPLOYMENT} python-app=${DOCKER_REPO}:${DOCKER_TAG} -n ${KUBE_NAMESPACE}
                    kubectl rollout status deployment/${KUBE_DEPLOYMENT} -n ${KUBE_NAMESPACE}
                    """
                }
            }
        }
    }

    post {
        success {
            echo "Deployment to AWS EKS successful!"
        }
        failure {
            echo "Deployment failed!"
        }
    }
}
```

---

### **Explanation**  
1. **Clone the GitHub repository**.  
2. **Build and push the Docker image** to DockerHub (or ECR).  
3. **Configure AWS CLI** and update `kubectl` config for EKS.  
4. **Deploy the updated Docker image** to AWS EKS.  

---

### **Next Steps**  
- Update `DOCKER_REPO` if using **AWS ECR** instead of DockerHub.  
- Ensure `kubectl` is configured to interact with your EKS cluster.  
- Create a `Kubernetes deployment.yaml` for initial deployment if needed.  

Would you like help with the **Kubernetes deployment.yaml**?