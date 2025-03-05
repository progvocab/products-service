To ensure that **test cases are executed** and the **Jenkins build fails if tests fail**, follow these steps based on the build tool you're using:  

---

### **If using pytest (for Python projects)**
Modify the **Jenkinsfile** to include a **test stage** before building the Docker image:  

#### **Updated Jenkinsfile (with Pytest)**
```groovy
pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        EKS_CLUSTER_NAME = "my-eks-cluster"
        ECR_REPOSITORY = "my-python-app"
        IMAGE_TAG = "latest"
        ACCOUNT_ID = "123456789012"
        ECR_REGISTRY = "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
        FULL_IMAGE_NAME = "${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
        HELM_RELEASE_NAME = "python-app"
        HELM_CHART_PATH = "./helm/python-app"
        KUBE_NAMESPACE = "default"
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

        stage('Install Dependencies') {
            steps {
                script {
                    sh "pip install -r requirements.txt"
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    sh "pytest --junitxml=report.xml"
                }
            }
        }

        stage('Login to AWS ECR') {
            steps {
                script {
                    withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'AWS_CREDENTIALS_ID']]) {
                        sh "aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}"
                    }
                }
            }
        }

        stage('Build and Push Docker Image to ECR') {
            steps {
                script {
                    sh "docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} ."
                    sh "docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${FULL_IMAGE_NAME}"
                    sh "docker push ${FULL_IMAGE_NAME}"
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

        stage('Deploy to EKS using Helm') {
            steps {
                script {
                    sh """
                    helm upgrade --install ${HELM_RELEASE_NAME} ${HELM_CHART_PATH} \
                        --namespace ${KUBE_NAMESPACE} \
                        --set image.repository=${ECR_REGISTRY}/${ECR_REPOSITORY} \
                        --set image.tag=${IMAGE_TAG} \
                        --set namespace=${KUBE_NAMESPACE}
                    """
                }
            }
        }
    }

    post {
        success {
            echo "Deployment to AWS EKS using Helm successful!"
        }
        failure {
            echo "Deployment failed!"
        }
    }
}
```

---

### **How This Ensures Tests Run and Affect Build**
✅ **Adds a `pytest` test stage** before the Docker build.  
✅ **Fails the pipeline if any test case fails.**  
✅ **Generates a test report (`report.xml`) for Jenkins reporting.**  

---

### **Next Steps**
1. **Make sure tests are located inside a `tests/` directory**.  
2. **Ensure `pytest` is installed** (`pip install pytest`).  
3. **Jenkins can be configured to publish test reports** using `report.xml`.  

Would you like to configure **JUnit-style test reporting in Jenkins** for better visibility?