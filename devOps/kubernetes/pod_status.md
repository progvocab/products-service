In Kubernetes, pods go through different lifecycle states and conditions. Below is a detailed explanation of the various **Pod Statuses** you may encounter, including `NotReady`, `Running`, `Error`, `Evicted`, `CrashLoopBackOff`, `ImagePullBackOff`, and others.  

---

## **1. Running (`Running`)**  
**Description**: The pod is running successfully, and all containers inside the pod are functioning as expected.  

**How to check:**  
```sh
kubectl get pods
```
**Example output:**
```
NAME        READY   STATUS    RESTARTS   AGE
my-pod      1/1     Running   0          10m
```

---

## **2. Not Ready (`NotReady`)**  
**Description**: The pod is running, but at least one of its containers is not ready to serve requests.  

**Causes**:
- A container inside the pod has failed its **readiness probe**.
- The node running the pod has issues (e.g., resource pressure).
- The pod is still initializing.  

**How to check:**  
```sh
kubectl describe pod <pod-name>
```
Check for events related to **readiness probe failures**.

---

## **3. Error (`Error`)**  
**Description**: One or more containers in the pod have failed due to some issue.  

**Common Causes**:
- The application inside the container has crashed.
- The container exited with a non-zero exit code.
- There was an issue mounting a volume.  

**How to check:**  
```sh
kubectl logs <pod-name>
```
or  
```sh
kubectl describe pod <pod-name>
```

---

## **4. Evicted (`Evicted`)**  
**Description**: The pod was terminated by the Kubernetes scheduler due to resource constraints on the node (e.g., memory or disk pressure).  

**Causes**:
- Node running out of memory (OOMKilled).
- Disk pressure on the node.  
- The node was drained or cordoned (e.g., during maintenance).  

**How to check:**  
```sh
kubectl get pod <pod-name> -o yaml
```
Check for:
```yaml
status:
  reason: Evicted
```

**Solution**:
- Check node resource usage:  
  ```sh
  kubectl describe node <node-name>
  ```
- Consider increasing node resources or adjusting pod resource requests/limits.

---

## **5. CrashLoopBackOff (`CrashLoopBackOff`)**  
**Description**: The container inside the pod keeps crashing and restarting in a loop. Kubernetes detects repeated failures and increases the time between restarts.  

**Causes**:
- The application inside the container crashes immediately after starting.  
- Incorrect configuration or missing environment variables.  
- Insufficient resources (CPU, memory).  
- Readiness/Liveness probe failures.  

**How to check:**  
```sh
kubectl describe pod <pod-name>
```
Look for events like:
```
Back-off restarting failed container
```

**Solution**:
- Check logs:
  ```sh
  kubectl logs <pod-name>
  ```
- Check container exit code:
  ```sh
  kubectl get pod <pod-name> -o yaml | grep exitCode
  ```
- Ensure correct configuration and sufficient resources.

---

## **6. ImagePullBackOff (`ImagePullBackOff`)**  
**Description**: Kubernetes is unable to pull the container image from the registry, and it retries pulling with an increasing delay.  

**Causes**:
- The image name or tag is incorrect.  
- The container registry is down or unreachable.  
- No credentials provided for private registries.  
- Network issues.  

**How to check:**  
```sh
kubectl describe pod <pod-name>
```
Look for:
```
Failed to pull image "myregistry.com/myimage": ImagePullBackOff
```

**Solution**:
- Verify image name and tag.  
- Check registry access and authentication.  
- Test manually:  
  ```sh
  docker pull myregistry.com/myimage
  ```

---

## **7. ErrImagePull (`ErrImagePull`)**  
**Description**: Similar to `ImagePullBackOff`, but this is the initial error before Kubernetes starts backing off.  

**Solution**: Same as **ImagePullBackOff**.

---

## **8. Completed (`Completed`)**  
**Description**: The pod has run to completion, which is normal for **Job** or **CronJob** workloads.  

**How to check:**  
```sh
kubectl get pods
```
**Example output:**
```
NAME        READY   STATUS      RESTARTS   AGE
my-job      0/1     Completed   0          15m
```

---

## **9. Terminating (`Terminating`)**  
**Description**: The pod is being shut down but is waiting for cleanup operations (e.g., finalizers, grace period).  

**How to check:**  
```sh
kubectl get pods
```
**Solution**:
- Force delete if stuck:  
  ```sh
  kubectl delete pod <pod-name> --grace-period=0 --force
  ```

---

## **10. Pending (`Pending`)**  
**Description**: The pod has been created but is waiting for resources (e.g., node scheduling, image pull).  

**Causes**:
- No available nodes with enough resources.  
- Persistent Volume (PV) is not bound.  
- Image is being pulled.  

**Solution**:
- Check node conditions:  
  ```sh
  kubectl describe node <node-name>
  ```
- Check storage and scheduling events:  
  ```sh
  kubectl describe pod <pod-name>
  ```

---

## **11. NodeLost (`NodeLost`)**  
**Description**: The node where the pod was running became unreachable, and Kubernetes marked the pod as lost.  

**Solution**:
- Check node status:  
  ```sh
  kubectl get nodes
  ```
- Restart the node or reschedule the pod.

---

## **12. Unknown (`Unknown`)**  
**Description**: The pod status is unknown because the node is unreachable.  

**Solution**:
- Investigate the node using:  
  ```sh
  kubectl describe node <node-name>
  ```

---

### **Summary Table**
| Status               | Description |
|----------------------|-------------|
| **Running**          | Pod is running successfully. |
| **NotReady**         | At least one container is not ready. |
| **Error**            | A container has failed. |
| **Evicted**          | Pod was removed due to resource constraints. |
| **CrashLoopBackOff** | Container keeps crashing and restarting. |
| **ImagePullBackOff** | Kubernetes is unable to pull the image. |
| **ErrImagePull**     | Image pull failed initially. |
| **Completed**        | Pod completed execution (Job or CronJob). |
| **Terminating**      | Pod is shutting down. |
| **Pending**         | Pod is waiting for scheduling or resources. |
| **NodeLost**        | Pod lost due to node failure. |
| **Unknown**        | Pod status is unknown due to node issues. |

Would you like help troubleshooting a specific pod issue?