@Service
public class ServiceA {
    
    @Autowired
    private ServiceB serviceB;

    @Transactional
    public void methodA() {
        // Business logic for Service A
        serviceB.methodB(); // Calls methodB in Service B
    }
}

@Service
public class ServiceB {

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void methodB() {
        // Business logic for Service B
    }
}
