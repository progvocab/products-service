Yes, **Redis** can be integrated into your Spring Boot + MongoDB application to **enhance performance** and **reduce MongoDB load**, especially for **read-heavy operations** like frequently fetching orders.

---

### ✅ Use Cases for Redis in Your `Order` Application:

| Use Case                                     | How Redis Helps                                 |
| -------------------------------------------- | ----------------------------------------------- |
| **Cache frequently read orders**             | Reduce MongoDB queries by caching `Order` data. |
| **Store recently created or updated orders** | Allow fast access to recently modified data.    |
| **Session/token storage**                    | Useful if you're implementing authentication.   |
| **Rate limiting or counters**                | Track order creation frequency by users.        |

---

### ✅ Steps to Integrate Redis (as cache for `Order`):

---

### 1. **Add Redis dependency in `pom.xml`**

```xml
<!-- Spring Boot Redis Starter -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>

<!-- Optional: Embedded Redis for testing -->
<dependency>
    <groupId>it.ozimov</groupId>
    <artifactId>embedded-redis</artifactId>
    <version>0.7.2</version>
    <scope>test</scope>
</dependency>
```

---

### 2. **Configure Redis in `application.yml`**

```yaml
spring:
  redis:
    host: localhost
    port: 6379
```

---

### 3. **Enable Caching in Main App Class**

```java
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@EnableCaching
public class RecyclingApplication {
    public static void main(String[] args) {
        SpringApplication.run(RecyclingApplication.class, args);
    }
}
```

---

### 4. **Update `OrderService` to Use Redis Cache**

```java
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.CacheEvict;

@Service
public class OrderService {

    @Autowired
    private OrderRepository orderRepository;

    public Order createOrder(Order order) {
        return orderRepository.save(order);
    }

    @Cacheable(value = "orders", key = "#id")
    public Optional<Order> getOrderById(String id) {
        return orderRepository.findById(id);
    }

    @CacheEvict(value = "orders", key = "#id")
    public Optional<Order> updateOrder(String id, Order updatedOrder) {
        return orderRepository.findById(id).map(existingOrder -> {
            existingOrder.setOrderId(updatedOrder.getOrderId());
            existingOrder.setCustomerId(updatedOrder.getCustomerId());
            existingOrder.setTotalWeight(updatedOrder.getTotalWeight());
            existingOrder.setStatus(updatedOrder.getStatus());
            existingOrder.setOrderDate(updatedOrder.getOrderDate());
            return orderRepository.save(existingOrder);
        });
    }

    @CacheEvict(value = "orders", key = "#id")
    public boolean deleteOrder(String id) {
        if (orderRepository.existsById(id)) {
            orderRepository.deleteById(id);
            return true;
        }
        return false;
    }
}
```

---

### 5. **(Optional) Create Redis Config with TTL and Serializer**

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.cache.RedisCacheConfiguration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.RedisSerializationContext;

import java.time.Duration;

@Configuration
public class RedisConfig {

    @Bean
    public RedisCacheConfiguration cacheConfiguration() {
        return RedisCacheConfiguration.defaultCacheConfig()
                .entryTtl(Duration.ofMinutes(10)) // TTL for cache
                .disableCachingNullValues()
                .serializeValuesWith(RedisSerializationContext
                    .SerializationPair
                    .fromSerializer(new GenericJackson2JsonRedisSerializer()));
    }

    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(connectionFactory);
        template.setValueSerializer(new GenericJackson2JsonRedisSerializer());
        return template;
    }
}
```

---

### ✅ Summary of Cache Behavior

| Method                            | Annotation    | Effect               |
| --------------------------------- | ------------- | -------------------- |
| `getOrderById()`                  | `@Cacheable`  | Cache read result    |
| `updateOrder()` / `deleteOrder()` | `@CacheEvict` | Evict outdated cache |

---

Would you like:

* Full test data to verify cache behavior?
* Caching list of orders or filtered queries?
* Using `@CachePut` to update cache post-update?

Let me know how you'd like to expand Redis usage.
