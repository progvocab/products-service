A news feed is a classic system design interview and a common architecture in social platforms. The main challenge is efficiently delivering a personalized, reverse-chronological feed to millions of users.

### 1. Requirements

### Functional Requirements

* Users can create posts.
* Users can follow/unfollow other users.
* Users see a personalized news feed containing posts from people they follow.
* Feed is ordered by time (or ranking algorithm).
* Pagination/infinite scrolling.
* Like, comment, and share posts (optional).

### Non-Functional Requirements

* Low latency (feed loads in <200 ms).
* High availability (99.9%+).
* Horizontal scalability.
* Eventual consistency is acceptable.
* Support millions of users.



### 2. High-Level Architecture

```text
                    +----------------------+
                    |   Web / Mobile App   |
                    +----------+-----------+
                               |
                         API Gateway
                               |
        +----------------------+----------------------+
        |                      |                      |
   User Service         Feed Service          Post Service
        |                      |                      |
        |                Feed Cache (Redis)          |
        |                      |                      |
        |               Feed Database               |
        |                      |                      |
        |                      |                      |
        +-----------+----------+----------------------+
                    |
             Message Queue (Kafka)
                    |
          Feed Generation Worker
                    |
      Fan-out to Followers Service
                    |
              Feed Storage
```

 

### 3. Data Model

### User

```text
User
----
userId
name
profileImage
createdAt
```
 

### Follow Relationship

```text
Follow
-------
followerId
followingId
createdAt
```

Example:

```
Alice → Bob
Alice → Charlie
David → Bob
```

 

### Post

```text
Post
----
postId
authorId
content
mediaUrl
timestamp
visibility
```

 

### Feed Entry

```text
FeedItem
--------
userId
postId
authorId
timestamp
```

Each user has a personalized feed.

 

### 4. Database Choices

| Data      | Database                          |
| --------- | --------------------------------- |
| Users     | SQL / PostgreSQL                  |
| Followers | Graph DB or Cassandra             |
| Posts     | Cassandra / DynamoDB              |
| Feed      | Redis + Cassandra                 |
| Media     | Object Storage (S3, Blob Storage) |

 

### 5. API Design

### Create Post

```
POST /posts
```

Request

```json
{
  "content":"Hello World"
}
```

 

### Follow User

```
POST /follow
```

```json
{
    "userId":101,
    "followUserId":205
}
```

 

### Get Feed

```
GET /feed?page=1&limit=20
```

Returns

```json
[
   {
      "postId":123,
      "author":"Bob",
      "content":"Good Morning"
   }
]
```

 
### 6. Feed Generation Strategies

### Push Model (Fan-out on Write)

When Bob posts:

```
Bob creates Post

↓

Find Bob's followers

↓

Insert post into every follower's feed
```

Example

```
Bob

Followers

Alice
John
Mike
Sara

↓

Write into

Alice Feed
John Feed
Mike Feed
Sara Feed
```

**Pros**

* Very fast feed reads.
* Low latency for end users.

**Cons**

* Expensive writes for users with many followers.

Best for:

* Most social networks where reads greatly outnumber writes.

 

### Pull Model (Fan-out on Read)

Bob posts.

Nothing is copied.

When Alice opens her feed:

```
Find everyone Alice follows

↓

Read their latest posts

↓

Merge

↓

Sort

↓

Return
```

**Pros**

* Very cheap writes.
* Simpler storage.

**Cons**

* Slow reads.

Best for:

* Platforms with relatively few follows per user.

 

### Hybrid Model (Used by Large Social Networks)

Most users:

* Push model (precompute feeds).

Celebrity accounts:

* Pull model (fetch posts on demand).

Why?

If a celebrity has 50 million followers, pushing every post to every follower is prohibitively expensive.

 

### 7. Feed Generation Flow

```text
User creates post

        |

 Save Post Database

        |

 Publish Event

        |

Kafka

        |

Feed Worker

        |

Find Followers

        |

Write Feed Entries

        |

Redis Cache

        |

User opens Feed

        |

Read Redis

        |

Return Feed
```



### 8. Scaling

### Partition Posts

```
Shard 1
Shard 2
Shard 3

based on postId
```



### Partition Followers

```
Follower Table

Shard by userId
```



### Cache

Keep latest 100–500 posts per user in Redis.

```
Redis

Feed:Alice

[
Post10
Post9
Post8
]
```

If cache misses, retrieve older posts from persistent storage.



### 9. Pagination

Avoid offset-based pagination.

Instead use a cursor.

```
GET /feed?cursor=171982001
```

Returns next 20 posts older than the cursor timestamp.

Advantages:

* Faster queries.
* Stable results even when new posts arrive.



### 10. Reliability

* Replicate databases.
* Replicate Kafka brokers.
* Use Redis clusters.
* Retry failed feed updates.
* Dead-letter queue for failed events.
* Idempotent feed generation to prevent duplicate entries.



### 11. End-to-End Flow

```text
                 User Creates Post
                         |
                         v
                 +---------------+
                 | Post Service  |
                 +---------------+
                         |
                  Save Post Database
                         |
                         v
                    Event (Kafka)
                         |
                         v
               Feed Generation Worker
                         |
               Get List of Followers
                         |
       +-----------------+------------------+
       |                 |                  |
       v                 v                  v
  Alice Feed        John Feed         Mike Feed
       |                 |                  |
       +-----------------+------------------+
                         |
                         v
                 Redis Feed Cache
                         |
                         v
                 User Requests Feed
                         |
                         v
                  Feed Service API
                         |
                         v
                 Return Top 20 Posts
```

### 12. Key Design Considerations

* **Push vs. Pull**: Use push (fan-out on write) for most users and pull (fan-out on read) for celebrities or accounts with very large follower counts.
* **Caching**: Cache recent feed entries to minimize read latency.
* **Asynchronous Processing**: Use a message queue (e.g., Kafka) so post creation isn't delayed by feed generation.
* **Scalability**: Shard users, posts, and follower relationships independently to distribute load.
* **Consistency**: Eventual consistency is acceptable; users may see a slight delay before new posts appear.
* **Ranking**: For a simple chronological feed, sort by timestamp. For production systems, a ranking service can reorder posts based on relevance, engagement, or personalization.

This design is the foundation used in many social networking platforms and can be extended with features such as recommendations, promoted content, stories, notifications, and machine learning–based feed ranking without changing the core architecture.
