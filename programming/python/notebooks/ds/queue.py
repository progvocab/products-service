# queue_module.py
from collections import deque
import heapq

# ------------------------
# Standard FIFO Queue
# ------------------------

class Queue:
    def __init__(self):
        self.queue = deque()

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.popleft()
        raise IndexError("Queue is empty")

    def peek(self):
        return self.queue[0] if not self.is_empty() else None

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)


# ------------------------
# Circular Queue
# ------------------------

class CircularQueue:
    def __init__(self, capacity):
        self.queue = [None] * capacity
        self.head = self.tail = -1
        self.capacity = capacity

    def enqueue(self, item):
        if (self.tail + 1) % self.capacity == self.head:
            raise OverflowError("Circular Queue is full")
        if self.head == -1:
            self.head = 0
        self.tail = (self.tail + 1) % self.capacity
        self.queue[self.tail] = item

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Circular Queue is empty")
        item = self.queue[self.head]
        if self.head == self.tail:
            self.head = self.tail = -1  # reset
        else:
            self.head = (self.head + 1) % self.capacity
        return item

    def is_empty(self):
        return self.head == -1

    def peek(self):
        if self.is_empty():
            return None
        return self.queue[self.head]


# ------------------------
# Deque (Double-Ended Queue)
# ------------------------

class Deque:
    def __init__(self):
        self.deque = deque()

    def add_front(self, item):
        self.deque.appendleft(item)

    def add_rear(self, item):
        self.deque.append(item)

    def remove_front(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.deque.popleft()

    def remove_rear(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.deque.pop()

    def is_empty(self):
        return len(self.deque) == 0

    def size(self):
        return len(self.deque)


# ------------------------
# Priority Queue (Min-Heap)
# ------------------------

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def enqueue(self, priority, item):
        heapq.heappush(self.heap, (priority, item))

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Priority Queue is empty")
        return heapq.heappop(self.heap)[1]

    def is_empty(self):
        return len(self.heap) == 0

    def peek(self):
        return self.heap[0][1] if not self.is_empty() else None
