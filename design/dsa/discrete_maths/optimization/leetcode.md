 let’s directly **map optimization techniques in discrete mathematics → real LeetCode problems**.

---

# 🔑 LeetCode Problems by Optimization Technique

## 1. **Greedy Algorithms**

* [55. Jump Game](https://leetcode.com/problems/jump-game/)
* [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)
* [134. Gas Station](https://leetcode.com/problems/gas-station/)
* [135. Candy](https://leetcode.com/problems/candy/)
* [502. IPO](https://leetcode.com/problems/ipo/)
* [406. Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)
* [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

---

## 2. **Dynamic Programming (DP)**

* [322. Coin Change](https://leetcode.com/problems/coin-change/)
* [518. Coin Change II](https://leetcode.com/problems/coin-change-ii/)
* [494. Target Sum](https://leetcode.com/problems/target-sum/)
* [416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
* [1049. Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/)
* [139. Word Break](https://leetcode.com/problems/word-break/)
* [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

---

## 3. **Divide & Conquer**

* [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) (Kadane’s is DP, but divide & conquer also works)
* [169. Majority Element](https://leetcode.com/problems/majority-element/)
* [493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/)
* [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/)
* [218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

---

## 4. **Graph Optimization**

* **Shortest Path**

  * [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/) → Dijkstra
  * [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) → DP + BFS
  * [1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/) → Dijkstra
* **Minimum Spanning Tree**

  * [1135. Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) → Kruskal/Prim
  * [1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)
* **Max Flow / Matching**

  * [1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)
  * [1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/) (reachability)

---

## 5. **Integer / Linear Programming Style Problems**

* [1235. Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/) → weighted interval scheduling (DP with binary search).
* [630. Course Schedule III](https://leetcode.com/problems/course-schedule-iii/) → greedy + heap scheduling.
* [871. Minimum Number of Refueling Stops](https://leetcode.com/problems/minimum-number-of-refueling-stops/) → greedy + heap.
* [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/) → DP with transaction limits.

---

## 6. **Approximation / Hard Problems**

* [847. Shortest Path Visiting All Nodes](https://leetcode.com/problems/shortest-path-visiting-all-nodes/) → TSP-like, DP with bitmask.
* [980. Unique Paths III](https://leetcode.com/problems/unique-paths-iii/) → backtracking optimization.
* [691. Stickers to Spell Word](https://leetcode.com/problems/stickers-to-spell-word/) → NP-hard, solved with DP + memoization.

---

✅ **Summary**:

* **Greedy** → Jump Game, Gas Station, IPO.
* **DP** → Coin Change, LIS, Partition Equal Subset Sum.
* **Divide & Conquer** → Maximum Subarray, Reverse Pairs.
* **Graph Optimization** → Network Delay Time, Min Cost to Connect Points.
* **Scheduling / ILP style** → Job Scheduling, Course Schedule III.
* **Hard (Approximation/Bitmask)** → TSP-like, Stickers to Spell Word.

---

👉 Do you want me to also prepare this as a **markdown table** (Technique → Problems with links), like a **cheat sheet for quick lookup** during contests?
