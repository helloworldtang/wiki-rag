# Redis常用数据结构及应用场景

## String（字符串）

最基础的数据类型，最大512MB。可用于：缓存、计数器、分布式锁、Session存储。

## Hash（哈希）

键值对集合，适合存储对象。如用户信息：`HSET user:1 name "张三" age 30`。比String+JSON更省内存。

## List（列表）

有序可重复。可用于：消息队列（LPUSH+BRPOP）、最新消息排行、时间线。

## Set（集合）

无序不重复。可用于：标签、共同好友（SINTER）、抽奖（SRANDMEMBER）、去重。

## Sorted Set（有序集合）

每个元素关联一个score，按score排序。可用于：排行榜、延时队列、滑动窗口限流。

## 选择原则

- 需要缓存对象 → Hash
- 需要排序 → Sorted Set
- 需要去重 → Set
- 需要队列 → List
- 其他 → String
