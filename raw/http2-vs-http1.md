# HTTP/2 vs HTTP/1.1

## 核心改进

### 多路复用
HTTP/1.1：每个请求需要一个TCP连接（或用keep-alive复用，但必须排队）。
HTTP/2：一个TCP连接上并行传输多个请求/响应，用stream ID区分。

### 头部压缩
HTTP/1.1：每次请求都要发送完整的Header（包括Cookie等）。
HTTP/2：HPACK算法压缩头部，维护动态表，重复的header只发送索引号。

### 服务端推送
Server可以主动向客户端推送资源，不需要客户端显式请求。

### 二进制协议
HTTP/1.1是文本协议，HTTP/2是二进制帧（Frame），解析更快、更紧凑。

## 仍然存在的问题

HTTP/2基于TCP，存在队头阻塞（Head-of-Line Blocking）——一个TCP包丢失会阻塞所有stream。这就是HTTP/3（QUIC/UDP）要解决的问题。

## 迁移建议

大多数场景，Nginx配置 `http2 on` 就够了。不需要改后端代码。
