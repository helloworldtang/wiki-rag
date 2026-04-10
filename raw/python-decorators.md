# Python装饰器

装饰器是Python中最强大的语法糖之一，本质上是一个高阶函数，接收一个函数作为参数，返回一个新函数。

## 基本用法

```python
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
```

## 带参数的装饰器

需要三层嵌套：最外层接收装饰器参数，中间层接收被装饰函数，最内层是wrapper。

## functools.wraps

永远使用 `@functools.wraps(func)` 保留原函数的元信息（名称、文档字符串等）。

## 类装饰器

当装饰器需要维护状态时，用类实现更清晰。`__call__` 方法就是 wrapper。

## 常见陷阱

1. 忘记 `@wraps` 导致调试困难
2. 装饰器顺序：从下往上执行，但从上往下的装饰器先wrap
3. 被装饰后的函数签名丢失
