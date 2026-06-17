"""
Redis 环境检查
================
检查当前环境是否已安装 redis 模块及其版本。
可作为其他 Redis 相关 Demo 的前置检查。
"""

try:
    import redis
    print(f"✅ Redis 模块已安装，版本: {redis.__version__}")
except ImportError:
    print("❌ redis 模块未安装。请执行: pip install redis")
