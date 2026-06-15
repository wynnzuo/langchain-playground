try :
    import redis
    print(f"Redis version: {redis.__version__}")
except ImportError :
    print("redis module not found. Please install it using 'pip install redis' to use Redis functionalities.")