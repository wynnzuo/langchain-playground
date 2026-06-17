"""
TypedDict 与 Pydantic BaseModel 对比
======================================
演示 Python TypedDict（类型提示）与 Pydantic BaseModel（运行时校验）
在字段定义、注解方式和验证能力上的关键区别。
"""

from typing import TypedDict, Annotated
from pydantic import BaseModel, Field, ValidationError

# ==========================================
# 方式 1：TypedDict — 类型提示，无运行时校验
# ==========================================
print("=" * 60)
print("1️⃣  TypedDict — 类型提示，无运行时校验")
print("=" * 60)

# Annotated 仅提供类型标注，不影响实际行为
age = Annotated[int, "年龄必须是一个整数，表示被问答对象的年龄。"]
num = Annotated[int, Field(gt=0, le=100, description="数字必须大于0且小于等于100")]


class Person(TypedDict):
    """TypedDict 仅提供静态类型提示，不验证实际值"""
    name: str
    age: age
    year: int


# 即使传入错误类型值，TypedDict 也不会报错（仅在 mypy/pyright 中提示）
p = Person(name="Alice", age=30, year=1990)
print(f"Person: {p}")

# ==========================================
# 方式 2：Pydantic BaseModel — 运行时类型校验
# ==========================================
print("\n" + "=" * 60)
print("2️⃣  Pydantic BaseModel — 运行时类型校验 + 字段约束")
print("=" * 60)


class Man(BaseModel):
    """BaseModel 会在创建实例时校验类型和约束"""
    name: str
    age: age
    year: num  # Field(gt=0, le=100) 约束 year 必须 >0 且 ≤100


try:
    man = Man(name="Bob", age=25, year=2020)
    print(f"Man: {man}")
except ValidationError as e:
    print(f"Validation error: {e}")
