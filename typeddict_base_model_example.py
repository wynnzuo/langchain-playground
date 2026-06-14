from typing import TypedDict, Annotated
from pydantic import BaseModel, Field, ValidationError

age = Annotated[int, "年龄必须是一个整数，表示被问答对象的年龄。"]
num = Annotated[int, Field(gt=0, le=100, description="数字必须大于0且小于等于100")]


class Person(TypedDict):
    name: str
    age: age
    year: int


p = Person(name="Alice", age=30, year=1990)
print(p)

class Man(BaseModel):
    name: str
    age: age
    year: num

try:  
    man = Man(name="Bob", age=25, year=2020)
    print(man)
except ValidationError as e:
    print(f"Validation error: {e}")