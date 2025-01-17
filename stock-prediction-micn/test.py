import os


# def custom_generator():
#     # list_number = []
#     for i in range(1,11):
#         if i % 2 == 0:
#             yield i
#     # return list_number

def custom_decorator(func, a):
    list_odd = []
    for i in range(0,11):
        if i % 2 != 0:
            list_odd.append(i)
    print("I am a decorator")
    print(list_odd)
    return func


@custom_decorator
def custom_list():
    list_number = []
    for i in range(1,11):
        if i % 2 == 0:
            list_number.append(i)
    return list_number

print(custom_list())

class OOP:
    def __init__(self, a):
        self.a = a

obj1 = OOP(0)
obj2 = OOP(1)
list_obj = [obj1, obj2]

for obj in list_obj:
    print(obj.a)

a = 0.0
b = 0.0
