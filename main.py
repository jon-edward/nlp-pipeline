import uuid

from records import RecordStore


mapping1 = RecordStore()
first_key = uuid.uuid4().hex
second_key = uuid.uuid4().hex
mapping1[first_key] = {"something": "else"}
mapping1[uuid.uuid4().hex] = {"something2": "else2"}
mapping1[second_key] = {"something3": "else3"}

subview = mapping1.slice(keys={first_key, second_key})

for key in subview:
    print(key)

for key in subview.keys():
    print(key)

for value in subview.values():
    print(value)

for key, value in subview.items():
    print(key, value)
