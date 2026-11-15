a = 10
b = 10.5
c = "Hello"
d = [1, 2, 3]
e = {"name": "Arun"}
f = {1, 2, 3}
g = (10, 20, 30)

print(a, type(a))
print(b, type(b))
print(c, type(c))
print(d, type(d))
print(e, type(e))
print(f, type(f))
print(g, type(g))

sales = [12000, 0, 8500, 15000, -200, 9800]

total = 0

for amount in sales:

    if amount <= 0:
        print("Skipped invalid value", amount)
        continue

    total = total + amount
    print("Added", amount, "total", total)

    if total > 30000:
        break
if total > 30000:
    print("Target reached. Stopped at", total)
elif total == 30000:
    print("Target reached exactly at", total)
else:
    print("Target not reached. Total:", total)