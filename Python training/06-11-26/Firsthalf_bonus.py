sales = [12000, 0, 8500, 15000, -200, 9800]

total = 0
target = 30000
i = 0

while i < len(sales):

    amount = sales[i]

    if amount <= 0:
        print("Skipped invalid value", amount)
        i += 1
        continue

    total += amount
    print("Added", amount, "total", total)

    if total > target:
        break

    i += 1

if total > target:
    print("Target reached. Stopped at", total)
elif total == target:
    print("Target reached exactly at", total)
else:
    print("Target not reached. Total:", total)
