prices = [500, 1200, 800, 2500, 300]

def apply_discount(price, discount=10):
    return round(price - (price * discount / 100))


print("apply_discount(1000) ->", apply_discount(1000))
print("apply_discount(1000, 25) ->", apply_discount(1000, 25))


def total(*args):
    return sum(args)


print("total(5, 10, 15) ->", total(5, 10, 15))


def describe(**kwargs):
    for key, value in kwargs.items():
        print(key, ":", value)


describe(name="Raja", age=22, role="Developer")


above_1000 = [price for price in prices if price > 1000]

print("Above 1000:", above_1000)


with_gst = list(map(lambda price: round(price * 1.18), prices))

print("With GST:", with_gst)


below_1000 = list(filter(lambda price: price < 1000, prices))

print("Below 1000:", below_1000)
