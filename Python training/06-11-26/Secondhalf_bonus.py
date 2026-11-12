
# Bonus Challenge: __str__() method
class InsufficientFundsError(Exception):
    pass
class Account:

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(self.owner, "deposited", amount, "-> balance", self.balance)

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount
        print(self.owner, "withdrew", amount, "-> balance", self.balance)

    def show(self):
        print("Owner:", self.owner)
        print("Balance:", self.balance)

    # Bonus: String representation
    def __str__(self):
        return f"Owner: {self.owner}, Balance: {self.balance}"


# Create account
account = Account("Arun", 1000)

# Print account object
print(account)

# Deposit money
account.deposit(500)

# Print updated account
print(account)
