class InsufficientFundsError(Exception):
    pass

class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{self.owner} deposited {amount} -> balance {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(f"Insufficient funds: balance {self.balance}, requested {amount}")
        self.balance -= amount
        print(f"{self.owner} withdrew {amount} -> balance {self.balance}")

    def show(self):
        print(f"Owner: {self.owner}, Balance: {self.balance}")

# Test Task 1
arun = Account("Arun", 1000)
try:
    arun.deposit(500)
    arun.withdraw(2000)
except InsufficientFundsError as e:
    print(f"Error: {e}")
finally:
    print("Transaction finished")


### Task 2: Inheritance and Overriding
class InsufficientFundsError(Exception):
    pass

class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

class SavingsAccount(Account):
    def __init__(self, owner, balance=0, min_balance=500):
        super().__init__(owner, balance)
        self.min_balance = min_balance

    def withdraw(self, amount):
        if self.balance - amount < self.min_balance:
            raise InsufficientFundsError(f"Insufficient funds: balance {self.balance}, requested {amount}")
        self.balance -= amount
        print(f"{self.owner} withdrew {amount} -> balance {self.balance}")