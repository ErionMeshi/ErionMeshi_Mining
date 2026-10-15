"""Test lokal: kontrollon qe udp.py + candidates.txt punojne dhe vlereson zgjidhjen shembull."""
import numpy as np
import udp as U

p = U.udp

lb, ub = p.get_bounds()
print("Dimensione:", len(lb), "| int vars:", p.get_nix(), "| ineq:", p.get_nic(), "| eq:", p.get_nec())

x = p.example()
f = p.fitness(x)
print("\nZgjidhja shembull")
print("objektivi (me i vogel = me mire):", f[0])
print("kufizimi i barazise (duhet 0):   ", f[1])
print("kufizimet e pabarazise max (duhet <= 0):", max(f[2:]))

print()
p.pretty(x)
