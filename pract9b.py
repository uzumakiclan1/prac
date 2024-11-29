#Tipping Problem fuzzy logic

# pip install scikit-fuzzy
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Define the universe of discourse for tip (0 to 25)
tip = np.arange(0, 26, 1)

# Fuzzy membership functions for low, medium, and high tips
low = fuzz.trimf(tip, [0, 0, 10])       # Triangular membership function for low tip
medium = fuzz.trimf(tip, [5, 15, 25])   # Triangular membership function for medium tip
high = fuzz.trimf(tip, [15, 25, 25])    # Triangular membership function for high tip

# Plot the membership functions
plt.figure()
plt.plot(tip, low, 'b', label='low')
plt.plot(tip, medium, 'orange', label='medium')
plt.plot(tip, high, 'g', label='high')
plt.title("Membership Functions for Tip")
plt.xlabel("Tip ($)")
plt.ylabel("Membership")
plt.legend()

# Use fuzzy logic to calculate suggested tip
# For demonstration purposes, let's assume the bill amount is 20
bill_amount = 20

# Fuzzify the input (20)
low_membership = fuzz.interp_membership(tip, low, bill_amount)
medium_membership = fuzz.interp_membership(tip, medium, bill_amount)
high_membership = fuzz.interp_membership(tip, high, bill_amount)

# Calculate the suggested tip based on the centroid method (defuzzification)
tip_suggested = fuzz.defuzz(tip, low * low_membership + medium * medium_membership + high * high_membership, 'centroid')

# Display the output
plt.axvline(x=tip_suggested, color='k', linestyle='--')
plt.text(tip_suggested + 0.5, 0.5, f"Suggested tip: {tip_suggested:.2f}", horizontalalignment='left')
plt.show()

# Print suggested tip
print(f"Suggested tip: {tip_suggested:.2f}")
