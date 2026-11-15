import numpy as np

marks = np.array([[70, 80, 90], [60, 75, 85], [88, 92, 79], [55, 65, 70]])

print(f"Shape: {marks.shape}")
print(f"Dtype: {marks.dtype}")
print(f"Size: {marks.size}")
slice_result = marks[:2, -1:]
print("Slice of first two students and last subject:\n", slice_result)

student_avg = marks.mean(axis=1)
subject_avg = marks.mean(axis=0)
print("Student averages:", student_avg)
print("Subject averages:", subject_avg)

marks_with_grace = marks + 5
print("With grace marks:\n", marks_with_grace)

weights = np.array([0.3, 0.3, 0.4])
weighted_scores = marks @ weights
print("Weighted scores:", weighted_scores)

reshaped_arr = np.arange(12).reshape(3, 4)
print("Reshaped array (3x4):\n", reshaped_arr)