import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

data = {
    "sleep_hours": np.random.uniform(4, 9, n),
    "sleep_quality": np.random.randint(1, 6, n),
    "stress_level": np.random.randint(1, 6, n),
    "water_intake": np.random.uniform(0.8, 3.5, n),
    "diet_type": np.random.randint(0, 3, n),
    "screen_time": np.random.uniform(2, 10, n),
    "exercise_minutes": np.random.randint(0, 61, n),
    "skincare_routine": np.random.randint(0, 3, n),
    "alcohol_smoking": np.random.randint(0, 2, n)
}

df = pd.DataFrame(data)

score = (
    100
    - (df["stress_level"] * 4)
    - ((7 - df["sleep_hours"]) * 5)
    - ((2.5 - df["water_intake"]) * 10)
    - (df["diet_type"] == 0) * 10
    - (df["screen_time"] > 7) * 5
    + (df["exercise_minutes"] >= 30) * 10
    + (df["skincare_routine"] == 2) * 10
)

df["skin_health_score"] = score.clip(0, 100).astype(int)

df["dry_skin_level"] = np.where(
    df["water_intake"] < 1.5, 2,
    np.where(df["water_intake"] < 2.0, 1, 0)
)

df.to_csv("skin_health_dataset.csv", index=False)
print("Dataset created successfully!")
