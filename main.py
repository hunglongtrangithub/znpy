from pathlib import Path
import numpy as np


def main():
    """Create a sample .npy file with a 2D array for testing."""
    arr = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    pth = Path("test-data")
    pth.mkdir(exist_ok=True)
    np.save("test-data/plain.npy", arr)


if __name__ == "__main__":
    main()
