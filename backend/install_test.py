# install_test.py
print("=" * 60)
print("LIBRARY INSTALLATION TEST")
print("=" * 60)

libraries = [
    "flask",
    "flask-cors",
    "opencv-python",
    "numpy",
    "mediapipe",
    "tensorflow",
    "mysql-connector-python",
    "Pillow",
    "joblib",
    "scikit-learn",
    "matplotlib"
]

print("Will install these libraries:")
for i, lib in enumerate(libraries, 1):
    print(f"{i}. {lib}")

print("\n" + "=" * 60)
input("Press Enter to start installation (this may take 10-15 minutes)...")