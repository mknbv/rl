import setuptools


setuptools.setup(
    name="rl",
    version="0.0.1",
    author="Michael Konobeev",
    author_email="konobeev.michael@gmail.com",
    url="https://github.com/MichaelKonobeev/rl",
    descriptions=(
        "Modular and flexible deep reinforcement learning package "
        "on top of tensorflow"
    ),
    packages=setuptools.find_packages(),
    install_requires=[
      "gym",
      "numpy",
      "tqdm",
      "opencv-python"
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
