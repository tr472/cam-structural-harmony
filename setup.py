from setuptools import setup, find_packages

setup(
    name="cam-structural-harmony",
    version="0.1.0",
    description="Modular pipeline for evaluating and harmonising structural MRI data across scanners",
    author="Cambridge Ageing and Neuroscience (Cam-CAN) PhD Project",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "nibabel>=4.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "neuroCombat-sklearn>=0.1",
        "intensity-normalization>=3.0",
        "anthropic>=0.30",
        "pyyaml>=6.0",
        "SimpleITK>=2.3",
        "pingouin>=0.5",
    ],
    entry_points={
        "console_scripts": [
            "cam-harmony=cam_harmony.run:main",
            "cam-harmony-qc=cam_harmony.qc_assistant:main",
        ]
    },
)
