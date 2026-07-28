from setuptools import setup, find_packages

setup(
    name="h2s-corrosion-prediction",
    version="0.1",
    description="ML-based H2S corrosion prediction and remaining useful life estimation",
    author="Ing. Kelvin Cabrera",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "h2s-train=train:main",
            "h2s-serve=app:main",
        ],
    },
)
