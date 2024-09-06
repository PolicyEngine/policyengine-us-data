from setuptools import setup, find_packages

setup(
    name="policyengine_us_data",
    version="1.0.0",
    author="PolicyEngine",
    author_email="hello@policyengine.org",
    description="A package to create representative microdata for the US.",
    long_description=open("README.md").read(),
    packages=find_packages(),
    include_package_data=True,  # Will read MANIFEST.in
    python_requires=">=3.6",
    install_requires=[
        "policyengine_core",
        "tables",
        "survey_enhance",
        "torch",
        "requests",
        "tqdm",
        "tabulate",
        "tables",
    ],
    extras_require={
        "dev": [
            "black",
            "pytest",
            "policyengine_us==1.71.1",
            "streamlit",
        ],
    },
)
