from setuptools import setup, find_packages

setup(
    name="ai_interviewer_pm",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
