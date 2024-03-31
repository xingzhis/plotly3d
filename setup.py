from setuptools import setup, find_packages

setup(
    name="plotly3d",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        "plotly",
        "scikit-learn",  # Add other dependencies here
    ],
    author="Xingzhi Sun",
    author_email="xingzhi.sun@yale.edu",
    description="A minimal helper for ploting 3D scatter plots with plotly",
    keywords="plotly 3d plotting",
)
