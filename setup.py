from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name="plotly3d",
    version="0.4.8",
    packages=find_packages(),
    install_requires=[
        "plotly",
        "scikit-learn",
        "pandas",
        "numpy",
    ],
    author="Xingzhi Sun",
    author_email="xingzhi.sun@yale.edu",
    description="A helper for ploting scatter plots and trajectory plots with plotly",
    keywords="plotly, plotly 3d, 3d plotting, 3d scatter, 3d trajectory",
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/xingzhis/plotly3d',
)
