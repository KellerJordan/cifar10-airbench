from setuptools import setup, find_packages

setup(
    name='airbench',
    version='0.1.1',
    author='Keller Jordan',
    author_email='kjordan4077@gmail.com',
    description='Utilities and baselines for fast neural network training on CIFAR-10',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KellerJordan/cifar10-airbench',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
