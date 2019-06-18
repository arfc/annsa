from setuptools import setup, find_packages
PACKAGES = find_packages()

with open("README.md", "r") as fh:
    long_description = fh.read()

opts = dict(name='annsa',
            maintainer='Mark Kamuda',
            maintainer_email='kamuda1@illinois.edu',
            description='Neural networks applied to gamma-ray spectroscopy',
            long_description=long_description,
            url='https://github.com/arfc/annsa',
            license='BSD 3',
            author='Mark Kamuda',
            author_email='kamuda1@illinois.edu',
            version='0.1dev',
            packages=find_packages(),
            install_requires=['tensorflow', 'numpy', 'scipy'])


if __name__ == '__main__':
    setup(**opts)
