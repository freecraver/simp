import setuptools

with open("README.md", "r") as f:
    readme_desc = f.read()

with open("requirements.txt", "r") as f:
    dependencies = [pkg for pkg in f.read().splitlines() if len(pkg) > 2 and not pkg.startswith('#')]
    print(",".join(dependencies))

setuptools.setup(
    name='zimp',
    version='0.0.1',
    author='Martin Freisehner',
    license='apache-2.0',
    author_email='mfreisehner@gmail.com',
    long_description=readme_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/freecraver/simp',
    packages=setuptools.find_packages(),
    install_requires=dependencies,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License'
    ]
)