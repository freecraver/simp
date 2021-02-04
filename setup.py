import setuptools

with open ("README.md", "r") as f:
    readme_desc = f.read()

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
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License'
    ]
)