from setuptools import setup

setup(
    name='diffusion-utils',
    version='0.1.0',
    description='various useful layers for doing diffusion on image data',
    url='https://github.com/GallagherCommaJack/diffusion-utils',
    author='Jack Gallagher',
    author_email='jack@gallabytes.com',
    license='MIT',
    packages=['diffusion-utils'],
    install_requires=[
        'einops>=0.3',
        'torch>=1.10',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
