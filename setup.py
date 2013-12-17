from setuptools import setup


if __name__ == '__main__':

    DOCUMENTATION = open('README.rst').read()

    setup(
        name='beclab',
        packages=[
            'beclab',
            'beclab/integrator',
            ],
        provides=['beclab'],
        install_requires=[
            ('reikna >= 0.5.2'),
            ('progressbar2 >= 2.6.0'),
            ],
        extras_require=dict(
            pyopencl=["pyopencl >= 2013.1"],
            pycuda=["pycuda >= 2013.1"],
            test=[
                "pytest >= 2.3",
                "pytest-cov",
                "matplotlib >= 1.3.0",
                ],
            ),
        package_data={
            #'beclab': ['*.mako'],
            #'beclab/integrator': ['*.mako'],
            },
        version="0.1.0",
        author='Bogdan Opanchuk',
        author_email='bogdan@opanchuk.net',
        url='http://github.com/Manticore/beclab',
        description='BEC dynamics simulator',
        long_description=DOCUMENTATION,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Topic :: Scientific/Engineering',
            'Operating System :: OS Independent'
        ]
    )
