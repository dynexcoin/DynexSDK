from setuptools import setup

setup(
    name='dynex',
    version='0.1.5',    
    description='Dynex SDK Neuromorphic Computing',
    url='https://github.com/dynexcoin/DynexSDK',
    author='Dynex Developers',
    author_email='office@dynexcoin.org',
    license='GPLv3',
    packages=['dynex'],
    install_requires=['pycryptodome>=3.18.0',
                      'dimod>=0.12.10',
                      'tabulate>=0.9.0',
                      'tqdm>=4.65.0',
                      'ipywidgets>=8.0.7',
                      'numpy'
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',     
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English',
        'Topic :: System :: Distributed Computing',
    ],

    long_description = 'The Dynex SDK provides a neuromorphic Ising/QUBO sampler which can be called from any Python code. Developers and application developers already familiar with the Dimod framework, PyQUBO or the Ocean SDK will find it very easy to run computations on the Dynex neuromorphic computing platform: The Dynex Sampler object can simply replace the default sampler object which typically is used to run computations on, for example, the D-Wave system – without the limitations of quantum machines. The Dynex SDK is a suite of open-source Python tools for solving hard problems with neuromorphic computing which helps reformulate your application’s problem for solution by the Dynex computing platform. It also handles communication between your application code and the Dynex neuromorphic computing platform automatically.',
)


