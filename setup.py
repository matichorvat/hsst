from setuptools import setup

setup(
    name="hsst",
    version="0.1",
    author="Matic Horvat",
    description="Hierarchical Statistical Semantic Translation/Realization software.",
    packages=['hsst'],
    install_requires=[
        'Cython==0.23.4',
        'MarkupSafe==0.23',
        'argparse==1.2.1',
        'backports.ssl-match-hostname==3.4.0.2',
        'certifi==14.05.14',
        'decorator==3.4.0',
        'networkx==1.9.1',
        'odict==1.5.1',
        'wsgiref==0.1.2',
        'langdetect==1.0.3',
        'six==1.9.0',
        'python-Levenshtein==0.12.0',
        'sarge==0.1.4',
        'PyYAML==3.11',
        'pystache==0.5.4',
        'distribute==0.7.3'
    ],
    dependency_links=[
        'git+ssh://git@github.com/matichorvat/pydelphin.git#egg=delphin'
    ]
)
