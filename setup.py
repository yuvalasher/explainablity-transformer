from setuptools import setup

setup(
    name='exp_transformers',
    version='0.1.0',
    author='Yuval Asher',
    author_email='asheryuvala@gmail.com',
    packages=['config', 'data', 'datasets', 'evaluation', 'feature_extractor', 'main', 'models', 'nlp', 'pickles',
              'research', 'utils'],
    scripts=[],
    install_requires=[
        "zmq"
    ],
)
