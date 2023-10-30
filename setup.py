from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name = 'AssetManagementSystem',
    version='0.0.1',
    description='Platform to manage and maintain assets.',
    keywords=['management', 
              'quality control;', 
              'decision-making',
              'performance indicators', 
              'degradation model', 
              'Markov'],
    author = 'Maurício Bonatte',
    author_email='mbonatte@ymail.com',
    url = 'https://github.com/mbonatte/AssetManagementSystem',
    license='GPL-3.0',
    long_description=long_description,
    
    # Dependencies
    install_requires=['numpy', 
                      'pymoo',
                      'scipy'],
    
    # Packaging
    packages =['ams',
               'ams.prediction',
               'ams.performance',
               'ams.optimization'],
    
)