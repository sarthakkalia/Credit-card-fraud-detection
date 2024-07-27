from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(filepath:str)->List[str]:
    requirements=[]
    with open(filepath) as f:
        requirements=f.readlines()
        requirements=[requirement.replace('\n','') for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements

setup(
    name='Credit-card-fraud-detection',
    version='0.0.1',
    author='Sarthak',
    author_email='sarthakkalia45@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)