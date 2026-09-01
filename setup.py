'''
The setup.py file is a essential part of packaging and distributed Python Project .
It is used by setup tools to define the configuration of your project , such as meta data , dependencies and more
'''

from setuptools import find_packages , setup
from typing import List


def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    """

    requirement_lst: List[str] = []

    try :
        with open("requirements.txt" , "r") as file :
            ## read lines from line
            lines = file.readlines()
            ## process each line
            for line in lines :
                requirement = line.strip() 
                ## ignore the empty lines and -e .
                if requirement and requirement != "-e ." :
                    requirement_lst.append(requirement)
            
    except FileNotFoundError :
        print("My requirement.txt is not found")
    
    return requirement_lst


setup (
    name="Network Security" ,
    version="1.0.0" ,
    author="Garv Agrawal" ,
    author_email="garvagrawal9084@gmail.com" ,
    packages=find_packages() ,
    install_requires = get_requirements()
)
    