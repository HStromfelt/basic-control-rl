import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="cmfs",
    version="0.0.1",
    author="Harald Str\"omfelt",
    author_email="harrystromfelt@gmail.com",
    description=(""),
    license="BSD",
    keywords="",
    url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=['scipy', 'sklearn', 'configparser', 'simplejson', 'cson', 'h5py', 'tqdm', 'six'],
    data_files=[
        # ('./faceKit/data/', ['./faceKit/data/faceKit_ResNet50.h5','./faceKit/data/mean_shape.h5', './faceKit/data/shape_predictor_68_face_landmarks.dat']),
        ],
    packages=find_packages(),
    entry_points={}
#        'console_scripts': [
#            'confer-to-images = supernn.sub.module:main',
#            ]
#        }
)
