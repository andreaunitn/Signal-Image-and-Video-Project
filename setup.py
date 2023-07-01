from setuptools import setup, find_packages

setup(name='open-reid',
      version='0.2.1',
      description='Implementation of a Deep Learning Library for Person Re-identification following the work done in "Bag of Tricks and A Strong Baseline for Deep Person Re-identification"',
      author='Andrea Tomasoni, Michele Lamon',
      author_email='andrea.tomasoni-1@studenti.unitn.it, michele.lamon@studenti.unitn.it',
      url='https://github.com/andreaunitn/Progetto-Signal-Image-and-Video',
      license='MIT',
      install_requires=[
          'numpy', 'scipy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow',
          'scikit-learn', 'metric-learn'],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme'],
      },
      packages=find_packages(),
      keywords=[
          'Person Re-identification',
          'Computer Vision',
          'Deep Learning',
      ])
