from setuptools import setup, find_packages

setup(
  name = 'dpfp-pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.3.0',
  license='BSD',
  description = 'DPFP - Pytorch',
  author = 'Ferris Kwaijtaal',
  author_email = 'ferris+gh@devdroplets.com',
  url = 'https://github.com/i404788/DPFP-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'efficient attention',
    'transformers'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.4',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

