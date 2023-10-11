import setuptools

class NoNumpy(Exception):
    pass

try:
    from numpy.distutils.core import Extension
    from numpy.distutils.core import setup
except ImportError:
    raise NoNumpy('Numpy Needs to be installed '
                  'for extensions to be compiled.')

with open("README.md", "r") as fh:
    long_description = fh.read()

fort_spec = Extension('prt_phasecurve.fort_spec', sources=['prt_phasecurve/fort_spec.f90'],
                                  extra_compile_args=["-O3", "-funroll-loops", "-ftree-vectorize", "-msse", "-msse2", "-m3dnow"])

setup(
    name='prt_phasecurve',
    version='v0.1.7',
    packages=setuptools.find_packages(),
    include_package_data=True,
    url='https://github.com/AaronDavidSchneider/prt_phasecurve',
    license='MIT',
    author='Aaron David Schneider',
    author_email='aarondavid.schneider@nbi.ku.dk',
    description='module that creates phasecurves using pRT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[fort_spec],
    install_requires=[
        "scipy>=1.7.0",
        "numpy",
        "petitRADTRANS>=2.2.1",
        "tqdm"
    ]
)
