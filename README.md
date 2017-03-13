# Hierarchical Statistical Semantic Realization/Translation #

This repository contains the full HSST pipeline from preprocessing to decoding. Consequently, it has several dependencies. Their installation steps are listed below.

## Installation ##

* Clone hsst repository
```
#!bash

git clone git@bitbucket.org:matichorvat/hsst.git
```

* (Optional) Set up virtual environment
```
#!bash

cd hsst
virtualenv venv
source venv/bin/activate
```

* Install the package
```
#!bash

pip install -e .
```

### Preprocessing ###

* ACE
* pydmrs

### Decoding ###

The decoding part of the pipeline relies heavily on the pyfst - a Python interface to OpenFST. pyfst in its current version requires OpenFST 1.5.0 to be installed.

#### OpenFST ####

* Prepare directories
```
#!bash

mkdir -p openfst-1.5.0
cd openfst-1.5.0
export PREFIX=$PWD
echo $PREFIX
```

* Download OpenFST
```
#!bash

mkdir -p openfst
cd openfst
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.5.0.tar.gz
tar xzf openfst-1.5.0.tar.gz
```

* Compile OpenFST (note: CXXFLAGS are only required if compiling statically so that pyfst can be used without runtime linking - solving the problem of running the code on an older cluster)
```
#!bash
rm -rf objdir
mkdir objdir
cd objdir/
../openfst-1.5.0/configure --prefix=$PREFIX \
                --enable-bin \
                --enable-compact-fsts \
                --enable-const-fsts \
                --enable-far \
                --enable-lookahead-fsts \
                --enable-pdt \
                --enable-static \
                --enable-ngram-fsts \
                CXXFLAGS='-static-libstdc++ -static-libgcc -fPIC'
make -j 4
make install
```

* Test the OpenFST installation by running one of the command line tools
```
#!bash

$PREFIX/bin/fstcompose --help
```

#### pyfst ####

* Set up environment variables so that pyfst can access OpenFST
```
#!bash

echo "export OPENFSTDIR=$PREFIX" >> ~/.bashrc
echo 'export CPLUS_INCLUDE_PATH=$OPENFSTDIR/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$OPENFSTDIR/lib:$LIBRARY_PATH
export LIBRARY_PATH=$OPENFSTDIR/lib/fst:$LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENFSTDIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENFSTDIR/lib/fst:$LD_LIBRARY_PATH
export PATH=$OPENFSTDIR/bin:$PATH
export PYTHONPATH=$OPENFSTDIR/lib/python2.7/site-packages:$PYTHONPATH' >> ~/.bashrc
```

* Reload .bashrc file and test the environment variables 
```
#!bash

source  ~/.bashrc
echo $OPENFSTDIR
```

* Clone pyfst2 repository
```
#!bash

git clone git@github.com:matichorvat/pyfst2.git
```

* Install pyfst (make sure virtualenv is active if you are using it)
```
#!bash

cd pyfst2
CPLUS_INCLUDE_PATH=$OPENFSTDIR/include python setup.py install
```

* If you need statically linked pyfst library, modify the setup file:
```
extra_compile_args = ['-std=c++0x', '-static']
extra_link_args = ['-static-libstdc++', '-static-libgcc']

#libraries['fst'],
extra_objects=['/home/blue1/mh693/installs/openfst-1.5.0/lib/libfst.a', '/home/blue1/mh693/installs/openfst-1.5.0/lib/libfstscript.a'],
```
