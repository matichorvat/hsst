This guide describes how to correctly install Python on the Cambridge Engineering cluster.

## 1. Install OpenSSL (shared):
```
CFLAGS="-fPIC"; ./config shared --openssldir=/home/miproj/mphil.acs.oct2012/mh693/installs/openssl-1.0.1t
make depend
make
make test
make install
```

## 2. Modify Python installation setup.py:
```
        mydir = '/home/miproj/mphil.acs.oct2012/mh693/downloads/openssl-1.0.1t'

        # Detect SSL support for the socket module (via _ssl)
        search_for_ssl_incs_in = [
                              '/usr/local/ssl/include',
                              '/usr/contrib/ssl/include/',
                              os.path.join(mydir, 'include')
                             ]
        ...
        
        ssl_libs = find_library_file(self.compiler, 'ssl',lib_dirs,
                                     ['/usr/local/ssl/lib',
                                      '/usr/contrib/ssl/lib/',
                                      mydir
                                     ] )
```

## 3. Install Python:
```
./configure --prefix=/home/miproj/mphil.acs.oct2012/mh693/installs/python-2.7.11
make
make install
```
After `make`, ensure that `_ssl` is not listed among missing modules.

## 4. Add Python installation to path:
Add the following to ~/.bashrc:
```
PATH=/home/miproj/mphil.acs.oct2012/mh693/installs/python-2.7.11/bin:$PATH
```
and source it:
```
source ~/.bashrc
```

## 4. Install pip and virtualenv:
If pip is not installed with Python, install it first:
```
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```

After, install virtualenv:
```
python -m pip install virtualenv
```

## 5. Setup a virtualenv where desired:
```
virtualenv venv
source venv/bin/activate
```

## References:
https://medium.com/@demianbrecht/compiling-cpython-with-a-custom-openssl-c9b668438517#.lh57y1tsv

https://gist.github.com/eddy-geek/9604982
