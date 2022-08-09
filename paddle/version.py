# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '2.0.2'
major           = '2'
minor           = '0'
patch           = '2'
rc              = '0'
istaged         = False
commit          = '5c7ad3bcb482dd93ea21b3b66b5a7ff964e0f239'
with_mkl        = 'ON'

def show():
    if istaged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)

def mkl():
    return with_mkl
