# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '1.8.5'
major           = '1'
minor           = '8'
patch           = '5'
rc              = '0'
istaged         = False
commit          = '8e1712a71057e3ca06a2927ceec51c18cb171fdc'
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
